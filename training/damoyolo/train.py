#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import argparse
import copy
import os
import datetime
import time

import torch
from loguru import logger
import mlflow
from torch.nn.parallel import DistributedDataParallel as DDP

from damo.apis import Trainer
from damo.config.base import parse_config
from damo.utils import synchronize, gpu_mem_usage


class MLflowTrainer(Trainer):
    """Trainer with MLflow logging capabilities"""
    
    def __init__(self, cfg, args, tea_cfg=None, is_train=True, enable_mlflow=False):
        super().__init__(cfg, args, tea_cfg, is_train)
        self.enable_mlflow = enable_mlflow
        self.mlflow_log_interval = cfg.miscs.print_interval_iters
        
    def train(self, local_rank):
        """Override train method to add MLflow logging"""
        from damo.utils import get_model_info
        
        logger.info('Model Summary: {}'.format(
            get_model_info(self.model, (640, 640))))

        # distributed model init
        from damo.detectors.detector import build_ddp_model
        self.model = build_ddp_model(self.model, local_rank)

        logger.info('Training start...')

        # ----------- start training ------------------------- #
        self.model.train()
        iter_start_time = time.time()
        iter_end_time = time.time()
        for data_iter, (inps, targets, ids) in enumerate(self.train_loader):
            cur_iter = self.start_iter + data_iter

            lr = self.lr_scheduler.get_lr(cur_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            inps = inps.to(self.device)
            targets = [target.to(self.device) for target in targets]

            model_start_time = time.time()

            if self.distill:
                outputs, fpn_outs = self.model(inps, targets, stu=True)
                loss = outputs['total_loss']
                with torch.no_grad():
                    fpn_outs_tea = self.tea_model(inps, targets, tea=True)
                import math
                distill_weight = (
                    (1 - math.cos(cur_iter * math.pi / len(self.train_loader)))
                    / 2) * (0.1 - 1) + 1

                distill_loss = distill_weight * self.feature_loss(
                    fpn_outs, fpn_outs_tea)
                loss += distill_loss
                outputs['distill_loss'] = distill_loss
            else:
                outputs = self.model(inps, targets)
                loss = outputs['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                import torch.nn as nn
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.grad_clip,
                                         norm_type=2)

            self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(cur_iter, self.model)

            iter_start_time = iter_end_time
            iter_end_time = time.time()

            outputs_array = {_name: _v.item() for _name, _v in outputs.items()}
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                model_time=iter_end_time - model_start_time,
                lr=lr,
                **outputs_array,
            )

            if cur_iter + 1 > self.total_iters - self.no_aug_iters:
                if self.mosaic_mixup:
                    logger.info('--->turn OFF mosaic aug now!')
                    self.train_loader.batch_sampler.set_mosaic(False)
                    self.eval_interval_iters = self.iters_per_epoch
                    self.ckpt_interval_iters = self.iters_per_epoch
                    self.mosaic_mixup = False

            # log needed information
            if (cur_iter + 1) % self.print_interval_iters == 0:
                left_iters = self.total_iters - (cur_iter + 1)
                eta_seconds = self.meter['iter_time'].global_avg * left_iters
                eta_str = 'ETA: {}'.format(
                    datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = 'epoch: {}/{}, iter: {}/{}'.format(
                    self.epoch + 1, self.total_epochs,
                    (cur_iter + 1) % self.iters_per_epoch,
                    self.iters_per_epoch)
                loss_meter = self.meter.get_filtered_meter('loss')
                loss_str = ', '.join([
                    '{}: {:.1f}'.format(k, v.avg)
                    for k, v in loss_meter.items()
                ])

                time_meter = self.meter.get_filtered_meter('time')
                time_str = ', '.join([
                    '{}: {:.3f}s'.format(k, v.avg)
                    for k, v in time_meter.items()
                ])

                logger.info('{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}'.format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter['lr'].latest,
                ) + (', size: ({:d}, {:d}), {}'.format(
                    inps.tensors.shape[2], inps.tensors.shape[3], eta_str)))
                
                # MLflow logging
                if self.enable_mlflow and local_rank == 0:
                    self._log_metrics_to_mlflow(cur_iter)
                
                self.meter.clear_meters()

            if (cur_iter + 1) % self.ckpt_interval_iters == 0:
                self.save_ckpt('epoch_%d' % (self.epoch + 1),
                               local_rank=local_rank)

            if (cur_iter + 1) % self.eval_interval_iters == 0:
                time.sleep(0.003)
                self.evaluate(local_rank, self.cfg.dataset.val_ann)
                self.model.train()
            synchronize()

            if (cur_iter + 1) % self.iters_per_epoch == 0:
                self.epoch = self.epoch + 1
                # Log epoch-level metrics to MLflow
                if self.enable_mlflow and local_rank == 0:
                    self._log_epoch_to_mlflow(self.epoch, cur_iter)

        self.save_ckpt(ckpt_name='latest', local_rank=local_rank)
    
    def _log_metrics_to_mlflow(self, step):
        """Log training metrics to MLflow"""
        try:
            # Log loss metrics
            loss_meter = self.meter.get_filtered_meter('loss')
            for name, meter in loss_meter.items():
                mlflow.log_metric(f'train/{name}', meter.avg, step=step)
            
            # Log learning rate
            if 'lr' in self.meter:
                mlflow.log_metric('train/learning_rate', self.meter['lr'].latest, step=step)
            
            # Log time metrics
            time_meter = self.meter.get_filtered_meter('time')
            for name, meter in time_meter.items():
                mlflow.log_metric(f'performance/{name}', meter.avg, step=step)
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def _log_epoch_to_mlflow(self, epoch, step):
        """Log epoch completion to MLflow"""
        try:
            mlflow.log_metric('epoch', epoch, step=step)
        except Exception as e:
            logger.warning(f"Failed to log epoch to MLflow: {e}")
    
    def evaluate(self, local_rank, val_ann):
        """Override evaluate to add MLflow logging"""
        from damo.apis.detector_inference import inference
        
        assert len(self.val_loader) == len(val_ann)
        if self.ema_model is not None:
            evalmodel = self.ema_model.model
        else:
            evalmodel = self.model
            if isinstance(evalmodel, DDP):
                evalmodel = evalmodel.module

        output_folders = [None] * len(val_ann)
        for idx, dataset_name in enumerate(val_ann):
            output_folder = os.path.join(self.output_dir, self.exp_name,
                                         'inference', dataset_name)
            if local_rank == 0:
                os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder

        # Run inference and collect results for MLflow logging
        for output_folder, dataset_name, data_loader_val in zip(
                output_folders, val_ann, self.val_loader):
            try:
                eval_results = inference(
                    evalmodel,
                    data_loader_val,
                    dataset_name,
                    device=self.device,
                    output_folder=output_folder,
                )
                
                # Log evaluation metrics to MLflow
                if self.enable_mlflow and local_rank == 0 and eval_results is not None:
                    self._log_eval_to_mlflow(eval_results, dataset_name)
            except Exception as e:
                logger.error(f"Evaluation failed for {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _log_eval_to_mlflow(self, eval_results, dataset_name):
        """Log evaluation results to MLflow"""
        try:
            # eval_results is a tuple: (results, coco_results)
            # results is a COCOResults object with a .results attribute
            if isinstance(eval_results, tuple) and len(eval_results) > 0:
                results = eval_results[0]
                
                # Check if results has a .results attribute (COCOResults object)
                if hasattr(results, 'results'):
                    results_dict = results.results
                else:
                    results_dict = results
                
                # Log COCO metrics to MLflow
                if isinstance(results_dict, dict):
                    for task, metrics in results_dict.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                mlflow.log_metric(
                                    f'eval/{dataset_name}/{task}/{metric_name}',
                                    float(value),
                                    step=self.epoch
                                )
                                logger.info(f"Logged eval/{dataset_name}/{task}/{metric_name}: {value:.4f}")
                    
                    logger.info(f"Successfully logged evaluation metrics for {dataset_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to MLflow: {e}")
            import traceback
            logger.warning(traceback.format_exc())


def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Damo-Yolo train parser')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='plz input your config file',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tea_config', type=str, default=None)
    parser.add_argument('--tea_ckpt', type=str, default=None)
    parser.add_argument('--mlflow_tracking_uri', type=str, 
                        default='http://192.168.0.100:5001',
                        help='MLflow tracking server URI')
    parser.add_argument('--experiment_name', type=str, 
                        default='damo_yolo_training',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, 
                        default=None,
                        help='MLflow run name')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    synchronize()
    
    # MLflow setup (only on main process)
    if args.local_rank == 0:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        
        # Start MLflow run
        run_name = args.run_name if args.run_name else f"run_{args.config_file.split('/')[-1].replace('.py', '')}"
        mlflow.start_run(run_name=run_name)
        
        logger.info(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
        logger.info(f"MLflow experiment: {args.experiment_name}")
        logger.info(f"MLflow run name: {run_name}")
    
    if args.tea_config is not None:
        tea_config = parse_config(args.tea_config)
    else:
        tea_config = None

    config = parse_config(args.config_file)
    config.merge(args.opts)

    # Log hyperparameters to MLflow (only on main process)
    if args.local_rank == 0:
        try:
            # Log basic training parameters
            mlflow.log_param('config_file', args.config_file)
            if args.tea_config:
                mlflow.log_param('tea_config', args.tea_config)
            if args.tea_ckpt:
                mlflow.log_param('tea_ckpt', args.tea_ckpt)
            
            # Log config parameters (safely)
            if hasattr(config, 'miscs'):
                if hasattr(config.miscs, 'exp_name'):
                    mlflow.log_param('exp_name', config.miscs.exp_name)
                if hasattr(config.miscs, 'output_dir'):
                    mlflow.log_param('output_dir', config.miscs.output_dir)
            
            if hasattr(config, 'train'):
                if hasattr(config.train, 'batch_size'):
                    mlflow.log_param('batch_size', config.train.batch_size)
                if hasattr(config.train, 'total_epochs'):
                    mlflow.log_param('total_epochs', config.train.total_epochs)
            
            if hasattr(config, 'optimizer'):
                if hasattr(config.optimizer, 'lr'):
                    mlflow.log_param('learning_rate', config.optimizer.lr)
                if hasattr(config.optimizer, 'momentum'):
                    mlflow.log_param('momentum', config.optimizer.momentum)
                if hasattr(config.optimizer, 'weight_decay'):
                    mlflow.log_param('weight_decay', config.optimizer.weight_decay)
                    
        except Exception as e:
            logger.warning(f"Failed to log some parameters to MLflow: {e}")

    # Create MLflowTrainer instead of regular Trainer
    enable_mlflow = (args.local_rank == 0)
    trainer = MLflowTrainer(config, args, tea_config, enable_mlflow=enable_mlflow)
    
    # Train and ensure MLflow run is ended properly
    if args.local_rank == 0:
        try:
            trainer.train(args.local_rank)
        finally:
            mlflow.end_run()
            logger.info("MLflow run ended")
    else:
        trainer.train(args.local_rank)


if __name__ == '__main__':
    main()
