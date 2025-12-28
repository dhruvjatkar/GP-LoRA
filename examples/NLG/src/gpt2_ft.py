#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools
import json
from collections import defaultdict

import torch
import random
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)


class TrainingMetrics:
    """
    Tracks comprehensive training metrics for comparing LoRA vs GP-LoRA.
    
    Metrics tracked:
    - Per-step time (average time per training step)
    - Steps to converge (steps to reach target loss/ppl thresholds)
    - Total training time
    - Loss/PPL progression over time
    """
    
    def __init__(self, target_losses=None, target_ppls=None):
        self.start_time = None
        self.step_times = []
        self.step_start_time = None
        
        # Loss/PPL history with timestamps
        self.history = []  # List of (step, time, loss, ppl)
        
        # Convergence tracking
        self.target_losses = target_losses or [3.0, 2.5, 2.0, 1.5]
        self.target_ppls = target_ppls or [20.0, 15.0, 10.0, 8.0, 6.0, 5.0]
        self.steps_to_loss = {}  # target_loss -> step when first reached
        self.time_to_loss = {}   # target_loss -> time when first reached
        self.steps_to_ppl = {}   # target_ppl -> step when first reached
        self.time_to_ppl = {}    # target_ppl -> time when first reached
        
        # Final metrics
        self.total_steps = 0
        self.total_time = 0
        self.final_loss = None
        self.final_ppl = None
        self.best_loss = float('inf')
        self.best_ppl = float('inf')
    
    def start_training(self):
        """Call at the start of training."""
        self.start_time = time.time()
    
    def start_step(self):
        """Call at the start of each training step."""
        self.step_start_time = time.time()
    
    def end_step(self, step, loss, ppl=None):
        """Call at the end of each training step."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if ppl is None:
            ppl = math.exp(loss) if loss < 100 else float('inf')
        
        self.history.append({
            'step': step,
            'time': elapsed,
            'loss': loss,
            'ppl': ppl
        })
        
        # Track best
        if loss < self.best_loss:
            self.best_loss = loss
        if ppl < self.best_ppl:
            self.best_ppl = ppl
        
        # Check convergence thresholds
        for target in self.target_losses:
            if target not in self.steps_to_loss and loss <= target:
                self.steps_to_loss[target] = step
                self.time_to_loss[target] = elapsed
        
        for target in self.target_ppls:
            if target not in self.steps_to_ppl and ppl <= target:
                self.steps_to_ppl[target] = step
                self.time_to_ppl[target] = elapsed
        
        self.total_steps = step
        self.final_loss = loss
        self.final_ppl = ppl
    
    def end_training(self):
        """Call at the end of training."""
        self.total_time = time.time() - self.start_time if self.start_time else 0
    
    def get_avg_step_time(self):
        """Get average time per step in milliseconds."""
        if not self.step_times:
            return 0
        return sum(self.step_times) / len(self.step_times) * 1000
    
    def get_steps_per_second(self):
        """Get training throughput."""
        if self.total_time == 0:
            return 0
        return self.total_steps / self.total_time
    
    def get_summary(self, gp_lora=False, args=None):
        """Get a summary dict of all metrics."""
        summary = {
            'method': 'GP-LoRA' if gp_lora else 'LoRA',
            'gp_lora': gp_lora,
            
            # Timing metrics
            'total_time_seconds': self.total_time,
            'total_time_minutes': self.total_time / 60,
            'avg_step_time_ms': self.get_avg_step_time(),
            'steps_per_second': self.get_steps_per_second(),
            
            # Training metrics
            'total_steps': self.total_steps,
            'final_loss': self.final_loss,
            'final_ppl': self.final_ppl,
            'best_loss': self.best_loss,
            'best_ppl': self.best_ppl,
            
            # Convergence metrics (steps to reach target)
            'steps_to_loss': self.steps_to_loss,
            'time_to_loss': self.time_to_loss,
            'steps_to_ppl': self.steps_to_ppl,
            'time_to_ppl': self.time_to_ppl,
        }
        
        # Add config info if available
        if args is not None:
            summary['config'] = {
                'lora_dim': getattr(args, 'lora_dim', None),
                'lora_alpha': getattr(args, 'lora_alpha', None),
                'lr': getattr(args, 'lr', None),
                'max_epoch': getattr(args, 'max_epoch', None),
                'train_batch_size': getattr(args, 'train_batch_size', None),
                'gp_mu': getattr(args, 'gp_mu', None) if gp_lora else None,
                'gp_eps': getattr(args, 'gp_eps', None) if gp_lora else None,
            }
        
        return summary
    
    def save(self, filepath, gp_lora=False, args=None):
        """Save metrics to JSON file."""
        summary = self.get_summary(gp_lora=gp_lora, args=args)
        summary['history'] = self.history
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self, gp_lora=False):
        """Print a formatted summary."""
        method = 'GP-LoRA' if gp_lora else 'LoRA'
        print('\n' + '=' * 80)
        print(f'TRAINING METRICS SUMMARY - {method}')
        print('=' * 80)
        
        print(f'\n[TIMING]')
        print(f'  Total training time:    {self.total_time:.2f}s ({self.total_time/60:.2f} min)')
        print(f'  Average step time:      {self.get_avg_step_time():.2f} ms')
        print(f'  Training throughput:    {self.get_steps_per_second():.2f} steps/sec')
        
        print(f'\n[FINAL METRICS]')
        print(f'  Total steps:            {self.total_steps}')
        print(f'  Final loss:             {self.final_loss:.4f}')
        print(f'  Final perplexity:       {self.final_ppl:.2f}')
        print(f'  Best loss:              {self.best_loss:.4f}')
        print(f'  Best perplexity:        {self.best_ppl:.2f}')
        
        print(f'\n[CONVERGENCE - Steps to reach target loss]')
        for target in sorted(self.target_losses, reverse=True):
            if target in self.steps_to_loss:
                print(f'  Loss <= {target:.1f}:  step {self.steps_to_loss[target]:>6d}  '
                      f'(time: {self.time_to_loss[target]:.1f}s)')
            else:
                print(f'  Loss <= {target:.1f}:  not reached')
        
        print(f'\n[CONVERGENCE - Steps to reach target PPL]')
        for target in sorted(self.target_ppls, reverse=True):
            if target in self.steps_to_ppl:
                print(f'  PPL <= {target:.1f}:   step {self.steps_to_ppl[target]:>6d}  '
                      f'(time: {self.time_to_ppl[target]:.1f}s)')
            else:
                print(f'  PPL <= {target:.1f}:   not reached')
        
        print('=' * 80 + '\n')


# Global metrics tracker (will be initialized in main)
training_metrics = None

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

# GP-LoRA (Gauge-Projected LoRA) arguments
parser.add_argument('--gp_lora', action='store_true', 
                    help='Enable GP-LoRA gauge projection after each optimizer step')

parser.add_argument('--gp_mu', default='auto', 
                    help='GP-LoRA imbalance ratio mu (default: auto = r/m)')

parser.add_argument('--gp_eps', type=float, default=1e-4, 
                    help='GP-LoRA regularization epsilon')

# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()
        
        # GP-LoRA: Apply gauge projection after optimizer step
        if args.lora_dim > 0 and getattr(args, 'gp_lora', False):
            # Parse mu value (can be 'auto' or a float)
            gp_mu = getattr(args, 'gp_mu', 'auto')
            if gp_mu != 'auto':
                try:
                    gp_mu = float(gp_mu)
                except ValueError:
                    gp_mu = 'auto'
            gp_eps = getattr(args, 'gp_eps', 1e-4)
            lora.gauge_project_model(_model, mu=gp_mu, eps=gp_eps)
        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args, 
    train_step=0, 
    epoch=0,
    metrics=None
):
    global training_metrics
    if metrics is not None:
        training_metrics = metrics
    
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        # Start step timing
        if training_metrics is not None:
            training_metrics.start_step()
        
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        ) 

        _lm_loss = _lm_loss.mean() 

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        
        # Record step metrics
        if training_metrics is not None and is_update:
            current_loss = avg_lm_loss.avg if avg_lm_loss.count > 0 else _lm_loss.item()
            current_ppl = math.exp(current_loss) if current_loss < 100 else float('inf')
            training_metrics.end_step(train_step, current_loss, current_ppl)
        
        if train_step % args.log_interval == 0: 
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            avg_step_ms = training_metrics.get_avg_step_time() if training_metrics else 0
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'
            
            # Add GP-LoRA indicator if enabled
            if args.gp_lora:
                log_str += ' [GP-LoRA]'

            if args.rank == 0: 
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    distributed_sync(args)
    return train_step


if __name__ == '__main__':
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))    

    lm_net = lm_net.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    # Initialize metrics tracking
    training_metrics = TrainingMetrics()
    
    try:
        train_step = 0
        
        # Start metrics tracking
        training_metrics.start_training()
        if args.rank == 0:
            method = 'GP-LoRA' if args.gp_lora else 'LoRA'
            print(f'\n{"="*80}')
            print(f'Starting training with {method}')
            print(f'{"="*80}\n')
        
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch, metrics=training_metrics
            )
            
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')
    
    # End metrics tracking and save results
    training_metrics.end_training()
    
    if args.rank == 0:
        # Print summary
        training_metrics.print_summary(gp_lora=args.gp_lora)
        
        # Save metrics to JSON
        metrics_file = os.path.join(args.work_dir, 'training_metrics.json')
        summary = training_metrics.save(metrics_file, gp_lora=args.gp_lora, args=args)
        print(f'Training metrics saved to: {metrics_file}')
        
        # Also save a simplified comparison-ready file
        comparison_file = os.path.join(args.work_dir, 'comparison_metrics.json')
        comparison_data = {
            'method': 'GP-LoRA' if args.gp_lora else 'LoRA',
            'total_time_seconds': summary['total_time_seconds'],
            'total_time_minutes': summary['total_time_minutes'],
            'avg_step_time_ms': summary['avg_step_time_ms'],
            'steps_per_second': summary['steps_per_second'],
            'total_steps': summary['total_steps'],
            'final_loss': summary['final_loss'],
            'final_ppl': summary['final_ppl'],
            'best_loss': summary['best_loss'],
            'best_ppl': summary['best_ppl'],
            'steps_to_ppl_10': summary['steps_to_ppl'].get(10.0),
            'steps_to_ppl_8': summary['steps_to_ppl'].get(8.0),
            'steps_to_ppl_6': summary['steps_to_ppl'].get(6.0),
            'time_to_ppl_10': summary['time_to_ppl'].get(10.0),
            'time_to_ppl_8': summary['time_to_ppl'].get(8.0),
            'time_to_ppl_6': summary['time_to_ppl'].get(6.0),
        }
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f'Comparison metrics saved to: {comparison_file}')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)
