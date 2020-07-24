#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import tensorpack.dataflow as df
import argparse
import time
import yaml
from datetime import datetime
from timm.models.skipnet import skip_v3
from torch.utils.tensorboard import SummaryWriter

has_apex = False

from timm.data.loader import Loader
from timm.data import Dataset, resolve_data_config

from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import torch
import torch.nn as nn
import torchvision.utils

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)') # TODO resize + crop
# parser.add_argument('--crop-pct', default=None, type=float,
#                     metavar='N', help='Input image center crop percent (for validation only)')
# parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
#                     help='Override mean pixel value of dataset')
# parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
#                     help='Override std deviation of of dataset')
# parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
#                     help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
# parser.add_argument('--jsd', action='store_true', default=False,
#                     help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
# parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
#                     help='learning rate cycle len multiplier (default: 1.0)')
# parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
#                     help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
# parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
#                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
# parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
#                     help='Color jitter factor (default: 0.4)')
# parser.add_argument('--aa', type=str, default=None, metavar='NAME',
#                     help='Use AutoAugment policy. "v0" or "original". (default: None)'),
# parser.add_argument('--aug-splits', type=int, default=0,
#                     help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
# parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
#                     help='Random erase prob (default: 0.)')
# parser.add_argument('--remode', type=str, default='const',
#                     help='Random erase mode (default: "const")')
# parser.add_argument('--recount', type=int, default=1,
#                     help='Random erase count (default: 1)')
# parser.add_argument('--resplit', action='store_true', default=False,
#                     help='Do not random erase first (clean) augmentation split')
# parser.add_argument('--mixup', type=float, default=0.0,
#                     help='mixup alpha, mixup enabled if > 0. (default: 0.)')
# parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
#                     help='turn off mixup after this epoch, disabled if 0 (default: 0)')
# parser.add_argument('--smoothing', type=float, default=0.1,
#                     help='label smoothing (default: 0.1)')
# parser.add_argument('--train-interpolation', type=str, default='random',
#                     help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
# parser.add_argument('--sync-bn', action='store_true',
#                     help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
# parser.add_argument('--dist-bn', type=str, default='',
#                     help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
# parser.add_argument('--split-bn', action='store_true',
#                     help='Enable separate BN layers per augmentation split.')
# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
# parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
#                     help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
# parser.add_argument('--amp', action='store_true', default=False,
#                     help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
# parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def update_lr(args, optimizer, epoch, per_epoch_update=True, it=None, warmup_its=None):
    if epoch < args.warmup_epochs and not per_epoch_update:
        step = (args.lr - args.warmup_lr) / warmup_its
        lr = it * step + args.warmup_lr
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
    elif epoch > args.warmup_epochs and per_epoch_update:
        lr = args.lr * (args.decay_rate ** (epoch // args.decay_epochs))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr

def main():
    setup_default_logging()
    args, args_text = _parse_args()


    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)

    # model = skip_v3(num_classes=args.num_classes)

    data_config = resolve_data_config(vars(args), model=model, verbose=True)



    optimizer = create_optimizer(args, model)

    use_amp = False

    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)
    # if resume_state and not args.no_resume_opt:
    #     if 'optimizer' in resume_state:
    #         if args.local_rank == 0:
    #             logging.info('Restoring Optimizer state from checkpoint')
    #         optimizer.load_state_dict(resume_state['optimizer'])
    #     if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
    #         if args.local_rank == 0:
    #             logging.info('Restoring NVIDIA AMP state from checkpoint')
    #         amp.load_state_dict(resume_state['amp'])
    del resume_state

    model = nn.DataParallel(model).cuda()
    print('model to data parallel')

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='',
            resume=args.resume)

    # lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # print((optimizer.param_groups[1]['lr']))
    # lambda_epoch = lambda epoch:( (args.decay_rate ** ((epoch - args.warmup_epochs) // args.decay_epochs)) if (epoch > args.warmup_epochs) else 1)

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
        if start_epoch < args.warmup_epochs:
            update_lr(args=args, optimizer=optimizer, epoch=start_epoch, per_epoch_update=False) #TODO iterations correction
        else:
            update_lr(args=args, optimizer=optimizer, epoch=start_epoch, per_epoch_update=True)


    # if lr_scheduler is not None and start_epoch > 0:
    #     print((optimizer.param_groups[1]['lr']))
    #     lr_scheduler.step(start_epoch)
    #     print(start_epoch)
    num_epochs = args.epochs
    logging.info('Scheduled epochs: {}'.format(num_epochs))

    train_dir = args.data
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    # dataset_train = Dataset(train_dir)

    # collate_fn = None
    # if args.prefetcher and args.mixup > 0:
    #     assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
    #     collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)
    #
    # if num_aug_splits > 1:
    #     dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # loader_train = create_loader(
    #     dataset_train,
    #     input_size=data_config['input_size'],
    #     batch_size=args.batch_size,
    #     is_training=True,
    #     use_prefetcher=args.prefetcher,
    #     re_prob=args.reprob,
    #     re_mode=args.remode,
    #     re_count=args.recount,
    #     re_split=args.resplit,
    #     color_jitter=args.color_jitter,
    #     auto_augment=args.aa,
    #     num_aug_splits=num_aug_splits,
    #     interpolation=args.train_interpolation,
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     collate_fn=collate_fn,
    #     pin_memory=args.pin_mem,
    #     use_multi_epochs_loader=args.use_multi_epochs_loader
    # )
    #
    # eval_dir = os.path.join(args.data, 'val')
    # if not os.path.isdir(eval_dir):
    #     eval_dir = os.path.join(args.data, 'validation')
    #     if not os.path.isdir(eval_dir):
    #         logging.error('Validation folder does not exist at: {}'.format(eval_dir))
    #         exit(1)
    # dataset_eval = Dataset(eval_dir)
    #
    # loader_eval = create_loader(
    #     dataset_eval,
    #     input_size=data_config['input_size'],
    #     batch_size=args.validation_batch_size_multiplier * args.batch_size,
    #     is_training=False,
    #     use_prefetcher=args.prefetcher,
    #     interpolation=data_config['interpolation'],
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     crop_pct=data_config['crop_pct'],
    #     pin_memory=args.pin_mem,
    # )

    loader_train = Loader('train', train_dir, batch_size=args.batch_size, num_workers=args.workers)
    loader_eval = Loader('val', train_dir, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)


    # train_loss_fn = nn.CrossEntropyLoss().cuda()
    train_loss_fn = nn.CrossEntropyLoss(reduction='none')

    validate_loss_fn = train_loss_fn

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(data_config['input_size'][-1])
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    writer =SummaryWriter()


    try:
        for epoch in range(start_epoch, num_epochs):

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args
                , saver=saver, output_dir=output_dir,
                use_amp=use_amp, model_ema=model_ema, summary_writer=writer)

            eval_metrics, _ = validate(model, loader_eval, validate_loss_fn, args, epoch=epoch)
            if model_ema is not None:

                ema_eval_metrics, top1 = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args,epoch=epoch, log_suffix=' (EMA)', summary_writer=writer)

                writer.add_scalar('accuracy top 1', top1, epoch)

            # if lr_scheduler is not None and epoch >= args.warmup_epochs:
            #     # step LR for next epoch
            #     lr_scheduler.step(epoch + 1)
            update_lr(args=args, optimizer=optimizer, epoch=epoch, per_epoch_update=True)

            update_summary(
                epoch, train_metrics, eval_metrics, ema_eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                save_metric_ema = ema_eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, metric_ema=save_metric_ema, use_amp=use_amp)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def avg_shift(tensor):
    average = tensor.clone().mean()
    normalised = tensor - average
    return normalised


def var_loss(x, y_gt, y_pred, ce_criterion):
    x_samples = x.clone().detach()
    x_per_sample = x_samples.reshape(x_samples.shape[0], -1)

    var_per_sample = (torch.var(x_per_sample, dim=1))
    var_per_sample_avg_shift = avg_shift(var_per_sample)
    beta = 0.3
    var_per_sample_norm = var_per_sample_avg_shift + beta
    # weights = torch.clamp(torch.log(1 + torch.exp(var_per_sample * 6)), 0.3, 3.)
    alpha = 0.3
    weights = torch.clamp(((torch.exp(alpha * var_per_sample_norm) - 1) / alpha) + alpha, 0.3, 2.)

    loss = weights * ce_criterion(y_pred, y_gt)
    # loss = ce_criterion(y_pred, y_gt)

    loss = loss.mean()

    return loss


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None, summary_writer=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        iter = len(loader) * epoch + batch_idx
        warmup_iters = args.warmup_epochs * len(loader)
        update_lr(args=args, optimizer=optimizer, epoch=epoch, per_epoch_update=False, it=iter, warmup_its=warmup_iters)

        # output = model(input)
        output, features = model(input)


        # loss = loss_fn(output, target)
        loss = var_loss(features, y_pred=output, y_gt=target, ce_criterion=loss_fn)

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0)  / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

            it = len(loader) * epoch + batch_idx
            _, pred = output.max(1)
            var_correct, var_inc = variance_per_pred(features, target, pred)
            summary_writer.add_scalars('var correct incorrect train', {
                'correct': var_correct,
                'incorrect': var_inc,
            }, it)
            summary_writer.add_scalar("loss", loss.detach().cpu().numpy(), it)

            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp, batch_idx=batch_idx)

        # if lr_scheduler is not None:
        #     lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, epoch, log_suffix='', summary_writer=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            input = input.cuda()
            target = target.cuda()

            # output = model(input)
            output, features = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            # loss = loss_fn(output, target)
            loss = var_loss(features, y_pred=output, y_gt=target, ce_criterion=loss_fn)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if  (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                if summary_writer != None:
                    _, pred = output.max(1)
                    var_correct, var_inc = variance_per_pred(features, target, pred)
                    summary_writer.add_scalars('var correct incorrect test', {
                        'correct': var_correct,
                        'incorrect': var_inc,
                    }, len(loader) * epoch + batch_idx)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics, top1_m.avg


if __name__ == '__main__':
    main()
