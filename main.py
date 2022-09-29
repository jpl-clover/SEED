#!/usr/bin/env python
from datetime import datetime
import os
import shutil
import sys
import time

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import wandb

import seed.builder
from seed.data_utils import get_dataset_from_name
from seed.utils import init_distributed_mode, parse_args, set_manual_seed
from tools.logger import setup_logger
from tools.utils import (
    AverageMeter,
    ProgressMeter,
    ValueMeter,
    adjust_learning_rate,
    load_moco_teacher_encoder,
    load_simclr_teacher_encoder,
    load_swav_teacher_encoder,
    mocov1_aug,
    mocov2_aug,
    resume_training,
    save_checkpoint,
    simclr_aug,
    soft_cross_entropy,
    swav_aug,
)


def main(gpu, args):
    init_distributed_mode(gpu, args)
    cudnn.benchmark = True

    if args.distributed:
        logger = setup_logger(
            output=args.model_path,
            distributed_rank=dist.get_rank(),
            color=True,
            name="SEED",
        )
        # save the distributed node machine
        logger.info(f"world size: {dist.get_world_size()}")
        logger.info(f"local_rank: {args.local_rank}")
        logger.info(f"dist.get_rank(): {dist.get_rank()}")
        logger.info(f"args.global_rank = {args.global_rank}")
    else:
        logger = setup_logger(output=args.model_path, color=True, name="SEED")
        logger.info('Single GPU mode for debugging.')

    # create model
    logger.info("=> creating student encoder '{}'".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'".format(args.teacher_arch))
    # initialize model object, feed student and teacher into encoders.
    model = seed.builder.SEED(args)
    model = model.to(args.device)

    logger.info(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # this is the linear LR scaling rule from "Training ImageNet in 1 hour" by Goyal, ..., Kaiming He
    # "When the batch size is increased by a factor of k, multiply the learning rate by k"
    args.lr_mult = (args.world_size * args.batch_size) / 256
    args.warmup_epochs = 5
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr_mult * args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            args.lr_mult * args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented yet.")

    # load the SSL pre-trained teacher encoder into model.teacher
    # if args.teacher_weights:
    #     if os.path.isfile(args.teacher_weights):
    #         if args.teacher_ssl == 'moco':
    #             model = load_moco_teacher_encoder(args, model, logger, distributed=args.distributed)
    #         elif args.teacher_ssl == 'simclr':
    #             model = load_simclr_teacher_encoder(args, model, logger, distributed=args.distributed)
    #         elif args.teacher_ssl == 'swav':
    #             model = load_swav_teacher_encoder(args, model, logger, distributed=args.distributed)

    #         logger.info(
    #             "=> Teacher checkpoint successfully loaded from '{}'".format(
    #                 args.teacher_weights
    #             )
    #         )
    #     else:
    #         logger.info("wrong distillation checkpoint.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # clear unnecessary weights
    torch.cuda.empty_cache()

    train_loader, train_dataset, train_sampler = get_train_loader(args)

    # tensorboard
    if args.global_rank == 0:
        summary_writer = setup_tensorboard_and_wandb(args, len(train_dataset))
    else:
        summary_writer = None


    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed: train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        if summary_writer is not None:
            # Tensor-board logger
            summary_writer.add_scalar("Distillation/loss", loss, epoch)
            summary_writer.add_scalar(
                "Distillation/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

        if args.global_rank == 0 and epoch % 10:
            file_str = 'Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar'\
                .format(args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch)

            save_checkpoint(
                {
                    # if resuming from this checkpoint, will start from next epoch
                    "epoch": epoch + 1,
                    "arch": args.student_arch,
                    "teacher_arch": args.teacher_arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=os.path.join(summary_writer.log_dir, file_str),
            )

            logger.info(
                "==============> checkpoint saved to {}".format(
                    os.path.join(args.model_path, file_str)
                )
            )

    ## End training - close TB writer add final checkpoint file to wandb
    if args.global_rank == 0:
        file_str = "Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar".format(
            args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch
        )

        save_checkpoint(
            {
                # if resuming from this checkpoint, will start from next epoch
                "epoch": epoch + 1,
                "arch": args.student_arch,
                "teacher_arch": args.teacher_arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(summary_writer.log_dir, file_str),
        )
        summary_writer.close()
        if "wandb" in sys.modules and args.use_wandb:
            wandb.save(os.path.join(summary_writer.log_dir, file_str))
            wandb.finish()


def get_train_loader(args):
    if args.teacher_ssl == "swav":
        augmentation = swav_aug
    elif args.teacher_ssl == "simclr":
        augmentation = simclr_aug
    elif args.teacher_ssl == "moco" and args.student_mlp:
        augmentation = mocov2_aug
    else:
        augmentation = mocov1_aug

    train_dataset, _ = get_dataset_from_name(args, transform=augmentation)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    return train_loader, train_dataset, train_sampler


def setup_tensorboard_and_wandb(args, num_images):
    writer = None
    if args.global_rank == 0:
        # if wandb is imported, do wandb.init
        # must call this BEFORE creating summary writer
        run_name = (
            f"SEED|{args.dataset}|"
            f"{num_images}|{args.image_size}|"
            f"S{args.seed}|"
            f"B{args.batch_size * args.world_size}|"
            f"{args.start_epoch}"
            f"-{args.epochs}|"
            f"{args.student_arch}|"
            f"T{args.teacher_arch}|{args.teacher_weights + '|' if args.teacher_weights else ''}"
            f"{args.dim}|{args.optimizer}|"
            f"{datetime.now().strftime('%y%m%d-%H%M%S')}"  # timestamp
        )
        # save current args as a yaml file in args.model_path
        config_backup_path = os.path.join(
            args.model_path, f"config_{datetime.now().strftime('%y%m%d-%H%M%S')}.yaml"
        )
        with open(config_backup_path, "w") as f:
            # copy the yaml file in args.config to config_backup_path
            shutil.copyfile(args.config, config_backup_path)

        args.global_step = 0
        if "wandb" in sys.modules and args.use_wandb:
            # note: w&b will crash if the provided run_id doesn't exist since there's nothing to resume from
            wandb_resume = None if args.wandb_run_id is None else "must"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                resume=wandb_resume,
                id=args.wandb_run_id,
                config=vars(args),
                sync_tensorboard=True,
                save_code=True,
            )
            args.global_step = wandb.run.step
            wandb.save(config_backup_path)
            wandb.run.log_code(".")
        writer = SummaryWriter(log_dir=os.path.join(args.model_path, run_name))
    return writer


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')
    mem = ValueMeter('GPU Memory Used', ':5.0f')

    progress = ProgressMeter(  # TODO: use tqdm
        len(train_loader),
        [batch_time, data_time, lr, losses, mem],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    mem.update(torch.cuda.max_memory_allocated(device=0) / 1024.0 / 1024.0)

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    if args.distributed:
        model.module.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.module.teacher.named_parameters():
            if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    else:
        model.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.teacher.named_parameters():
           if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i, (images, _) in enumerate(train_loader):

        if not args.distributed:
            images = images.to(args.device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):

            logit, label = model(image=images)
            loss = criterion(logit, label)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    set_manual_seed(args.seed)

    if args.world_size > 1 and args.multiprocessing_distributed:
        print(
            f"Training with {args.nodes} nodes, {args.ngpus_per_node} GPUs per node. "
            f"Waiting until all nodes join before starting training"
        )
        # Use torch.multiprocessing.spawn to launch distributed `main_worker` processes
        mp.spawn(main, nprocs=args.ngpus_per_node, args=(args,))
    else:  # Simply call main_worker function
        main(None, args)
