import os
import builtins
import warnings

import torch
import torch.distributed as dist


def init_distributed_mode(gpu, args):
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    if gpu is None and torch.cuda.is_available():
        gpu = 0  # use the first GPU by default

    args.gpu = gpu

    try:  # try submitit first
        import submitit

        job_env = submitit.JobEnvironment()
        args.local_rank = job_env.local_rank  # range: 0 to (num_gpus_per_node - 1)
        args.global_rank = job_env.global_rank  # 0 to (num_nodes * gpus_per_node - 1)
        args.world_size = job_env.num_tasks
        args.node_rank = job_env.node
        gpu = args.local_rank
    except Exception as e:  # try to use SLURM vars if local_rank and node_rank not specified
        print(
            f"Ran into exception {e} when loading distributed vars"
            f"from submitit. Using environment vars instead."
        )
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("SLURM_LOCALID", gpu))
        if args.node_rank == -1:
            args.node_rank = int(
                os.environ.get("SLURM_NODEID", os.environ.get("RANK", 0))
            )
        # https://github.com/facebookincubator/submitit/blob/main/submitit/slurm/slurm.py#L179
        args.global_rank = int(
            os.environ.get(
                "SLURM_PROCID", args.node_rank * args.ngpus_per_node + args.local_rank
            )
        )
        args.world_size = args.nodes * args.ngpus_per_node
        if "SLURM_SUBMIT_HOST" in os.environ:
            args.master_addr = os.environ["SLURM_SUBMIT_HOST"]

    if args.world_size > 1 and not args.multiprocessing_distributed:
        warnings.warn(
            f"WARNING:\n"
            f"========\n"
            f"World size is {args.world_size}, but neither DDP nor DataParallel"
            f" are enabled. Setting gpus=1, nodes=1, dataparallel=False, ddp=False"
        )
        args.gpus = args.nodes = args.world_size = 1
        args.multiprocessing_distributed = False
    elif args.world_size == 1 and args.multiprocessing_distributed:
        warnings.warn(
            f"WARNING:\n"
            f"========\n"
            f"World size is 1, but DDP is enabled. Setting gpus=1, nodes=1,"
            f" dataparallel=False, ddp=False"
        )
        args.gpus = args.nodes = args.world_size = 1
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.distributed:
        if "dist_url" not in args:
            args.dist_url = f"tcp://{args.master_addr}:{args.port}"
        print(
            f"Initializing process group on global rank {args.global_rank} "
            f"on node {args.node_rank}, gpu {args.local_rank} "
            f"with port {os.environ.get('MASTER_PORT')}"
        )
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.global_rank,
        )
        print(f"Process initialization completed for global rank {args.global_rank}")

        args.workers = int(
            (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node
        )
    elif gpu is None and not torch.cuda.is_available():
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    args.device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(args.device)
