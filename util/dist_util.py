import os
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def create_logger(logging_dir):
    rank = get_rank()
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_logger(configs, rank):
    if rank == 0:
        log_dir = configs['log_dir']
        tb_dir = configs['tb_path']
        os.makedirs(log_dir, exist_ok=True)
        logger = create_logger(log_dir)
        writer = SummaryWriter(tb_dir)
        logger.info(f"Experiment directory created at {configs['save_path']}")
    else:
        logger = create_logger(None)
        writer = None
    return logger, writer


def init_dist():
    assert torch.cuda.is_available(), "requires at least one GPU."
    
    # Check if running in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Setup DDP:
        dist.init_process_group("nccl")
        rank = dist.get_rank()  # global rank
        device = rank % torch.cuda.device_count()  # local rank
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
        return rank, device
    else:
        # Single GPU mode for evaluation
        print("Running in single GPU mode (no distributed training)")
        device = 0
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            # Use the first visible device
            device = 0
        torch.cuda.set_device(device)
        return 0, device


def set_seed(seed=3407):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()  # global rank
        seed = seed * dist.get_world_size() + rank
    else:
        rank = 0
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup():
    # End DDP training
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def rank_zero_info(message: str):
    if get_rank() == 0 and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info(message)


def rank_zero_warn(message: str):
    if get_rank() == 0 and logging.getLogger().isEnabledFor(logging.WARNING):
        logging.warning(message)
