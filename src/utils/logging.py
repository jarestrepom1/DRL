from torch.utils.tensorboard import SummaryWriter


def create_tb_writer(log_dir: str) -> SummaryWriter:
    return SummaryWriter(log_dir=log_dir)
