import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm

from fontGAN import FontGenerator, FontDiscriminator
from dataloader import FontDataset

def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def main(args=None):
    parser = argparse.ArgumentParser(description="FontGAN pytorch.")

    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--dev_folder', type=str)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=236)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=5000)

    args = parser.parse_args()

    log = util.get_logger(args.save_dir, args.name)

    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Build Models
    log.info('Building model...')
    generator = FontGenerator()
    discriminator = FontDiscriminator()

    # Get optimizer and scheduler
    gen_optimizer = optim.Adadelta(generator.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    disc_optimizer = optim.Adadelta(discriminator.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)

    # Build dataset
    train_dataset = FontDataset() # Path
    dev_dataset = FontDataset() # Path

    train_loader = data.Dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = data.Dataloader(dev_dataset, batch_size=1)

    step = 0
    steps_til_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target in train_loader:
                print(orig.shape)
                print(condition.shape)
                print(target.shape)

    return 0

if __name__ == '__main__':
    main(args)