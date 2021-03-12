import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import src.util as util
import os
import cv2

from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.models.fontGAN import DualHeadFontDiscriminator, DiscResNet, ResnetGenerator
from src.dataloader import FontDataset

"""
Sanity check for 
"""

def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def main(args=None):
    parser = argparse.ArgumentParser(description="FontGAN pytorch.")

    parser.add_argument('--train_dir', type=str,
                        default='data/font_characters/train')
    parser.add_argument('--dev_dir', type=str,
                        default='data/font_characters/dev')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=236)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--name', type=str, default='train_disc')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--eval_steps', type=int, default=2000)
    parser.add_argument('--l2_wd', type=float, default=1.)
    parser.add_argument('--lambda_recon', type=float, default=.2)
    parser.add_argument('--mode', type=str, default='style')
    parser.add_argument('--metric_name', type=str, default='dev_loss')
    parser.add_argument('--maximize_metric', action='store_true')
    parser.add_argument('--max_checkpoints', type=int, default=2)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--resume_step', action='store_true')
    parser.add_argument('--no_save', action='store_true')

    args = parser.parse_args()

    generator = ResnetGenerator()
    exit(0)

    log = util.get_logger(args.save_dir, args.name)

    save_dir = util.get_save_dir(
        args.save_dir, args.name, training=True)

    tbx = SummaryWriter(save_dir)
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    step = 0

    saver = util.CheckpointSaver(save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Build Models
    log.info('Building model...')
    discriminator = DiscResNet()
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        discriminator, step = util.load_model(discriminator, args.load_path, args.gpu_ids, return_step=True)

        if not args.resume_step:
            step = 0

    discriminator = discriminator.to(device)
    discriminator.train()

    # Set criterion
    criterion = nn.TripletMarginLoss(margin=4.8, p=2)

    # Get optimizer and scheduler
    disc_optimizer = optim.Adadelta(discriminator.parameters(), args.lr,
                                    weight_decay=args.l2_wd)
    disc_scheduler = sched.LambdaLR(disc_optimizer, lambda s: 1.)

    # Build dataset
    train_dataset = FontDataset(args.train_dir, rand=True)  # Path
    dev_dataset = FontDataset(args.dev_dir, rand=True)  # Path

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size)

    # Set up outputs folder
    # if not os.path.isdir('outputs'):
    #     os.mkdir('outputs')

    # if not os.path.isdir('outputs/{}'.format(args.name)):
    #     os.mkdir('outputs/{}'.format(args.name))

    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {}...'.format(epoch))

        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target, false_target in train_loader:

                curr_batch_size = len(orig)
                progress_bar.update(curr_batch_size)

                # To pass into discriminator: target + original font
                orig = orig.to(device)
                condition = condition.to(device)
                target = target.to(device)
                false_target = false_target.to(device)

                # Update discriminator
                disc_optimizer.zero_grad()
                # To pass into discriminator: generated + original font

                if args.mode == 'style':
                    # Train discriminator to match style (font)
                    x = torch.cat([orig, target, false_target], dim=0)
                else:
                    # Train discrminiator to match content (orig is odd one out in this case)
                    x = torch.cat([condition, target, orig], dim=0)

                feats = discriminator(x)  # shape = (batch_size, hidden_size)

                anchor = feats[:curr_batch_size]
                positive = feats[curr_batch_size:curr_batch_size * 2]
                negative = feats[curr_batch_size*2:]

                disc_loss = criterion(anchor, positive, negative)

                disc_loss.backward()
                disc_optimizer.step()
                tbx.add_scalar('train/trainDiscTripLoss', disc_loss.item(), step)

                step += curr_batch_size

        # Save generator outputs and evaluate
        log.info('Evaluating')
        metrics = evaluate(discriminator, dev_loader, device, criterion, args.mode)
        tbx.add_scalar('dev/TrainDiscTripLoss', metrics['dev_loss'], epoch)
        tbx.add_scalar('dev/TrainDiscAccuracy', metrics['dev_acc'], epoch)

        if not args.no_save:
            saver.save(step, discriminator, metrics[args.metric_name], device)
        epoch += 1

    return 0

def evaluate(disc, dataloader, device, criterion, mode):
    disc.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for orig, condition, target, false_target in dataloader:
            curr_batch_size = len(orig)
            progress_bar.update(curr_batch_size)

            # To pass into discriminator: target + original font
            orig = orig.to(device)
            target = target.to(device)
            condition = condition.to(device)
            false_target = false_target.to(device)

            if mode == 'style':
                # Train discriminator to match style (font)
                x = torch.cat([orig, target, false_target], dim=0)
            else:
                # Train discrminiator to match content (orig is odd one out in this case)
                x = torch.cat([condition, target, orig], dim=0)

            feats = disc(x)  # shape = (batch_size, hidden_size)

            anchor = feats[:curr_batch_size]
            positive = feats[curr_batch_size:curr_batch_size * 2]
            negative = feats[curr_batch_size*2:]
            # Rest are of fake images

            disc_loss = criterion(anchor, positive, negative)
            total_loss += disc_loss.item() * curr_batch_size

            real_correct = l2_dist(anchor, positive) <= 4.8
            fake_correct = l2_dist(anchor, negative) > 4.8
            total_correct += real_correct.sum().float() + fake_correct.sum().float()

    disc.train()

    return {'dev_loss': total_loss / len(dataloader.dataset), 'dev_acc': total_correct / (len(dataloader.dataset) * 2)}

def l2_dist(anchor, samples):
    return torch.sqrt(((anchor - samples) ** 2).sum(dim=1))

if __name__ == '__main__':
    main()
