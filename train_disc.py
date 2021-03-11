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

from src.models.fontGAN import DualHeadFontDiscriminator
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

    args = parser.parse_args()

    log = util.get_logger(args.save_dir, args.name)

    tbx = SummaryWriter(util.get_save_dir(
        args.save_dir, args.name, training=True))
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
    discriminator = DualHeadFontDiscriminator()

    discriminator = discriminator.to(device)
    discriminator.train()

    # Set criterion
    criterion = F.binary_cross_entropy

    # Get optimizer and scheduler
    disc_optimizer = optim.Adadelta(discriminator.parameters(), args.lr,
                                    weight_decay=args.l2_wd)
    disc_scheduler = sched.LambdaLR(disc_optimizer, lambda s: 1.)

    # Build dataset
    train_dataset = FontDataset(args.train_dir)  # Path
    dev_dataset = FontDataset(args.dev_dir)  # Path

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size)

    # Set up outputs folder
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    if not os.path.isdir('outputs/{}'.format(args.name)):
        os.mkdir('outputs/{}'.format(args.name))

    step = 0
    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {}...'.format(epoch))

        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target, false_target in train_loader:
                curr_batch_size = len(orig)
                progress_bar.update(curr_batch_size)

                # To pass into discriminator: target + original font
                head1 = torch.cat([target, false_target], dim=0).to(device)
                head2 = torch.cat([orig, orig], dim=0).to(device)

                # Update discriminator
                disc_optimizer.zero_grad()
                # To pass into discriminator: generated + original font

                disc_pred = discriminator(head1, head2).squeeze(
                    1)  # shape = (batch_size,)
                # First [batch_size] images are of the real
                disc_real = disc_pred[:curr_batch_size]
                # Rest are of fake images
                disc_fake = disc_pred[curr_batch_size:]

                labels_real = torch.ones_like(disc_real)
                labels_fake = torch.zeros_like(disc_fake)

                disc_real_loss = criterion(disc_real, labels_real)  # BCE Loss
                disc_fake_loss = criterion(disc_fake, labels_fake)  # BCE Loss

                disc_loss = (disc_real_loss + disc_fake_loss) / 2

                disc_loss.backward()
                disc_optimizer.step()
                tbx.add_scalar('train/trainDiscBCELoss',
                               disc_loss.item(), step)

                step += curr_batch_size

        # Save generator outputs and evaluate
        log.info('Evaluating')
        evaluate(discriminator, dev_loader, device,
                 epoch, step, tbx, args.name, criterion)
        epoch += 1

    return 0


def evaluate(disc, dataloader, device, epoch, step, tb, name, criterion):
    disc.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for orig, condition, target, false_target in dataloader:
            curr_batch_size = len(orig)
            progress_bar.update(curr_batch_size)

            # To pass into discriminator: target + original font
            head1 = torch.cat([target, false_target], dim=0).to(device)
            head2 = torch.cat([orig, orig], dim=0).to(device)

            disc_pred = disc(head1, head2).squeeze(1)  # shape = (batch_size,)
            # First [batch_size] images are of the real
            disc_real = disc_pred[:curr_batch_size]
            # Rest are of fake images
            disc_fake = disc_pred[curr_batch_size:]

            labels_real = torch.ones_like(disc_real)
            labels_fake = torch.zeros_like(disc_fake)

            disc_real_loss = criterion(disc_real, labels_real)  # BCE Loss
            disc_fake_loss = criterion(disc_fake, labels_fake)  # BCE Loss

            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            total_loss += disc_loss.item() * curr_batch_size

            real_correct = torch.where(disc_real >= 0.5, torch.ones_like(
                disc_real), torch.zeros_like(disc_real))
            fake_correct = torch.where(disc_fake < 0.5, torch.ones_like(
                disc_fake), torch.zeros_like(disc_fake))
            total_correct += real_correct.sum() + fake_correct.sum()

    tb.add_scalar('dev/TrainDiscBCELoss', total_loss / len(dataloader), epoch)
    tb.add_scalar('dev/TrainDiscAccuracy',
                  total_correct / len(dataloader), epoch)

    disc.train()


if __name__ == '__main__':
    main()
