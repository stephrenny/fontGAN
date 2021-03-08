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

from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.models.fontGAN import FontGenerator, FontDiscriminator
from src.dataloader import FontDataset

def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def main(args=None):
    parser = argparse.ArgumentParser(description="FontGAN pytorch.")

    parser.add_argument('--train_dir', type=str, default='data/font_characters/train')
    parser.add_argument('--dev_dir', type=str, default='data/font_characters/dev')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=236)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--l2_wd', type=float, default=1.)

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

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.train()
    discriminator.train()

    # Set criterion
    criterion = nn.BCEWithLogitsLoss()

    # Get optimizer and scheduler
    gen_optimizer = optim.Adadelta(generator.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    disc_optimizer = optim.Adadelta(discriminator.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    gen_scheduler = sched.LambdaLR(gen_optimizer, lambda s: 1.)
    disc_scheduler = sched.LambdaLR(disc_optimizer, lambda s: 1.)

    # Build dataset
    train_dataset = FontDataset(args.train_dir) # Path
    dev_dataset = FontDataset(args.dev_dir) # Path

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=1)

    step = 0
    steps_til_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target in train_loader:
                curr_batch_size = len(orig)
                progress_bar.update(curr_batch_size)
                x = torch.cat([orig, condition], dim=1).to(device) # Pass into generator: original font + target text
                output_real = torch.cat([target, orig], dim=1).to(device) # Pass into discriminator: target + original font
                orig = orig.to(device)

                # Update discriminator
                disc_optimizer.zero_grad()
                gen_imgs = generator(x)
                output_fake = torch.cat([gen_imgs, orig], dim=1) # Pass into discriminator: generated + original font

                disc_pred = discriminator(torch.cat([output_real, output_fake], dim=0)).squeeze()
                disc_real = disc_pred[:curr_batch_size]
                disc_fake = disc_pred[curr_batch_size:]

                labels_real = torch.ones_like(disc_real)
                labels_fake = torch.zeros_like(disc_fake)

                disc_real_loss = criterion(disc_real, labels_real)
                disc_fake_loss = criterion(disc_fake, labels_fake)

                disc_loss = (disc_real_loss + disc_fake_loss) / 2

                disc_loss.backward(retain_graph=True)
                tbx.add_scalar('train/DiscLoss', disc_loss.item(), step)
                disc_optimizer.step()

                # Train generator
                gen_optimizer.zero_grad()
                gen_imgs = generator(x)
                output_fake = torch.cat([gen_imgs, orig], dim=1)
                disc_pred = discriminator(output_fake).squeeze()

                labels_fake = torch.ones_like(disc_fake)

                gen_loss = criterion(disc_pred, labels_fake)
                gen_loss.backward()
                gen_optimizer.step()

                progress_bar.set_postfix(epoch=epoch, GenLoss=gen_loss.item())
                tbx.add_scalar('train/GenLoss', gen_loss.item(), step)

                step += 1
                if step > 10:
                    exit(0)

    return 0

if __name__ == '__main__':
    main()