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

from src.models.fontGAN import FontGenerator, FontDiscriminator
from src.dataloader import FontDataset

"""
Much of the training harness is recycled from the cs244n squad program
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--name', type=str, default='vanilla')
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
    generator = FontGenerator()
    discriminator = FontDiscriminator()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.train()
    discriminator.train()

    # Set criterion
    criterion = F.binary_cross_entropy_with_logits

    # Get optimizer and scheduler
    gen_optimizer = optim.Adadelta(generator.parameters(), args.lr,
                                   weight_decay=args.l2_wd)
    disc_optimizer = optim.Adadelta(discriminator.parameters(), args.lr,
                                    weight_decay=args.l2_wd)
    gen_scheduler = sched.LambdaLR(gen_optimizer, lambda s: 1.)
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
    steps_til_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {}...'.format(epoch))

        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target, false_target in train_loader:
                curr_batch_size = len(orig)
                progress_bar.update(curr_batch_size)
                # Pass into generator: original font + target text
                x = torch.cat([orig, condition], dim=1).to(device)
                # To pass into discriminator: target + original font
                output_real = torch.cat([target, orig], dim=1).to(device)

                # Dataset-generated false fonts
                # To pass into discriminator: false_target + original font
                output_false = torch.cat(
                    [false_target, orig], dim=1).to(device)

                orig = orig.to(device)

                # Update discriminator
                disc_optimizer.zero_grad()
                gen_imgs = generator(x).detach()
                # To pass into discriminator: generated + original font
                output_fake = torch.cat([gen_imgs, orig], dim=1)

                disc_pred = discriminator(torch.cat([output_real, output_fake, output_false], dim=0)).squeeze(
                    dim=1)  # shape = (batch_size,)
                # First [batch_size] images are of the real
                disc_real = disc_pred[:curr_batch_size]
                # Rest are of fake images
                disc_fake = disc_pred[curr_batch_size:]

                labels_real = torch.ones_like(disc_real)
                labels_fake = torch.zeros_like(disc_fake)

                disc_real_loss = criterion(disc_real, labels_real)  # BCE Loss
                disc_fake_loss = criterion(disc_fake, labels_fake)  # BCE Loss

                disc_loss = (disc_real_loss + disc_fake_loss) / 2

                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
                tbx.add_scalar('train/DiscLoss', disc_loss.item(), step)

                # Train generator
                gen_optimizer.zero_grad()
                gen_imgs = generator(x)
                # To pass into discriminator: generated + original font
                output_fake = torch.cat([gen_imgs, orig], dim=1)
                disc_pred = discriminator(output_fake).squeeze(dim=1)

                # We wish for the discrminator to mark these images as real
                labels_fake = torch.ones_like(disc_pred)

                gen_loss = criterion(
                    disc_pred, labels_fake) + args.lambda_recon * recon_crit(gen_imgs, target.to(device))
                gen_loss.backward()
                gen_optimizer.step()

                progress_bar.set_postfix(epoch=epoch, GenLoss=gen_loss.item())
                tbx.add_scalar('train/GenLoss', gen_loss.item(), step)

                step += curr_batch_size

                steps_til_eval -= curr_batch_size
                if steps_til_eval <= 0:
                    save_outputs(gen_imgs, orig, condition, target,
                                 step, 'outputs/{}/train'.format(args.name))
                    steps_til_eval = args.eval_steps

        # Save generator outputs and evaluate
        log.info('Evaluating')
        evaluate(generator, dev_loader, device, epoch, step, tbx, args.name)
        epoch += 1

    return 0


def save_outputs(gen_images, orig, condition, target, step, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    gen_imgs = gen_images.cpu().detach().squeeze(dim=1).numpy() * 255
    orig_imgs = orig.cpu().detach().squeeze(dim=1).numpy() * 255
    target_imgs = target.cpu().detach().squeeze(dim=1).numpy() * 255
    condition_imgs = condition.cpu().detach().squeeze(dim=1).numpy() * 255

    if not os.path.isdir(os.path.join(save_dir, 'step-{}'.format(step))):
        os.mkdir(os.path.join(save_dir, 'step-{}'.format(step)))

    idx = 0
    for gen_img, orig_img, target_img, condition_img in zip(gen_imgs, orig_imgs, target_imgs, condition_imgs):
        cv2.imwrite(os.path.join(save_dir, 'step-{}'.format(step),
                                 'orig-' + str(idx) + '.jpg'), orig_img)
        cv2.imwrite(os.path.join(save_dir, 'step-{}'.format(step),
                                 'gen-' + str(idx) + '.jpg'), gen_img)
        cv2.imwrite(os.path.join(save_dir, 'step-{}'.format(step),
                                 'target-' + str(idx) + '.jpg'), target_img)
        cv2.imwrite(os.path.join(save_dir, 'step-{}'.format(step),
                                 'condition-' + str(idx) + '.jpg'), condition_img)
        idx += 1


def recon_crit(generated, target):
    return float((generated - target).abs().mean())


def recon_crit2(generated, target):
    return float(((generated - target) ** 2).mean())


def evaluate(gen, dataloader, device, epoch, step, tb, name):
    gen.eval()
    total_loss = 0
    saved = False
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for orig, condition, target, _ in dataloader:
            curr_batch_size = len(orig)
            progress_bar.update(curr_batch_size)

            orig = orig.to(device)
            condition = condition.to(device)
            target = target.to(device)

            x = torch.cat([orig, condition], dim=1).to(device)
            gen_imgs = gen(x)
            total_loss += float((gen_imgs -
                                 target).abs().mean(dim=(1, 2, 3)).sum())

            if not saved:
                save_outputs(gen_imgs, orig, condition, target,
                             step, 'outputs/{}/dev'.format(name))
                saved = True

    tb.add_scalar('dev/L1Loss', total_loss / len(dataloader), epoch)
    print('L1: {}'.format(total_loss / len(dataloader)))
    gen.train()


if __name__ == '__main__':
    main()
