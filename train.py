import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import src.util as util
import os
import cv2

from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.models.fontGAN import DiscResNet, ResUnetGenerator
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
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--name', type=str, default='vanilla')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--l2_wd', type=float, default=1.)
    parser.add_argument('--lambda_recon', type=float, default=.2)
    parser.add_argument('--mode', type=str, default='style')
    parser.add_argument('--metric_name', type=str, default='disc_style_acc')
    parser.add_argument('--maximize_metric', action='store_true')
    parser.add_argument('--max_checkpoints', type=int, default=2)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--resume_step', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--skip_epochs', type=int)

    args = parser.parse_args()

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
    log.info('Building models...')
    style_discriminator = DiscResNet()
    content_discriminator = DiscResNet()
    generator = ResUnetGenerator()

    step = 0

    # TODO Polish up save and laod
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        discriminator, step = util.load_model(
            discriminator, args.load_path, args.gpu_ids, return_step=True)

        if not args.resume_step:
            step = 0

    style_discriminator = style_discriminator.to(device)
    content_discriminator = content_discriminator.to(device)
    generator = generator.to(device)
    style_discriminator.train()
    content_discriminator.train()
    generator.train()

    # Set criterion
    criterion = criterion = nn.TripletMarginLoss(margin=4.0, p=2)

    # Get optimizer and scheduler
    gen_optimizer = optim.Adadelta(generator.parameters(), args.lr,
                                   weight_decay=args.l2_wd)
    style_disc_optimizer = optim.Adadelta(style_discriminator.parameters(), args.lr,
                                          weight_decay=args.l2_wd)
    content_disc_optimizer = optim.Adadelta(content_discriminator.parameters(), args.lr,
                                            weight_decay=args.l2_wd)
    # Build dataset
    train_dataset = FontDataset(args.train_dir, rand=True)
    dev_dataset = FontDataset(args.dev_dir, rand=True)

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size)

    # Set up outputs folder
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    if not os.path.isdir('outputs/{}'.format(args.name)):
        os.mkdir('outputs/{}'.format(args.name))

    # steps_til_eval = len(train_dataset)
    steps_til_eval = 0
    epoch = step // len(train_dataset)
    while epoch < args.epochs:
        log.info('Starting epoch {}...'.format(epoch))

        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for orig, condition, target, false_target in train_loader:

                curr_batch_size, c, h, w = orig.shape
                progress_bar.update(curr_batch_size)

                orig = orig.to(device)
                condition = condition.to(device)
                target = target.to(device)
                false_target = false_target.to(device)

                # Update discriminator
                style_disc_optimizer.zero_grad()
                content_disc_optimizer.zero_grad()

                # Maybe we don't need false target at all
                style_real_loss = get_disc_loss(style_discriminator, torch.cat(
                    [orig, target, false_target]), curr_batch_size, criterion)
                content_real_loss = get_disc_loss(content_discriminator, torch.cat(
                    [condition, target, orig]), curr_batch_size, criterion)

                gen_imgs = F.gumbel_softmax(
                    generator(orig, condition).detach())

                if args.skip_epochs and epoch < args.skip_epochs:
                    style_loss = style_real_loss
                    content_loss = content_real_loss
                else:
                    style_fake_loss = get_disc_loss(style_discriminator, torch.cat(
                        [orig, target, gen_imgs]), curr_batch_size, criterion)
                    content_fake_loss = get_disc_loss(content_discriminator, torch.cat(
                        [condition, target, gen_imgs]), curr_batch_size, criterion)

                    style_loss = (style_real_loss + style_fake_loss) / 2
                    content_loss = (content_real_loss + content_fake_loss) / 2
                # To pass into discriminator: generated + original font
                # output_fake = torch.cat([gen_imgs, orig], dim=1)

                # disc_pred = discriminator(torch.cat([output_real, output_fake, output_false], dim=0)).squeeze(
                #     dim=1)  # shape = (batch_size,)
                # # First [batch_size] images are of the real
                # disc_real = disc_pred[:curr_batch_size]
                # # Rest are of fake images
                # disc_fake = disc_pred[curr_batch_size:]

                # labels_real = torch.ones_like(disc_real)
                # labels_fake = torch.zeros_like(disc_fake)

                # disc_real_loss = criterion(disc_real, labels_real)  # BCE Loss
                # disc_fake_loss = criterion(disc_fake, labels_fake)  # BCE Loss

                # disc_loss = (disc_real_loss + disc_fake_loss) / 2

                style_loss.backward(retain_graph=True)
                content_loss.backward(retain_graph=True)
                style_disc_optimizer.step()
                content_disc_optimizer.step()

                tbx.add_scalar('train/DiscContentLoss',
                               content_loss.item(), step)
                tbx.add_scalar('train/DiscStyleLoss',
                               content_loss.item(), step)
                tbx.add_scalar('train/DiscLoss',
                               style_loss.item() + content_loss.item(), step)

                # Train generator
                gen_optimizer.zero_grad()
                gen_imgs = generator(orig, condition)
                # To pass into discriminator: generated + original font
                style_pred = style_discriminator(torch.cat([orig, gen_imgs]))
                content_pred = content_discriminator(
                    torch.cat([condition, gen_imgs]))

                # Objective - minimize distance
                style_loss = l2_dist(
                    style_pred[:curr_batch_size], style_pred[curr_batch_size:]).mean()
                content_loss = l2_dist(
                    content_pred[:curr_batch_size], content_pred[curr_batch_size:]).mean()

                # gen_loss = style_loss + content_loss
                gen_loss = style_loss

                gen_loss.backward()
                gen_optimizer.step()

                progress_bar.set_postfix(epoch=epoch, GenLoss=gen_loss.item())
                tbx.add_scalar('train/GenLoss', gen_loss.item(), step)
                tbx.add_scalar('train/GenStyleLoss', style_loss.item(), step)
                tbx.add_scalar('train/GenContentLoss',
                               content_loss.item(), step)

                step += curr_batch_size

                steps_til_eval -= curr_batch_size
                if steps_til_eval <= 0:
                    save_outputs(gen_imgs, orig, condition, target,
                                 step, 'outputs/{}/train'.format(args.name))
                    steps_til_eval = len(train_dataset)

        # Save generator outputs and evaluate
        log.info('Evaluating')
        stats = evaluate(generator, style_discriminator,
                         content_discriminator, dev_loader, device, criterion)
        for key, val in stats.items():
            tbx.add_scalar('dev/{}'.format(key), val, step)

        epoch += 1

    return 0


def get_disc_loss(disc, x, curr_batch_size, criterion, margin=None):
    feats = disc(x)  # shape = (batch_size, hidden_size)

    anchor = feats[:curr_batch_size]
    positive = feats[curr_batch_size:curr_batch_size * 2]
    negative = feats[curr_batch_size*2:]

    if margin:
        real_correct = l2_dist(anchor, positive) <= margin
        fake_correct = l2_dist(anchor, negative) > margin
        return criterion(anchor, positive, negative), real_correct.sum().float(), fake_correct.sum().float()

    return criterion(anchor, positive, negative)


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


def l2_dist(anchor, samples):
    return torch.sqrt(((anchor - samples) ** 2).sum(dim=1))


def evaluate(generator, style_discriminator, content_discriminator, dataloader, device, criterion):
    style_discriminator.eval()
    content_discriminator.eval()
    generator.eval()

    disc_style_loss = 0
    disc_content_loss = 0
    gen_style_loss = 0
    gen_content_loss = 0

    disc_style_correct = 0  # Accuracy of style discriminator against ground truths
    disc_content_correct = 0  # Accuracy of content discriminator against ground truths

    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for orig, condition, target, false_target in dataloader:
            curr_batch_size = len(orig)
            progress_bar.update(curr_batch_size)

            # Evaluate discriminator
            orig = orig.to(device)
            target = target.to(device)
            condition = condition.to(device)
            false_target = false_target.to(device)

            style_real_loss, style_real_correct, style_fake_correct = get_disc_loss(
                style_discriminator, torch.cat([orig, target, false_target]), curr_batch_size, criterion, margin=4.0)

            content_real_loss, content_real_correct, content_fake_correct = get_disc_loss(
                content_discriminator, torch.cat([condition, target, orig]), curr_batch_size, criterion, margin=4.0)

            disc_style_loss += style_real_loss * curr_batch_size
            disc_content_loss += content_real_loss * curr_batch_size

            disc_style_correct += style_real_correct + style_fake_correct
            disc_content_correct += content_real_correct + content_fake_correct

            # Evaluate generator
            gen_imgs = F.gumbel_softmax(generator(orig, condition).detach())

            style_pred = style_discriminator(torch.cat([orig, gen_imgs]))
            content_pred = content_discriminator(
                torch.cat([condition, gen_imgs]))

            gen_style_loss += l2_dist(style_pred[:curr_batch_size],
                                      style_pred[curr_batch_size:]).sum()
            gen_content_loss += l2_dist(
                content_pred[:curr_batch_size], content_pred[curr_batch_size:]).sum()

    style_discriminator.train()
    content_discriminator.train()
    generator.train()

    return {'disc_style_loss': disc_style_loss.item() / len(dataloader.dataset),
            'disc_content_loss': disc_content_loss.item() / len(dataloader.dataset),
            'gen_style_loss': gen_style_loss.item() / len(dataloader.dataset),
            'gen_content_loss': gen_content_loss.item() / len(dataloader.dataset),
            'disc_style_acc': disc_style_correct.item() / (2 * len(dataloader.dataset)),
            'disc_content_acc': disc_content_correct.item() / (2 * len(dataloader.dataset)),
            }


if __name__ == '__main__':
    main()
