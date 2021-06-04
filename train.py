import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from utils.dataset import MultiResolutionDataset
from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from torch.utils.tensorboard import SummaryWriter


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def prepare_noise(n_sample, args, device):
    if not args.w_space:
        return torch.randn(n_sample, args.token, args.latent, device=device)
    else:
        if args.mixing > 0 and random.random() < args.mixing:
            inject_index = random.randint(1, args.token - 1)
            noise1 = torch.randn(n_sample, 1, args.latent, device=device).repeat(1, inject_index, 1)
            noise2 = torch.randn(n_sample, 1, args.latent, device=device).repeat(1, args.token - inject_index, 1)
            noise = torch.cat([noise1, noise2], 1)
        else:
            noise = torch.randn(n_sample, 1, args.latent, device=device).repeat(1, args.token, 1)
        return noise


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def save_similarity(similarity_list, path, iteration):
    for layer in range(len(similarity_list)):
        similarity = similarity_list[layer]
        for head in range(4):
            similarity_select = torch.mean(similarity, dim=0)[head]
            plt.imshow(similarity_select.cpu().numpy())
            plt.colorbar()
            plt.savefig(path + f"/{str(iteration).zfill(6)}_{str(layer).zfill(2)}_{str(head).zfill(2)}.png")
            plt.close()


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, tb_writer, exp_name):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = prepare_noise(args.n_sample, args, device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = prepare_noise(args.batch, args, device)
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)

        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = prepare_noise(args.batch, args, device)
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = prepare_noise(path_batch_size, args, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            check_path = None
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                    f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                )
            )

            tb_writer.add_scalars("Loss", {'d': d_loss_val, 'g': g_loss_val, 'r1': r1_val}, i)
            tb_writer.add_scalars("Path", {'path': path_loss_val, 'mean path': mean_path_length_avg}, i)
            tb_writer.add_scalars("Score", {"real": real_score_val, "fake": fake_score_val}, i)

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Generator': g_loss_val,
                        'Discriminator': d_loss_val,
                        'R1': r1_val,
                        'Path Length Regularization': path_loss_val,
                        'Mean Path Length': mean_path_length,
                        'Real Score': real_score_val,
                        'Fake Score': fake_score_val,
                        'Path Length': path_length_val,
                    }
                )

            if i % 500 == 0:
                similarity_path = exp_name + f"similarity"
                sample_path = exp_name + f'sample'
                check_path = exp_name + f'checkpoint'
                if not os.path.exists(similarity_path):
                    os.makedirs(similarity_path)
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)
                if not os.path.exists(check_path):
                    os.makedirs(check_path)

                with torch.no_grad():
                    g_ema.eval()

                    if not args.w_space:
                        similarity_list = g_ema.get_similarity(sample_z)
                        save_similarity(similarity_list, similarity_path, i)

                    sample, _ = g_ema(sample_z)
                    utils.save_image(
                        sample,
                        sample_path + f'/{str(i).zfill(6)}.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    check_path + f'/{str(i).zfill(6)}.pt',
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--wandb', action='store_true')
    # use w++ space by default
    parser.add_argument('--w_space', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    # 14 style codes for resolution 256x256, 18 style codes for resolution 1024x1024
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.token, args.n_mlp, w_space=args.w_space, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.token, args.n_mlp, w_space=args.w_space, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.local_rank == 0:
        print(f"Generator params count: {sum([m.numel() for m in generator.parameters()])}")
        print(f"Discriminator params count: {sum([m.numel() for m in discriminator.parameters()])}")

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)

        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='stylegan 2')

    # instance tensorboard writer
    tensorboard_writer = None
    if get_rank() == 0:
        _tensorboard_path = args.exp_name + "tensorboard_log"
        tensorboard_writer = SummaryWriter(_tensorboard_path)
        if not os.path.exists(_tensorboard_path):
            os.makedirs(_tensorboard_path)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, tensorboard_writer, args.exp_name)

    if get_rank() == 0:
        tensorboard_writer.close()
