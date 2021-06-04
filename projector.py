import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from model import Generator
from train import data_sampler, sample_data
from utils import lpips
from utils.dataset import MultiResolutionDataset


def noise_regularize_(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * torch.unsqueeze(strength, -1)

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to('cpu')
            .numpy()
    )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./projection')

    args = parser.parse_args()

    n_mean_latent = 10000
    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    args.n_mlp = 8
    args.w_space = False

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = MultiResolutionDataset(args.dataset, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    loader = sample_data(loader)
    imgs = next(loader).to(device)

    g_ema = Generator(args.size, args.latent, args.token, args.n_mlp, w_space=args.w_space)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():

        noise_sample = torch.randn(n_mean_latent, args.token, args.latent, device=device)
        noise_sample = torch.cat([noise_sample, g_ema.token.repeat(noise_sample.size()[0], 1, 1)], 2)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum([0, 2]) / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    noise_single = g_ema.make_noise()
    noises = []
    for noise in noise_single:
        noises.append(noise.repeat(args.batch, 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch, 1, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []
    perceptual_values = []
    noise_values = []
    mse_values = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, rampdown=args.lr_rampdown, rampup=args.lr_rampup)
        optimizer.param_groups[0]['lr'] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength)

        img_gen, _ = g_ema(latent_n, input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize_(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        if (i + 1) % 10 == 0:
            perceptual_values.append(p_loss.item())
            noise_values.append(n_loss.item())
            mse_values.append(mse_loss.item())

        pbar.set_description(
            (
                f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
                f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'
            )
        )

    img_gen, _ = g_ema(latent_path[-1], input_is_latent=True, noise=noises)

    img_or = make_image(imgs)
    img_ar = make_image(img_gen)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f'latents.npy'), latent_path[-1].cpu().numpy())
    np.save(os.path.join(args.output_dir, f'perceptual.npy'), perceptual_values)
    np.save(os.path.join(args.output_dir, f'noise.npy'), noise_values)
    np.save(os.path.join(args.output_dir, f'mse.npy'), mse_values)
    for i in range(args.batch):
        img1 = Image.fromarray(img_or[i])
        img1.save(os.path.join(args.output_dir, f'origin_{i}.png'))
        img2 = Image.fromarray(img_ar[i])
        img2.save(os.path.join(args.output_dir, f'project_{i}.png'))
