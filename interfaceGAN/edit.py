import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from interfaceGAN.linear_interpolation import linear_interpolate
from interfaceGAN.train_boundary import train_boundary
from model import Generator
from projector import make_image
from utils import dex

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--latent', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--num_sample', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--start_distance', type=int, default=-30)
    parser.add_argument('--end_distance', type=int, default=30)
    parser.add_argument('--steps', type=int, default=61)
    parser.add_argument('--ratio', type=float, default=0.02)

    args = parser.parse_args()

    args.style_dim = 512
    args.token_dim = 2 * (int(math.log(args.size, 2)) - 1)
    args.n_mlp = 8
    args.w_space = False

    os.makedirs(args.output_dir, exist_ok=True)

    g_ema = Generator(args.size, args.style_dim, args.token_dim, args.n_mlp, w_space=args.w_space)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)

    batch_size = args.batch_size
    num_batch = args.num_sample // batch_size
    last_batch = args.num_sample - (batch_size * num_batch)

    latents = []
    ages = []
    genders = []
    dex.eval()

    print(f"Starting to generate {args.num_sample} random samples...")
    with torch.no_grad():
        for b in tqdm(range(num_batch)):
            noise = torch.randn(batch_size, args.token_dim, args.style_dim, device=device)
            img, latent = g_ema(noise, return_latents=True)
            latents.append(latent.cpu())
            # change from RGB to GBR
            image = img[:, [2, 1, 0], :, :]
            # normalize to [0, 255]
            image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
            # estimate age
            age = dex.estimate_age(image)
            ages.append(age.cpu())

        if last_batch != 0:
            noise = torch.randn(last_batch, args.token_dim, args.style_dim, device=device)
            img, latent = g_ema(noise, return_latents=True)
            latents.append(latent.cpu())
            # change from RGB to GBR
            image = img[:, [2, 1, 0], :, :]
            # normalize to [0, 255]
            image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
            # estimate age
            age = dex.estimate_age(image)
            ages.append(age.cpu())

    print(f"{args.num_sample} random samples generated...")
    latent_codes = torch.cat(latents, dim=0).reshape(args.num_sample, -1).numpy()
    scores_age = torch.cat(ages, dim=0).reshape(args.num_sample, -1).numpy()

    chosen_num_or_ratio = args.ratio
    split_ratio = 0.7
    invalid_value = None
    boundary_age = train_boundary(latent_codes=latent_codes,
                                  scores=scores_age,
                                  chosen_num_or_ratio=chosen_num_or_ratio,
                                  split_ratio=split_ratio,
                                  invalid_value=invalid_value)
    print("Age Boundary trained...")

    latent_projected = np.load(args.latent)
    count = latent_projected.shape[0]
    latent_projected = np.reshape(latent_projected, (count, -1))
    start_distance = args.start_distance
    end_distance = args.end_distance
    steps = args.steps
    with torch.no_grad():
        for i in tqdm(range(count)):
            # edit age
            latent_interpolated = linear_interpolate(latent_projected[i:i + 1],
                                                     boundary_age,
                                                     start_distance=start_distance,
                                                     end_distance=end_distance,
                                                     steps=steps)
            for j in range(steps):
                latent = torch.from_numpy(latent_interpolated[j:j + 1]).reshape(1, -1, args.style_dim).to(device)
                img_gen, _ = g_ema(latent, input_is_latent=True)
                image = img_gen[:, [2, 1, 0], :, :]
                image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                age = dex.estimate_age(image)
                img_ar = make_image(img_gen)
                img = Image.fromarray(img_ar[0])
                img.save(os.path.join(args.output_dir, f'origin_{i}_edit_{j}_age_{round(age.cpu().numpy()[0])}.png'))
    print(f"{steps} interpolation generated for {count} samples...")
