import argparse
import os
import pickle

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics.calc_inception import load_patched_inception_v3
from model import Generator


@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 14, 512, device=device)
        img, _ = generator(latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None, required=True)
    parser.add_argument('--ckpt', default='./checkpoint')

    args = parser.parse_args()

    _tensorboard_path = "tensorboard_log"
    tensorboard_writer = SummaryWriter(_tensorboard_path)
    if not os.path.exists(_tensorboard_path):
        os.makedirs(_tensorboard_path)

    if os.path.isdir(args.ckpt):
        files = os.listdir(args.ckpt)
        ckpt = sorted([os.path.join(args.ckpt, x) for x in files])
        ckpt = list(filter(lambda x: int(x.split('/')[-1].split('.')[0])>=args.start_num, ckpt))
        print(args.ckpt)
    else:
        ckpt = [args.ckpt]

    print(ckpt)

    for model_path in ckpt:
        iteration = int(os.path.splitext(os.path.basename(model_path))[0])
        print(f'Iteration = {iteration}')

        g = Generator(args.size, 512, 8).to(device)
        model = torch.load(model_path, map_location='cpu')
        g.load_state_dict(model['g_ema'])
        g = nn.DataParallel(g)
        g.eval()

        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g.mean_latent(args.truncation_mean)

        else:
            mean_latent = None

        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()

        features = extract_feature_from_samples(
            g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        ).numpy()
        print(f'extracted {features.shape[0]} features')

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        with open(args.inception, 'rb') as f:
            embeds = pickle.load(f)
            real_mean = embeds['mean']
            real_cov = embeds['cov']

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

        print('fid:', fid)
        tensorboard_writer.add_scalar("FID", fid, iteration)
    tensorboard_writer.close()
