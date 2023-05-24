import argparse
import torch
import yaml
import os

from torchvision import utils
#from model import Generator
from model_drum import Generator
from tqdm import tqdm
from util import *
import numpy as np
import soundfile as sf
import sys
sys.path.append('./melgan')
from modules import Generator_melgan

def generate(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, vocoder, vmean, vstd):
    with torch.no_grad():
        g_ema.eval()
        t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
        print(t_dict_list)
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)
            print(sample.shape)

            if not os.path.exists('sample'):
                os.makedirs('sample')

            de_norm = sample.squeeze() * vstd + vmean
            audio_output = vocoder(de_norm)
            sf.write(f'sample/{str(i).zfill(6)}.wav', audio_output.squeeze().detach().cpu().numpy(), 44100)

            utils.save_image(
                sample,
                f'sample/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

def generate_from_latent(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noise):
    with torch.no_grad():
        g_ema.eval()
        slice_latent = latent[0,:]
        slce_latent = slice_latent.unsqueeze(0)
        print(slice_latent.size())
        for i in tqdm(range(args.pics)):
            t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
            sample, _ = g_ema([slce_latent], input_is_latent=True, noise=noises, transform_dict_list=t_dict_list)

            if not os.path.exists('sample'):
                os.makedirs('sample')

            utils.save_image(
                sample,
                f'sample/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

def read_yaml(fp):
    with open(fp) as file:
        # return yaml.load(file)
        return yaml.load(file, Loader=yaml.Loader)
    
def load_vocoder(vocoder_path, device):
    feat_dim = 80
    mean_fp = f'{args.data_path}/mean.mel.npy'
    std_fp = f'{args.data_path}/std.mel.npy'
    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1).to(device)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1).to(device)
    vocoder_config_fp = './melgan/args.yml'
    vocoder_config = read_yaml(vocoder_config_fp)

    n_mel_channels = vocoder_config.n_mel_channels
    ngf = vocoder_config.ngf
    n_residual_layers = vocoder_config.n_residual_layers
    sr=44100

    vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(device)
    vocoder.eval()

    #vocoder_param_fp = os.path.join('./melgan', 'best_netG.pt')
    vocoder.load_state_dict(torch.load(vocoder_path))

    return vocoder, mean, std


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--config', type=str, default="configs/example_transform_config.yaml")
    parser.add_argument('--load_latent', type=str, default="") 
    parser.add_argument('--clusters', type=str, default="configs/example_cluster_dict.yaml")
    parser.add_argument('--data_path', type=str, default="./data/idm/")
    parser.add_argument('--vocoder_path', type=str, default="./melgan/best_netG.pt")
    

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    yaml_config = {}
    with open(args.config, 'r') as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cluster_config = {}
    if args.clusters != "":
        with open(args.clusters, 'r') as stream:
            try:
                cluster_config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    new_state_dict = g_ema.state_dict()
    checkpoint = torch.load(args.ckpt)
    
    ext_state_dict  = torch.load(args.ckpt)['g_ema']
    new_state_dict.update(ext_state_dict)
    g_ema.load_state_dict(new_state_dict)
    g_ema.eval()
    g_ema.to(device)

    # MELGAN vocoder
    vocoder, vmean, vstd = load_vocoder(args.vocoder_path, device=device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    layer_channel_dims = create_layer_channel_dim_dict(args.channel_multiplier)
    transform_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
    
    if args.load_latent == "":
        generate(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, vocoder, vmean, vstd)
    else:
        latent=torch.load(args.load_latent)['latent']
        noises=torch.load(args.load_latent)['noises']
        generate_from_latent(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noises)
    
    config_out = {}
    config_out['transforms'] = yaml_config['transforms']
    with open(r'sample/config.yaml', 'w') as file:
        documents = yaml.dump(config_out, file)

