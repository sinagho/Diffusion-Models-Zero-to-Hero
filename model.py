############################# Sina Ghorbani Kolahi ###############################
                        #Implementing DDPM from Scratch#
                # Original Paper: https://arxiv.org/abs/2006.11239
##################################################################################

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import *
from unet import *
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.INFO, 
                    format="%(asctime)s - %(levelname)s: %(message)s",
                    datefmt="%I:%M:%S",
                    filename= 'ddpm.log')

class Diffusion(nn.Module):
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 64, device= 'cuda'):
        super().__init__()

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(start = self.beta_start, end = self.beta_end, steps= self.noise_steps)
    
    def noisy_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None, None, None]
        one_minus_sqrt_alpha_hat = torch.sqrt(input=1. -self.alpha_hat[t])[:,None, None, None]
        epsilon = torch.randn_like(x)

        return sqrt_alpha_hat * x + one_minus_sqrt_alpha_hat * epsilon, epsilon
    
    def sample_time_steps(self, n):
        return torch.randint(low= 1, high=self.noise_steps, size = (n,))
    
    def sample(self, model, n):
        logging.info(f"sampling {n} new images")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)),position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1: 
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                x = 1 / (torch.sqrt(alpha)) *(x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * z
        model.train()
        x = (x.clamp(-1,1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x 

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size= args.img_size, device= device)
    logger = SummaryWriter(os.path.join('runs', args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.noisy_image(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i) 

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 20
    args.batch_size = 6
    args.img_size = 64
    args.dataset_path = r"E:\Data"
    
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()