# train.py
import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset import PokemonDataset
from models.unet import UNet
from models.diffusion import Diffusion

def train(
    dataset,
    model,
    diffusion,
    device="cuda",
    num_epochs=3500,
    batch_size=32,
    lr=2e-4,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    iters = 0
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, cond in pbar:
            imgs = imgs.to(device)
            cond = cond.to(device)
            loss = diffusion.training_step(imgs, cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
            pbar.set_postfix({"loss": loss.item(), "iters": iters})

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                _, cond_batch = next(iter(dataloader))
                cond_batch = cond_batch[:8].to(device)

                samples = diffusion.sample(cond=cond_batch, batch_size=8)
                samples = (samples * 0.5 + 0.5).clamp(0, 1)

            fig, axes = plt.subplots(1, 8, figsize=(16, 2))
            for i in range(8):
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                axes[i].imshow(img)
                axes[i].axis("off")
            plt.show()
            model.train()


if __name__ == "__main__":
    from pandas import read_csv
    import torchvision.transforms as T

    device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    LABELS_PATH = "C:/Users/danil/PycharmProjects/ProjectMon/data/data/processed/pokemon_labels.csv"
    IMG_DIR = "C:/Users/danil/PycharmProjects/ProjectMon/data/data/raw/pokemon_images/sprites"

    pokemonData = pd.read_csv(LABELS_PATH)
    transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = PokemonDataset(root_dir=IMG_DIR, dataframe=pokemonData, transform=transform)

    cond_dim = dataset[0][1].shape[0]
    model = UNet(img_channels=3, base_channels=128, time_emb_dim=256, cond_dim=cond_dim).to(device)
    diffusion = Diffusion(model, img_size=96, device=device, timesteps=1000)

    train(dataset, model, diffusion, device=device)
