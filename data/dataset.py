import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        # Filter out rows where the folder is missing
        valid_indices = []
        invalid_ids = []
        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx]
            poke_id = str(row["id"])
            folder_found = False
            for f in os.listdir(self.root_dir):
                if f.startswith(f"{poke_id}-"):
                    folder_found = True
                    break
            if folder_found:
                valid_indices.append(idx)
            else:
                invalid_ids.append(poke_id)

        self.df = dataframe.iloc[valid_indices].reset_index(drop=True)
        self.transform = transform

        print(f"Original dataset size: {len(dataframe)}")
        print(f"Filtered dataset size: {len(self.df)}")
        print(f"Number of entries filtered out: {len(invalid_ids)}")
        if invalid_ids:
            print(f"First 10 filtered IDs: {invalid_ids[:10]}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        poke_id = str(row["id"])

        folder = None
        for f in os.listdir(self.root_dir):
            if f.startswith(f"{poke_id}-"):
                folder = os.path.join(self.root_dir, f)
                break

        if folder is None:
            # This should not happen after filtering, but keep for safety
            raise FileNotFoundError(f"Brak folderu dla id={poke_id}")

        sprite_dir = os.path.join(folder, "front", "normal")
        files = [os.path.join(sprite_dir, fn) for fn in os.listdir(sprite_dir) if fn.endswith(".png")]
        if len(files) == 0:
            raise FileNotFoundError(f"Brak sprite'ów w {sprite_dir}")

        img_path = files[0]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = row.drop(["id", "name"]).values.astype("float32")
        label = torch.tensor(label)

        return img, label