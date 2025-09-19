import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "pokemon_images")
INPUT_CSV = os.path.join(RAW_DIR, "pokedex.csv")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pokemon_labels.csv")


def prepare_labels(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"[ERROR] {input_path} not found. "
            f"Download dataset: python data/download_pokemon.py"
        )

    print(f"[INFO] Loading data from {input_path}...")
    pokemonData = pd.read_csv(input_path)

    pokemonData = pokemonData[pokemonData['image_fn'] != "[]"].reset_index(drop=True)

    pokemonData = pokemonData.drop(columns=['pokedex_id', 'image_fn'])

    pokemonData["id"] = pokemonData["id"].astype(str).str.zfill(4)

    one_hot_types = pd.get_dummies(pokemonData[["type1", "type2"]])
    one_hot_types = one_hot_types.groupby(one_hot_types.columns.str.split("_").str[1], axis=1).sum()
    one_hot_types = (one_hot_types > 0).astype(int)
    pokemonData = pokemonData.drop(columns=['type1', 'type2'])

    one_hot_shapes = pd.get_dummies(pokemonData["shape"], prefix="shape").astype(int)
    one_hot_colors = pd.get_dummies(pokemonData["primary_color"], prefix="color").astype(int)
    pokemonData = pokemonData.drop(columns=['shape', 'primary_color'])

    pokemonData = pd.concat([pokemonData, one_hot_types, one_hot_shapes, one_hot_colors], axis=1)

    binary_cols = ["legendary", "mega_evolution", "alolan_form", "galarian_form", "gigantamax"]
    for col in binary_cols:
        pokemonData[col] = pokemonData[col].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pokemonData.to_csv(output_path, index=False)
    print(f"[OK] Labels saved to {output_path}")


if __name__ == "__main__":
    prepare_labels()
