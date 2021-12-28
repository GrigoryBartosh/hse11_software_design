import os
import argparse

import numpy as np
from src.model_clip import CLIP

from tqdm.auto import trange


def read_paths(data_path):
    paths = os.listdir(args.data_path)
    paths = [os.path.join(args.data_path, p) for p in paths]
    paths = np.array(paths)
    return paths


def build_embeds(clip, paths, bs):
    embeds = []

    n = len(paths)
    bs = args.batch_size
    for i in trange(0, n, bs):
        batch_paths = paths[i:i + bs]
        batch_embeds = clip.encode_images(batch_paths)
        embeds += [batch_embeds]

    embeds = np.concatenate(embeds)

    return embeds


def save_embeds(path, paths, embeds):
    np.savez(path, paths=paths, embeds=embeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--embeds_path", required=True)
    parser.add_argument("--batch_size", default=128)
    args = parser.parse_args()

    clip = CLIP()

    paths = read_paths(args.data_path)

    embeds = build_embeds(clip, paths, args.batch_size)

    save_embeds(args.embeds_path, paths, embeds)