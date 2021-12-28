import argparse

import numpy as np

from src.model_asr import ASR
from src.model_translator import Ru2EnTranslator
from src.model_clip import CLIP
from src.db_embed import EmbedCosDB
from src.tg_bot import TgBot


def load_token(path):
    with open(path) as file:
        token = file.read()

    return token


def load_start_text(path):
    with open(path) as file:
        start_text = file.read()

    return start_text


def load_embeds(path):
    file = np.load(path)
    return file["paths"], file["embeds"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_path", required=True)
    parser.add_argument("--start_text_path", required=True)
    parser.add_argument("--embeds_path", required=True)
    parser.add_argument("--k_nearest", default=5)
    args = parser.parse_args()

    token = load_token(args.token_path)
    start_text = load_start_text(args.start_text_path)
    paths, embeds = load_embeds(args.embeds_path)

    asr = ASR()
    translator = Ru2EnTranslator()
    clip = CLIP()

    embed_db = EmbedCosDB(embeds, paths, k_nearest=args.k_nearest)

    tg_bot = TgBot(token, start_text, asr, translator, clip, embed_db)

    print("Bot is ready!")

    tg_bot.start()