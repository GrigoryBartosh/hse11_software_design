import torch
import clip
from PIL import Image


class CLIP:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @torch.no_grad()
    def encode_images(self, image_paths):
        images = [Image.open(path) for path in image_paths]
        images = [self.preprocess(img) for img in images]
        images = torch.stack(images, axis=0)
        images = images.to(self.device)

        embeds = self.model.encode_image(images)

        return embeds.cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts):
        texts = clip.tokenize(texts)
        texts = texts.to(self.device)

        embeds = self.model.encode_text(texts)

        return embeds.cpu().numpy()