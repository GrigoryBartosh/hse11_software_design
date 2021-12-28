import numpy as np
import faiss


class EmbedCosDB:
    def __init__(self, embeds, paths, k_nearest=5):
        self.paths = paths
        self.k_nearest = k_nearest

        d = embeds.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, 100)

        embeds = self.normalize(embeds)
        self.index.train(embeds)
        self.index.add(embeds)

    def normalize(self, embeds):
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds.astype(np.float32)
        return embeds

    def path_by_id(self, ids):
        shape = ids.shape

        ids = ids.reshape(-1)
        id_paths = self.paths[ids]

        return id_paths.reshape(shape)

    def find(self, embeds):
        embeds = self.normalize(embeds)
        _, ids = self.index.search(embeds, self.k_nearest)
        return self.path_by_id(ids)