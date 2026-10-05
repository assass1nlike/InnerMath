from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def load_embedding(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if path.suffix.lower() == ".json":
        import json

        with path.open("r", encoding="utf-8") as file:
            return np.asarray(json.load(file), dtype=np.float64).reshape(-1)
    raise ValueError(f"Unsupported embedding file format: {path.suffix}. Use .npy or .json.")


def save_embedding(path: str | Path, vector: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, np.asarray(vector, dtype=np.float32))
        return
    if path.suffix.lower() == ".json":
        import json

        with path.open("w", encoding="utf-8") as file:
            json.dump(np.asarray(vector, dtype=float).tolist(), file)
        return
    raise ValueError(f"Unsupported embedding file format: {path.suffix}. Use .npy or .json.")


def _feature_tensor(output):
    if hasattr(output, "detach"):
        return output
    for attr in ("image_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract image features from output type {type(output)!r}.")


class ImageEmbedder:
    """Lazy Transformers image embedder.

    The package core stays lightweight. This class imports Pillow, Torch, and
    Transformers only when image embedding is requested.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Image embedding requires optional dependencies. "
                "Install with: python -m pip install -e .[image]"
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_image(self, path: str | Path, normalize: bool = True) -> np.ndarray:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                output = self.model.get_image_features(**inputs)
            else:
                output = self.model(**inputs)
            features = _feature_tensor(output)

        vector = features.detach().cpu().numpy()[0].astype(np.float64)
        if normalize:
            length = np.linalg.norm(vector)
            if length > 0:
                vector = vector / length
        return vector
