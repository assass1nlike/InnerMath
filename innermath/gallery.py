from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from innermath.embedder import ImageEmbedder, save_embedding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class SearchResult:
    path: str
    score: float


def iter_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Gallery directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D gallery matrix, got shape {matrix.shape}.")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"Gallery dimension {matrix.shape[1]} does not match query dimension {query.shape[0]}."
        )

    matrix_norm = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denominator = np.maximum(matrix_norm * query_norm, 1e-12)
    return matrix @ query / denominator


def rank_nearest(
    vectors: np.ndarray,
    paths: list[str | Path],
    query: np.ndarray,
    top_k: int = 5,
) -> list[SearchResult]:
    if len(paths) != len(vectors):
        raise ValueError("Number of paths and vectors must match.")
    scores = cosine_scores(vectors, query)
    order = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(path=str(Path(paths[index])), score=float(scores[index]))
        for index in order
    ]


def build_gallery_embeddings(
    gallery_dir: str | Path,
    output_npz: str | Path,
    embedder: ImageEmbedder,
) -> tuple[np.ndarray, list[str]]:
    image_paths = iter_images(gallery_dir)
    if not image_paths:
        raise ValueError(f"No image files found in gallery: {gallery_dir}")

    vectors = [embedder.embed_image(path) for path in image_paths]
    matrix = np.vstack(vectors).astype(np.float32)
    path_strings = [str(path) for path in image_paths]

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, vectors=matrix, paths=np.asarray(path_strings))
    return matrix, path_strings


def load_gallery_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(Path(path), allow_pickle=False)
    return np.asarray(data["vectors"], dtype=np.float64), [str(item) for item in data["paths"]]


def write_results(
    output_dir: str | Path,
    query: np.ndarray,
    results: list[SearchResult],
    operation: str,
    copy_best: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_embedding(output_dir / "target.npy", query)

    manifest = {
        "operation": operation,
        "target_embedding": "target.npy",
        "results": [asdict(result) for result in results],
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    if copy_best and results:
        best_path = Path(results[0].path)
        if best_path.exists():
            shutil.copy2(best_path, output_dir / f"best{best_path.suffix.lower()}")
