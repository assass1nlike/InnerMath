from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from innermath.embedder import DEFAULT_MODEL, ImageEmbedder
from innermath.gallery import (
    build_gallery_embeddings,
    load_gallery_embeddings,
    rank_nearest,
)
from innermath.vector_ops import OperationConfig, apply_operation


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def expand_params(params: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not params:
        return [{}]

    keys = list(params)
    values = [value if isinstance(value, list) else [value] for value in params.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def operation_label(operation: str, params: dict[str, Any]) -> str:
    if not params:
        return operation
    suffix = ",".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{operation}({suffix})"


def config_for_params(params: dict[str, Any], normalize_output: bool) -> OperationConfig:
    return OperationConfig(
        t=float(params.get("t", 0.5)),
        alpha=float(params.get("alpha", 1.0)),
        beta=float(params.get("beta", 1.0)),
        normalize_output=normalize_output,
    )


def path_rank(results, target_path: str) -> int | None:
    normalized_target = str(Path(target_path))
    for index, result in enumerate(results, start=1):
        if str(Path(result.path)) == normalized_target:
            return index
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    lines = [
        "# Image Embedding Eval Summary",
        "",
        f"- model: `{config.get('model', DEFAULT_MODEL)}`",
        f"- gallery: `{config['gallery_dir']}`",
        f"- top_k: `{config.get('top_k', 5)}`",
        f"- records: `{len(records)}`",
        "",
        "| pair | operation | best | score | rank A | rank B |",
        "|---|---|---|---:|---:|---:|",
    ]

    for record in records:
        best = record["results"][0] if record["results"] else {"path": "", "score": 0.0}
        rank_a = record["input_a_rank"] if record["input_a_rank"] is not None else "-"
        rank_b = record["input_b_rank"] if record["input_b_rank"] is not None else "-"
        lines.append(
            "| {pair} | `{operation}` | `{best}` | {score:.6f} | {rank_a} | {rank_b} |".format(
                pair=record["pair"],
                operation=record["operation_label"],
                best=Path(best["path"]).name,
                score=float(best["score"]),
                rank_a=rank_a,
                rank_b=rank_b,
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tile(path: str | Path, size: int = 180) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    fill: tuple[int, int, int] = (30, 30, 30),
) -> None:
    font = ImageFont.load_default()
    words = text.replace("\\", "/").split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:4]):
        draw.text((xy[0], xy[1] + index * 14), line, fill=fill, font=font)


def write_preview(record: dict[str, Any], path: Path, top_n: int = 5) -> None:
    tile_size = 180
    label_height = 96
    gap = 14
    header_height = 64
    result_count = min(top_n, len(record["results"]))
    columns = 2 + result_count
    width = columns * tile_size + (columns + 1) * gap
    height = header_height + tile_size + label_height + gap * 3
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title = f"{record['pair']} | {record['operation_label']}"
    draw.text((gap, gap), title, fill=(0, 0, 0), font=ImageFont.load_default())
    draw.text((gap, gap + 20), f"model: {record['model']}", fill=(80, 80, 80), font=ImageFont.load_default())

    tiles = [
        ("A", record["image_a"], None),
        ("B", record["image_b"], None),
    ]
    for rank, result in enumerate(record["results"][:top_n], start=1):
        tiles.append((f"#{rank} {result['score']:.3f}", result["path"], result["score"]))

    y = header_height
    for column, (label, source_path, _score) in enumerate(tiles):
        x = gap + column * (tile_size + gap)
        tile = load_tile(source_path, tile_size)
        image.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline=(210, 210, 210))
        label_y = y + tile_size + 8
        draw.rectangle((x, label_y - 2, x + tile_size, label_y + label_height - 8), fill=(255, 255, 255))
        draw.text((x, label_y), label, fill=(0, 0, 0), font=ImageFont.load_default())
        filename = Path(source_path).stem
        if len(filename) > 42:
            filename = filename[:39] + "..."
        draw_wrapped_text(draw, (x, label_y + 18), filename, tile_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_preview_index(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Eval Preview Index",
        "",
        "| pair | operation | preview | best | score |",
        "|---|---|---|---|---:|",
    ]
    for record in records:
        preview = Path(record["preview"]).name
        best = record["results"][0] if record["results"] else {"path": "", "score": 0.0}
        lines.append(
            f"| {record['pair']} | `{record['operation_label']}` | "
            f"[{preview}](preview/{preview}) | `{Path(best['path']).name}` | {float(best['score']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch image embedding math evaluations.")
    parser.add_argument("config", help="YAML config path.")
    parser.add_argument("--run-name", help="Override run name.")
    parser.add_argument("--device", help="Override torch device.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild gallery index even if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    model = config.get("model", DEFAULT_MODEL)
    gallery_dir = Path(config["gallery_dir"])
    top_k = int(config.get("top_k", 5))
    normalize_output = bool(config.get("normalize_output", True))

    run_name = args.run_name or config.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(config.get("output_root", "outputs/eval_runs"))
    run_dir = output_root / run_name
    target_dir = run_dir / "target_vectors"
    preview_dir = run_dir / "preview"
    target_dir.mkdir(parents=True, exist_ok=True)

    gallery_index = Path(config.get("gallery_index", run_dir / "gallery_index.npz"))

    embedder = ImageEmbedder(model, args.device or config.get("device"))
    if gallery_index.exists() and not args.rebuild_index:
        gallery_vectors, gallery_paths = load_gallery_embeddings(gallery_index)
    else:
        gallery_vectors, gallery_paths = build_gallery_embeddings(gallery_dir, gallery_index, embedder)

    pairs = config.get("pairs", [])
    operations = config.get("operations", [])
    if not pairs:
        raise ValueError("Config must contain at least one pair.")
    if not operations:
        raise ValueError("Config must contain at least one operation.")

    embedding_cache: dict[str, np.ndarray] = {}

    def embed(path: str) -> np.ndarray:
        if path not in embedding_cache:
            embedding_cache[path] = embedder.embed_image(path)
        return embedding_cache[path]

    records: list[dict[str, Any]] = []
    counter = 0
    for pair in pairs:
        pair_name = pair["name"]
        image_a = pair["image_a"]
        image_b = pair["image_b"]
        vector_a = embed(image_a)
        vector_b = embed(image_b)

        for operation_spec in operations:
            operation = operation_spec["name"]
            for params in expand_params(operation_spec.get("params")):
                counter += 1
                op_config = config_for_params(params, normalize_output)
                target = apply_operation(operation, vector_a, vector_b, op_config)
                results = rank_nearest(gallery_vectors, gallery_paths, target, top_k)

                target_name = f"{counter:04d}_{pair_name}_{operation}.npy"
                target_path = target_dir / target_name
                np.save(target_path, target.astype(np.float32))

                record = {
                    "pair": pair_name,
                    "image_a": image_a,
                    "image_b": image_b,
                    "operation": operation,
                    "operation_label": operation_label(operation, params),
                    "params": params,
                    "model": model,
                    "target_embedding": str(target_path),
                    "target_norm": float(np.linalg.norm(target)),
                    "input_a_rank": path_rank(results, image_a),
                    "input_b_rank": path_rank(results, image_b),
                    "results": [
                        {"path": result.path, "score": result.score}
                        for result in results
                    ],
                }
                preview_path = preview_dir / f"{counter:04d}_{pair_name}_{operation}.jpg"
                write_preview(record, preview_path)
                record["preview"] = str(preview_path)
                records.append(record)
                best = results[0] if results else None
                if best:
                    print(
                        f"{counter:04d} {pair_name} {record['operation_label']} -> "
                        f"{Path(best.path).name} ({best.score:.6f})"
                    )

    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "results.jsonl", records)
    write_summary(run_dir / "summary.md", records, config)
    write_preview_index(run_dir / "preview_index.md", records)
    print(f"saved eval run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
