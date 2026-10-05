from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a new image from a weighted blend of image embeddings with Kandinsky. "
            "This is a generative decoder path, not nearest-neighbor retrieval."
        )
    )
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="low quality, bad quality, blurry")
    parser.add_argument("--model", default="kandinsky-community/kandinsky-2-1")
    parser.add_argument("--prior-model", default="kandinsky-community/kandinsky-2-1-prior")
    parser.add_argument("--output", default="outputs/generated/kandinsky_blend.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> int:
    try:
        from diffusers import KandinskyPipeline, KandinskyPriorPipeline
    except ImportError as exc:
        raise ImportError("Install generation dependencies with: python -m pip install -e .[generate]") from exc

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)

    image_a = load_rgb(args.image_a)
    image_b = load_rgb(args.image_b)
    total = args.weight_a + args.weight_b
    if total == 0:
        raise ValueError("weight-a + weight-b must not be zero.")
    weights = [args.weight_a / total, args.weight_b / total]

    prior = KandinskyPriorPipeline.from_pretrained(args.prior_model, torch_dtype=dtype)
    pipe = KandinskyPipeline.from_pretrained(args.model, torch_dtype=dtype)
    prior = prior.to(device)
    pipe = pipe.to(device)

    prior_out = prior.interpolate([image_a, image_b], weights)
    image = pipe(
        args.prompt,
        **prior_out,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved generated image to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
