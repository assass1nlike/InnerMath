from __future__ import annotations

import argparse
from pathlib import Path

from innermath.embedder import DEFAULT_MODEL, ImageEmbedder, load_embedding
from innermath.gallery import (
    build_gallery_embeddings,
    load_gallery_embeddings,
    rank_nearest,
    write_results,
)
from innermath.vector_ops import OperationConfig, apply_operation, list_operations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply mathematical vector operations to two embeddings or images, "
            "then map the result back to images by nearest-neighbor retrieval."
        )
    )
    parser.add_argument("--image-a", help="First image path. Requires optional image dependencies.")
    parser.add_argument("--image-b", help="Second image path. Requires optional image dependencies.")
    parser.add_argument("--embedding-a", help="First embedding file (.npy or .json).")
    parser.add_argument("--embedding-b", help="Second embedding file (.npy or .json).")
    parser.add_argument(
        "--operation",
        required=True,
        choices=list_operations(),
        help="Vector operation to apply.",
    )
    parser.add_argument("--t", type=float, default=0.5, help="Interpolation parameter for lerp/slerp.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for weighted-sum.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for weighted-sum.")
    parser.add_argument(
        "--no-normalize-output",
        action="store_true",
        help="Do not L2-normalize the operation result before retrieval.",
    )
    parser.add_argument("--gallery-dir", help="Directory of candidate images for retrieval decoding.")
    parser.add_argument(
        "--gallery-index",
        help="Path to a cached .npz gallery index. Created if it does not exist and --gallery-dir is set.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face image embedding model.")
    parser.add_argument("--device", help="Torch device, such as cpu or cuda.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest images to return.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for target vector and results.")
    return parser


def needs_image_embedder(args: argparse.Namespace) -> bool:
    if args.image_a or args.image_b:
        return True
    gallery_index = Path(args.gallery_index) if args.gallery_index else Path(args.output_dir) / "gallery_index.npz"
    return bool(args.gallery_dir and not gallery_index.exists())


def load_input_vector(
    image_path: str | None,
    embedding_path: str | None,
    embedder: ImageEmbedder | None,
):
    if embedding_path:
        return load_embedding(embedding_path)
    if image_path:
        if embedder is None:
            raise RuntimeError("Image input requires an ImageEmbedder.")
        return embedder.embed_image(image_path)
    raise ValueError("Provide either an image path or an embedding file.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not ((args.image_a or args.embedding_a) and (args.image_b or args.embedding_b)):
        parser.error("Provide both inputs via --image-a/--embedding-a and --image-b/--embedding-b.")

    embedder = ImageEmbedder(args.model, args.device) if needs_image_embedder(args) else None

    vector_a = load_input_vector(args.image_a, args.embedding_a, embedder)
    vector_b = load_input_vector(args.image_b, args.embedding_b, embedder)
    config = OperationConfig(
        t=args.t,
        alpha=args.alpha,
        beta=args.beta,
        normalize_output=not args.no_normalize_output,
    )
    target = apply_operation(args.operation, vector_a, vector_b, config)

    results = []
    if args.gallery_dir or args.gallery_index:
        index_path = Path(args.gallery_index) if args.gallery_index else Path(args.output_dir) / "gallery_index.npz"
        if index_path.exists():
            gallery_vectors, gallery_paths = load_gallery_embeddings(index_path)
        elif args.gallery_dir:
            if embedder is None:
                embedder = ImageEmbedder(args.model, args.device)
            gallery_vectors, gallery_paths = build_gallery_embeddings(args.gallery_dir, index_path, embedder)
        else:
            parser.error("--gallery-index does not exist; provide --gallery-dir to build it.")

        results = rank_nearest(gallery_vectors, gallery_paths, target, args.top_k)

    write_results(args.output_dir, target, results, args.operation)

    if results:
        print(f"Saved target vector and top-{len(results)} results to {args.output_dir}")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result.score:.6f} {result.path}")
    else:
        print(f"Saved target vector to {args.output_dir}; no gallery search was requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
