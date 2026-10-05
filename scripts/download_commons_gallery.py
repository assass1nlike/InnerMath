from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "innermath/0.1 local-research-gallery-builder"

DEFAULT_QUERIES = [
    "cat animal",
    "dog animal",
    "car vehicle",
    "flower plant",
    "mountain landscape",
    "beach landscape",
    "apple fruit",
    "chair furniture",
]

ALLOWED_LICENSE_TOKENS = (
    "cc-by",
    "cc-by-sa",
    "cc0",
    "public domain",
    "pd-",
)


def request_json(params: dict[str, str | int]) -> dict:
    query = urlencode({**params, "format": "json", "formatversion": "2"})
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def sanitize(value: str) -> str:
    value = value.lower().replace("file:", "")
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-")[:120] or "image"


def license_allowed(metadata: dict) -> bool:
    license_short = str(metadata.get("LicenseShortName", {}).get("value", "")).lower()
    license_url = str(metadata.get("LicenseUrl", {}).get("value", "")).lower()
    usage_terms = str(metadata.get("UsageTerms", {}).get("value", "")).lower()
    joined = " ".join([license_short, license_url, usage_terms])
    return any(token in joined for token in ALLOWED_LICENSE_TOKENS)


def page_records(query: str, limit: int, batch_size: int = 25) -> list[dict]:
    params: dict[str, str | int] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": batch_size,
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 768,
    }
    data = request_json(params)
    pages = data.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        return []
    records: list[dict] = []
    for page in pages:
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        metadata = info.get("extmetadata", {})
        if not license_allowed(metadata):
            continue
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        if width < 600 or height < 400:
            continue
        aspect = max(width / max(height, 1), height / max(width, 1))
        if aspect > 3.0:
            continue
        mime = str(info.get("mime", ""))
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        records.append(
            {
                "title": page.get("title", ""),
                "page_url": page.get("fullurl", ""),
                "download_url": info.get("thumburl") or info.get("url"),
                "original_url": info.get("url"),
                "width": width,
                "height": height,
                "mime": mime,
                "artist": metadata.get("Artist", {}).get("value", ""),
                "credit": metadata.get("Credit", {}).get("value", ""),
                "license": metadata.get("LicenseShortName", {}).get("value", ""),
                "license_url": metadata.get("LicenseUrl", {}).get("value", ""),
            }
        )
        if len(records) >= limit:
            break
    return records


def download_file(urls: list[str], path: Path, retries: int = 3) -> bool:
    urls = [url for url in urls if url]
    if not urls:
        return False
    last_error = None
    for url in urls:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(1, retries + 1):
            try:
                with urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
                return True
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * attempt)
        print(f"failed url: {url} ({last_error})")
    print(f"skip download after trying {len(urls)} url(s): {path.name}")
    return False


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def make_eval_config(output_dir: Path, images: dict[str, list[Path]], model: str) -> dict:
    pairs = []
    query_names = [name for name, paths in images.items() if len(paths) >= 2]

    for name in query_names:
        paths = images[name]
        pairs.append(
            {
                "name": f"{name}__same_class",
                "image_a": str(paths[0]),
                "image_b": str(paths[1]),
            }
        )

    for first, second in [("cat_animal", "dog_animal"), ("mountain_landscape", "beach_landscape"), ("apple_fruit", "flower_plant")]:
        if first in images and second in images and images[first] and images[second]:
            pairs.append(
                {
                    "name": f"{first}__{second}",
                    "image_a": str(images[first][0]),
                    "image_b": str(images[second][0]),
                }
            )

    return {
        "run_name": "commons_clip_base",
        "model": model,
        "gallery_dir": str(output_dir),
        "gallery_index": "outputs/commons_gallery_clip_base.npz",
        "output_root": "outputs/eval_runs",
        "top_k": 10,
        "normalize_output": True,
        "pairs": pairs,
        "operations": [
            {"name": "average"},
            {"name": "slerp", "params": {"t": [0.25, 0.5, 0.75]}},
            {"name": "weighted-sum", "params": {"alpha": [1.0], "beta": [-0.5, 0.5, 1.0]}},
            {"name": "project-a-on-b"},
            {"name": "reject-a-from-b"},
            {"name": "reflect-a-across-b"},
            {"name": "cosine-bisector"},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small licensed image gallery from Wikimedia Commons.")
    parser.add_argument("--output-dir", default="data/commons_gallery", help="Image output directory.")
    parser.add_argument("--config-out", default="configs/commons_eval.generated.yml", help="Generated eval YAML path.")
    parser.add_argument("--per-query", type=int, default=6, help="Images to keep for each query.")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="Model name for generated eval config.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between downloads.")
    parser.add_argument("--user-agent", help="Override Wikimedia request User-Agent. Include contact info for larger runs.")
    parser.add_argument("--query", action="append", help="Override default queries; can be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global USER_AGENT
    if args.user_agent:
        USER_AGENT = args.user_agent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = args.query or DEFAULT_QUERIES
    manifest = []
    images_by_query: dict[str, list[Path]] = {}

    for query in queries:
        query_key = sanitize(query)
        query_dir = output_dir / query_key
        query_dir.mkdir(parents=True, exist_ok=True)
        images_by_query[query_key] = []
        records = page_records(query, args.per_query)
        print(f"{query}: {len(records)} candidates")

        for index, record in enumerate(records, start=1):
            title_name = sanitize(record["title"])
            suffix = ".jpg" if "jpeg" in record["mime"] else ".png"
            image_path = query_dir / f"{index:02d}_{title_name}{suffix}"
            if not image_path.exists():
                if not download_file([record["download_url"], record["original_url"]], image_path):
                    continue
                time.sleep(args.sleep)
            if not validate_image(image_path):
                image_path.unlink(missing_ok=True)
                continue
            images_by_query[query_key].append(image_path)
            manifest.append({**record, "query": query, "local_path": str(image_path)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    config = make_eval_config(output_dir, images_by_query, args.model)
    config_out = Path(args.config_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"downloaded {len(manifest)} images to {output_dir}")
    print(f"wrote manifest to {manifest_path}")
    print(f"wrote eval config to {config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
