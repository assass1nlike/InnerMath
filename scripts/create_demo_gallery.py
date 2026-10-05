from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    root = Path("data/demo_gallery")
    root.mkdir(parents=True, exist_ok=True)

    items = [
        ("red_square.png", "square", (220, 40, 40)),
        ("blue_square.png", "square", (40, 80, 220)),
        ("purple_square.png", "square", (145, 55, 190)),
        ("green_square.png", "square", (40, 170, 80)),
        ("red_circle.png", "circle", (220, 40, 40)),
        ("blue_circle.png", "circle", (40, 80, 220)),
        ("purple_circle.png", "circle", (145, 55, 190)),
        ("green_circle.png", "circle", (40, 170, 80)),
    ]

    for name, shape, color in items:
        image = Image.new("RGB", (256, 256), (245, 245, 240))
        draw = ImageDraw.Draw(image)
        if shape == "square":
            draw.rectangle((64, 64, 192, 192), fill=color)
        else:
            draw.ellipse((58, 58, 198, 198), fill=color)
        image.save(root / name)

    print(f"created {len(items)} images in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
