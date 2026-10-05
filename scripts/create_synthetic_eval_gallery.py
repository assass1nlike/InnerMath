from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 80),
    "purple": (145, 55, 190),
    "yellow": (230, 190, 45),
}

SHAPES = ["square", "circle", "triangle", "bar"]
STYLES = ["solid", "outline", "striped"]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: tuple[int, int, int], style: str) -> None:
    bbox = (58, 58, 198, 198)
    fill = color if style == "solid" else None
    width = 8 if style == "outline" else 1

    if shape == "square":
        draw.rectangle((64, 64, 192, 192), fill=fill, outline=color, width=width)
    elif shape == "circle":
        draw.ellipse(bbox, fill=fill, outline=color, width=width)
    elif shape == "triangle":
        draw.polygon([(128, 48), (48, 202), (208, 202)], fill=fill, outline=color)
        if style == "outline":
            draw.line([(128, 48), (48, 202), (208, 202), (128, 48)], fill=color, width=width)
    elif shape == "bar":
        draw.rounded_rectangle((46, 96, 210, 160), radius=12, fill=fill, outline=color, width=width)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    if style == "striped":
        for offset in range(-140, 260, 18):
            draw.line((offset, 220, offset + 220, 0), fill=color, width=6)


def main() -> int:
    root = Path("data/eval_gallery")
    root.mkdir(parents=True, exist_ok=True)

    count = 0
    for color_name, color in COLORS.items():
        for shape in SHAPES:
            for style in STYLES:
                image = Image.new("RGB", (256, 256), (245, 245, 240))
                draw = ImageDraw.Draw(image)
                draw_shape(draw, shape, color, style)
                path = root / f"{color_name}_{shape}_{style}.png"
                image.save(path)
                count += 1

    print(f"created {count} images in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
