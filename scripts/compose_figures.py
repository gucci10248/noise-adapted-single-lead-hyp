"""Compose multi-panel manuscript figures from the analysis PNGs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def stack(paths: list[Path], labels: list[str], out: Path, width: int = 2400) -> None:
    ims = [load(p) for p in paths]
    resized = []
    for im in ims:
        h = round(im.height * width / im.width)
        resized.append(im.resize((width, h), Image.LANCZOS))
    gap = 70
    total_h = sum(im.height for im in resized) + gap * (len(resized) + 1)
    canvas = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = gap
    for im, label in zip(resized, labels):
        canvas.paste(im, (0, y))
        draw.text((24, y - 58), label, fill="black", font=font(52))
        y += im.height + gap
    canvas.save(out, dpi=(300, 300))
    canvas.save(out.with_suffix(".jpg"), quality=95, dpi=(300, 300))
    print(f"wrote {out} ({canvas.width}x{canvas.height})")


def side_by_side(paths: list[Path], labels: list[str], out: Path, height: int = 1500) -> None:
    ims = [load(p) for p in paths]
    resized = []
    for im in ims:
        w = round(im.width * height / im.height)
        resized.append(im.resize((w, height), Image.LANCZOS))
    gap = 70
    total_w = sum(im.width for im in resized) + gap * (len(resized) + 1)
    canvas = Image.new("RGB", (total_w, height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    x = gap
    for im, label in zip(resized, labels):
        canvas.paste(im, (x, gap))
        draw.text((x, 8), label, fill="black", font=font(52))
        x += im.width + gap
    canvas.save(out, dpi=(300, 300))
    canvas.save(out.with_suffix(".jpg"), quality=95, dpi=(300, 300))
    print(f"wrote {out} ({canvas.width}x{canvas.height})")



def grid2x2(paths: list[Path], labels: list[str], out: Path, cell: int = 1400) -> None:
    ims = [load(p) for p in paths]
    gap = 70
    canvas = Image.new("RGB", (2 * cell + 3 * gap, 2 * cell + 3 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (im, label) in enumerate(zip(ims, labels)):
        im = im.resize((cell, cell), Image.LANCZOS)
        r, c = divmod(idx, 2)
        x = gap + c * (cell + gap)
        y = gap + r * (cell + gap)
        canvas.paste(im, (x, y))
        draw.text((x + 10, y + 6), label, fill="black", font=font(52))
    canvas.save(out, dpi=(300, 300))
    canvas.save(out.with_suffix(".jpg"), quality=95, dpi=(300, 300))
    print(f"wrote {out} ({canvas.width}x{canvas.height})")


def main() -> None:
    grid2x2(
        [
            FIG / "hyp_v1_test_roc.png",
            FIG / "hyp_v1_aug_test_roc.png",
            FIG / "hyp_v1_test_calibration.png",
            FIG / "hyp_v1_aug_test_calibration.png",
        ],
        ["A", "B", "C", "D"],
        FIG / "figure2_internal_validation.png",
    )
    stack(
        [
            FIG / "hyp_v1_aug_chapman_roc_pr.png",
            FIG / "hyp_v1_aug_chapman_calibration.png",
        ],
        ["A", "B"],
        FIG / "figure3_chapman.png",
    )
    stack(
        [
            FIG / "georgia_ext_roc_pr.png",
            FIG / "georgia_ext_calibration.png",
        ],
        ["A", "B"],
        FIG / "figure4_georgia.png",
    )
    # Supplementary subgroup forest plots
    for src, out in [
        (FIG / "hyp_v1_aug_chapman_subgroups.png", FIG / "figureS1_chapman_subgroups.png"),
        (FIG / "georgia_ext_subgroups.png", FIG / "figureS2_georgia_subgroups.png"),
    ]:
        im = load(src)
        im.save(out, dpi=(300, 300))
        print(f"wrote {out}")
    # Figure 5 already has two panels; copy as a submission JPEG.
    im = load(FIG / "noise_robustness.png")
    im.save(FIG / "figure5_noise_robustness.jpg", quality=95, dpi=(300, 300))
    print(f"wrote {FIG / 'figure5_noise_robustness.jpg'}")


if __name__ == "__main__":
    main()
