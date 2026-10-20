#!/usr/bin/env python3
"""Rebuild the README artwork and charts from the checked-in result summary.

No fundus images, annotations, checkpoints, or machine-specific paths are used.
Run from any directory: python tools/build_figures.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
C = {"ink": "#112B46", "muted": "#53677C", "teal": "#00756D",
     "teal_light": "#E1F4F0", "coral": "#B85236", "coral_light": "#FCEDE6",
     "navy_light": "#EAF0F7", "edge": "#D8E3ED", "white": "#FFFFFF", "canvas": "#F6F9FC"}
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "text.color": C["ink"],
                     "axes.unicode_minus": False, "figure.facecolor": C["white"]})


def base(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, face, edge=None, r=18, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % r,
                       linewidth=lw, facecolor=face, edgecolor=edge or face)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=18, weight="normal", color=None, **kw):
    return ax.text(x, y, s, fontsize=size, weight=weight, color=color or C["ink"], va="top", **kw)


def line(ax, x1, y1, x2, y2, color=None, lw=2, **kw):
    ax.plot([x1, x2], [y1, y2], color=color or C["edge"], lw=lw, **kw)


def arrow(ax, a, b, color=None, lw=2.4, style="-|>", **kw):
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, linewidth=lw,
                        color=color or C["muted"], **kw)
    ax.add_patch(p)
    return p


def save(fig, directory, name):
    for ext in ("svg", "png"):
        fig.savefig(directory / (name + "." + ext), dpi=200 if ext == "png" else 100,
                    facecolor=C["white"], metadata={"Creator": "retinal-blip2-vqa / tools/build_figures.py"})
    plt.close(fig)


def banner(directory):
    fig, ax = base(1120, 330)
    box(ax, 1, 1, 1118, 328, C["canvas"], C["edge"], r=22)
    box(ax, 33, 36, 5, 258, C["teal"], r=2)
    txt(ax, 60, 38, "RETINAL VISION  /  LANGUAGE", 15, "bold", C["teal"])
    txt(ax, 58, 86, "Retinal BLIP-2", 42, "bold")
    txt(ax, 58, 143, "Visual Question Answering", 27, "normal")
    txt(ax, 60, 207, "Adapt the bridge. Evaluate the evidence.", 17, color=C["muted"])
    box(ax, 60, 253, 174, 32, C["navy_light"], r=16)
    txt(ax, 147, 261, "MODEL ADAPTATION", 10.5, "bold", ha="center")
    box(ax, 245, 253, 182, 32, C["teal_light"], r=16)
    txt(ax, 336, 261, "LESION RECOGNITION", 10.5, "bold", C["teal"], ha="center")
    # Abstract token geometry: this artwork is not a retinal photograph or evidence map.
    bx, by = 680, 54
    box(ax, bx, by + 28, 111, 160, C["white"], C["edge"], r=16)
    for row in range(3):
        for col in range(3):
            box(ax, bx + 19 + col * 25, by + 64 + row * 25, 16, 16,
                C["navy_light"] if (row + col) % 2 else "#B8CADC", r=4)
    arrow(ax, (798, by + 110), (836, by + 110), C["teal"])
    box(ax, 841, by + 5, 95, 207, C["teal_light"], C["teal"], r=17, lw=1.3)
    for iy, y in enumerate((by + 47, by + 88, by + 129, by + 170)):
        for ix, x in enumerate((866, 907)):
            ax.add_patch(Circle((x, y), 7.5, facecolor=C["teal"] if (ix+iy)%2 else "#71BDB3", edgecolor="none"))
    for y in (by + 47, by + 88, by + 129):
        line(ax, 866, y+8, 907, y+33, C["teal"], 1, alpha=.5)
        line(ax, 907, y+8, 866, y+33, C["teal"], 1, alpha=.5)
    arrow(ax, (944, by + 110), (980, by + 110), C["teal"])
    box(ax, 985, by + 28, 97, 160, C["white"], C["edge"], r=16)
    for iy, w in enumerate((52, 42, 57, 35, 48)):
        box(ax, 1005, by + 61 + iy * 19, w, 6, "#B8CADC", r=3)
    txt(ax, 735, 270, "VISION", 11, "bold", C["muted"], ha="center")
    txt(ax, 889, 285, "BRIDGE", 11, "bold", C["teal"], ha="center")
    txt(ax, 1033, 270, "LANGUAGE", 11, "bold", C["muted"], ha="center")
    save(fig, directory, "banner")


def method(directory):
    fig, ax = base(1120, 535)
    box(ax, 1, 1, 1118, 533, C["white"], C["edge"], r=22)
    txt(ax, 35, 30, "Frozen backbones. Trainable visual bridge.", 27, "bold")
    txt(ax, 35, 76, "Same pretrained initialization, inputs and answer-scoring rule", 17, color=C["muted"])
    # Main image pathway.
    cards = [(35, 205, "Image encoder", "ViT", "Frozen", C["navy_light"], C["muted"]),
             (288, 257, "Visual bridge", "Q-Former + query tokens", "M1: train", C["teal_light"], C["teal"]),
             (593, 205, "Projection", "To language space", "B1 + M1: train", C["coral_light"], C["coral"]),
             (846, 237, "Language model", "OPT", "Frozen", C["navy_light"], C["muted"])]
    y = 179
    for x, w, title, subtitle, badge, fill, accent in cards:
        box(ax, x, y, w, 154, fill, r=17)
        txt(ax, x+w/2, y+22, title, 16, "bold", ha="center")
        txt(ax, x+w/2, y+61, subtitle, 11.5, ha="center", color=C["muted"])
        box(ax, x+15, y+104, w-30, 30, C["white"], r=14)
        txt(ax, x+w/2, y+112, badge, 11.5, "bold", accent, ha="center")
    arrow(ax, (249, 255), (278, 255))
    arrow(ax, (554, 255), (583, 255))
    arrow(ax, (807, 255), (836, 255))
    txt(ax, 125, 133, "Fundus image", 16, "bold", ha="center")
    arrow(ax, (125, 158), (125, 176), lw=1.5)
    txt(ax, 965, 133, "Question text", 16, "bold", ha="center")
    arrow(ax, (965, 158), (965, 176), lw=1.5)
    txt(ax, 965, 358, "Answer", 17, "bold", ha="center")
    arrow(ax, (965, 335), (965, 353), lw=1.5)
    # OPT is frozen without cutting gradients to the visual prefix.
    arrow(ax, (963, 409), (427, 409), C["teal"], lw=1.8, linestyle=(0, (5, 4)))
    txt(ax, 560, 432, "Answer loss backpropagates through frozen OPT to the bridge", 15,
        color=C["teal"], ha="center")
    line(ax, 35, 475, 1084, 475, lw=1)
    txt(ax, 35, 493, "B0  No adaptation", 14, "bold")
    txt(ax, 345, 493, "B1  Projection only", 14, "bold", C["coral"])
    txt(ax, 723, 493, "M1  Q-Former + projection", 14, "bold", C["teal"])
    save(fig, directory, "method")


def method_mobile(directory):
    fig, ax = base(640, 875)
    box(ax, 1, 1, 638, 873, C["white"], C["edge"], r=20)
    txt(ax, 27, 25, "Frozen backbones.\nTrainable visual bridge.", 23, "bold")
    txt(ax, 30, 116, "Fundus image", 14, "bold", color=C["muted"])
    cards = [(153, "Image encoder", "ViT", "Frozen", C["navy_light"], C["muted"]),
             (289, "Visual bridge", "Q-Former + query tokens", "M1: train", C["teal_light"], C["teal"]),
             (425, "Projection", "To language space", "B1 + M1: train", C["coral_light"], C["coral"]),
             (561, "Language model", "OPT", "Frozen", C["navy_light"], C["muted"])]
    for y, title, subtitle, badge, fill, accent in cards:
        box(ax, 30, y, 580, 104, fill, r=15)
        txt(ax, 50, y+17, title, 20, "bold")
        txt(ax, 50, y+59, subtitle, 14, color=C["muted"])
        box(ax, 405, y+34, 183, 35, C["white"], r=15)
        txt(ax, 496, y+43, badge, 11, "bold", accent, ha="center")
    for y in (257, 393, 529):
        arrow(ax, (320, y+3), (320, y+27), lw=1.8)
    txt(ax, 30, 685, "Question text → OPT → Answer", 16, "bold")
    line(ax, 30, 736, 610, 736, lw=1)
    txt(ax, 30, 758, "Answer-loss gradients pass through frozen OPT", 12.5, color=C["teal"])
    txt(ax, 30, 787, "to the trainable visual prefix.", 12.5, color=C["teal"])
    txt(ax, 30, 836, "B0: no adaptation   ·   B1: projection   ·   M1: both", 10.5, "bold")
    save(fig, directory, "method-mobile")


def find_method(block, identifier):
    items = block["methods"]
    if isinstance(items, dict):
        return items[identifier]
    return next(v for v in items if v["id"] == identifier)


def benchmark(directory, result):
    fig, ax = base(1120, 736)
    box(ax, 1, 1, 1118, 734, C["white"], C["edge"], r=22)
    txt(ax, 35, 26, "Two studies. Two distinct endpoints.", 27, "bold")
    txt(ax, 35, 72, "B1: projection only     M1: Q-Former + projection     /     Three fixed training seeds", 15, color=C["muted"])
    panels = [(126, "vqa", "01  ORIGINAL RETINAL VQA", "Type-balanced answer accuracy", "96 original-image groups · 192 questions", "vqa"),
              (419, "lesion", "02  INDEPENDENT LESION STUDY", "Lesion recognition: binary macro-F1", "225 images · 4 annotated lesion types", "lesion")]
    for y, key, title, metric, cohort, _ in panels:
        block = result[key]
        b1, m1 = find_method(block, "B1"), find_method(block, "M1")
        a, b = float(b1["mean"]), float(m1["mean"])
        comp = block["primary_comparison"]
        diff = float(comp["difference"])
        lo, hi = comp["ci95"]
        box(ax, 24, y, 1072, 269, C["canvas"], r=16)
        txt(ax, 48, y+20, title, 13, "bold", C["teal"])
        txt(ax, 48, y+52, metric, 22, "bold")
        txt(ax, 48, y+90, cohort, 14, color=C["muted"])
        x0, barwidth = 126, 610
        bar_y = y+143
        for j, (lab, val, col) in enumerate((("B1", a, C["muted"]), ("M1", b, C["teal"]))):
            yy = bar_y + j * 45
            txt(ax, 49, yy-1, lab, 18, "bold", col)
            box(ax, x0, yy, barwidth, 25, C["navy_light"], r=6)
            box(ax, x0, yy, barwidth * val, 25, col, r=6)
            txt(ax, x0 + barwidth + 17, yy-3, "%.2f%%" % (val * 100), 21, "bold", col)
        box(ax, 891, y+137, 177, 96, C["teal_light"], r=12)
        txt(ax, 979, y+148, "+%.2f pp" % (diff*100), 22, "bold", C["teal"], ha="center")
        txt(ax, 979, y+185, "95% CI", 11, "bold", C["muted"], ha="center")
        txt(ax, 979, y+206, "[+%.2f, +%.2f]" % (lo*100, hi*100), 12, color=C["teal"], ha="center")
        txt(ax, 127, y+236, "0", 10, color=C["muted"])
        txt(ax, 126+barwidth, y+236, "100%", 10, color=C["muted"], ha="right")
    txt(ax, 35, 710, "Paired original-image-group bootstrap · 2,000 resamples · Separate cohorts; scores are not interchangeable", 11.3, color=C["muted"])
    save(fig, directory, "results")


def benchmark_mobile(directory, result):
    fig, ax = base(640, 875)
    box(ax, 1, 1, 638, 873, C["white"], C["edge"], r=20)
    txt(ax, 26, 25, "Two studies.\nTwo distinct endpoints.", 23, "bold")
    txt(ax, 28, 122, "B1: projection only · M1: Q-Former + projection", 11.5, color=C["muted"])
    txt(ax, 28, 149, "Same three fixed training seeds", 11.5, color=C["muted"])
    panels = [(190, "vqa", "01  ORIGINAL RETINAL VQA", "Type-balanced accuracy", "96 image groups · 192 questions"),
              (496, "lesion", "02  INDEPENDENT LESION STUDY", "Binary macro-F1", "225 images · 4 lesion types")]
    for y, key, title, metric, cohort in panels:
        block = result[key]
        a = float(find_method(block, "B1")["mean"])
        b = float(find_method(block, "M1")["mean"])
        comp = block["primary_comparison"]
        lo, hi = comp["ci95"]
        box(ax, 20, y, 600, 286, C["canvas"], r=15)
        txt(ax, 38, y+17, title, 12, "bold", C["teal"])
        txt(ax, 38, y+52, metric, 21, "bold")
        txt(ax, 38, y+94, cohort, 13, color=C["muted"])
        for j, (lab, val, col) in enumerate((("B1", a, C["muted"]), ("M1", b, C["teal"]))):
            yy = y + 140 + j*44
            txt(ax, 38, yy-1, lab, 16, "bold", col)
            box(ax, 99, yy, 350, 23, C["navy_light"], r=5)
            box(ax, 99, yy, 350*val, 23, col, r=5)
            txt(ax, 467, yy-3, "%.2f%%" % (val*100), 19, "bold", col)
        box(ax, 38, y+238, 562, 32, C["teal_light"], r=8)
        txt(ax, 51, y+246, "+%.2f pp" % (comp["difference"]*100), 12.5, "bold", C["teal"])
        txt(ax, 224, y+246, "95%% CI [+%.2f, +%.2f]" % (lo*100, hi*100), 11.5, color=C["teal"])
    txt(ax, 28, 807, "Paired image-group bootstrap · 2,000 resamples", 11, color=C["muted"])
    txt(ax, 28, 836, "Separate cohorts and metrics", 11, color=C["muted"])
    save(fig, directory, "results-mobile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "main_results.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--artwork-only", action="store_true", help="Build non-data artwork before the checked-in summary exists")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banner(args.output_dir)
    method(args.output_dir)
    method_mobile(args.output_dir)
    source = None
    if not args.artwork_only:
        raw = args.results.read_bytes()
        result = json.loads(raw)
        benchmark(args.output_dir, result)
        benchmark_mobile(args.output_dir, result)
        source = {"file": "results/main_results.json", "sha256": hashlib.sha256(raw).hexdigest()}
    files = {}
    for p in sorted(args.output_dir.glob("*")):
        if p.suffix in (".svg", ".png"):
            files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    manifest = {"schema_version": 1, "scientific_chart_source": source,
                "artwork": "Abstract tokens and architecture; no medical images or inferred lesion maps.",
                "script": "tools/build_figures.py", "files": files}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built %d SVG/PNG files." % len(files))


if __name__ == "__main__":
    main()
