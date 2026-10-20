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


def method_tokens(ax, x, y, n=7, w=25, h=27, gap=7, fill=None, edge=None):
    """Schematic token strip; the adjacent tensor label gives the actual count."""
    for i in range(n):
        box(ax, x+i*(w+gap), y, w, h, fill or C["teal_light"], edge or C["teal"], r=4, lw=1.1)
    return x+n*w+(n-1)*gap


def method_path(ax, points, color=None, lw=2.2, dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        line(ax,*a,*b,color=color or C["muted"],lw=lw,linestyle=(0,(5,4)) if dashed else "-")
    arrow(ax,points[-2],points[-1],color or C["muted"],lw=lw,linestyle=(0,(5,4)) if dashed else "-")


def method_patch_icon(ax, x, y, size):
    box(ax,x,y,size,size,C["white"],C["edge"],r=9)
    cell=(size-24)/4
    for row in range(4):
        for col in range(4):
            color=C["navy_light"] if (row+col)%3 else "#B8CADC"
            box(ax,x+12+col*cell+1.5,y+12+row*cell+1.5,cell-4,cell-4,color,r=2)


def method(directory):
    fig,ax=base(1120,1750)
    box(ax,1,1,1118,1748,C["white"],C["edge"],r=22)
    txt(ax,30,26,"BLIP-2 retinal adaptation",31,"bold")
    txt(ax,30,84,"Decoder-only OPT path · Query-transformer detail · Answer-only supervision",17,color=C["muted"])
    # A. Frozen image features; this is an original abstract patch icon.
    box(ax,25,140,300,655,C["canvas"],C["edge"],r=15)
    txt(ax,45,161,"A  Visual features",19,"bold")
    txt(ax,175,217,"RGB · 224 × 224",18,ha="center")
    method_patch_icon(ax,105,259,140)
    arrow(ax,(175,407),(175,439),lw=2)
    box(ax,55,445,240,105,C["navy_light"],r=14)
    txt(ax,175,463,"ViT-g/14",22,"bold",ha="center")
    txt(ax,175,506,"Frozen · 39 blocks",16,ha="center",color=C["muted"])
    arrow(ax,(175,553),(175,599),lw=2)
    method_tokens(ax,66,610,n=6,w=30,h=30,gap=6,fill=C["navy_light"],edge=C["muted"])
    txt(ax,175,664,"257 × 1408",21,"bold",ha="center")
    txt(ax,175,707,"256 patches + CLS",16,ha="center",color=C["muted"])
    txt(ax,175,748,"Fixed visual tokens",16,ha="center",color=C["muted"])
    # B. Query-only Q-Former path used by this implementation.
    box(ax,345,140,750,655,C["white"],C["edge"],r=15)
    txt(ax,365,161,"B  Q-Former · trainable in M1",20,"bold")
    txt(ax,735,215,"Learned queries · 32 × 768",20,"bold",C["teal"],ha="center")
    method_tokens(ax,575,266,n=8,w=32,h=29,gap=7)
    box(ax,467,341,560,360,C["teal_light"],r=12)
    box(ax,461,335,560,360,"#F0FAF7",C["teal"],r=12,lw=1)
    box(ax,455,329,560,360,C["white"],C["teal"],r=12,lw=1.4)
    txt(ax,990,345,"12 blocks",17,"bold",C["teal"],ha="right")
    box(ax,490,395,490,64,C["teal_light"],r=9)
    txt(ax,735,412,"Query self-attention",20,"bold",ha="center")
    arrow(ax,(735,298),(735,389),C["teal"],lw=2)
    arrow(ax,(735,460),(735,489),C["teal"],lw=2)
    txt(ax,756,464,"Q",16,"bold",C["teal"])
    box(ax,490,495,490,91,C["teal_light"],r=9)
    txt(ax,735,507,"Cross-attention",20,"bold",ha="center")
    txt(ax,735,548,"Every 2nd block · 6 of 12",16,ha="center",color=C["muted"])
    method_path(ax,[(294,625),(384,625),(384,540),(485,540)],C["muted"],lw=2.1)
    txt(ax,400,505,"K,V",18,"bold",C["muted"])
    arrow(ax,(735,588),(735,614),C["teal"],lw=2)
    box(ax,490,620,490,55,C["teal_light"],r=9)
    txt(ax,735,632,"Feed-forward",20,"bold",ha="center")
    arrow(ax,(735,678),(735,713),C["teal"],lw=2)
    method_tokens(ax,575,719,n=8,w=32,h=27,gap=7)
    txt(ax,735,759,"Z · 32 query states × 768",18,"bold",C["teal"],ha="center")
    # C. Projected queries form a single visual prefix for a causal OPT decoder.
    box(ax,25,825,1070,615,C["canvas"],C["edge"],r=15)
    txt(ax,45,846,"C  Prefix-conditioned language generation",21,"bold")
    txt(ax,183,914,"Query states Z",18,"bold",C["teal"],ha="center")
    method_tokens(ax,70,951,n=6,w=31,h=25,gap=7)
    arrow(ax,(182,979),(182,998),C["teal"],lw=2)
    box(ax,55,1005,255,101,C["coral_light"],C["coral"],r=12,lw=1)
    txt(ax,183,1023,"FC projection",21,"bold",C["coral"],ha="center")
    txt(ax,183,1065,"768 → 2560",18,ha="center")
    arrow(ax,(182,1109),(182,1159),C["coral"],lw=2)
    method_tokens(ax,60,1170,n=6,w=35,h=29,gap=7)
    txt(ax,183,1224,"V · 32 × 2560",19,"bold",C["teal"],ha="center")
    method_path(ax,[(310,1185),(381,1185),(381,1267),(424,1267)],C["teal"],lw=2.2)
    txt(ax,450,914,"Answer predictions",19,"bold")
    method_tokens(ax,480,952,n=6,w=35,h=28,gap=7,fill=C["coral_light"],edge=C["coral"])
    arrow(ax,(742,967),(798,967),C["coral"],lw=2)
    box(ax,805,923,260,90,C["coral_light"],C["coral"],r=11,lw=1)
    txt(ax,935,939,"Answer-only NLL",17.5,"bold",C["coral"],ha="center")
    txt(ax,935,979,"+ EOS · mean CE",15.5,ha="center")
    box(ax,430,1045,635,112,C["navy_light"],r=13)
    txt(ax,748,1065,"OPT-2.7B · frozen",23,"bold",ha="center")
    txt(ax,748,1112,"Causal decoder · 32 blocks · width 2560",16.5,ha="center",color=C["muted"])
    arrow(ax,(610,1041),(610,987),lw=2)
    txt(ax,430,1191,"Concatenate embeddings once",17,"bold")
    chunks=[(430,212,"Visual prefix","32 tokens",C["teal_light"],C["teal"]),
            (642,211,"Question","T tokens",C["navy_light"],C["muted"]),
            (853,212,"Answer + EOS","A tokens · train",C["coral_light"],C["coral"])]
    for x,w,title,sub,fill,accent in chunks:
        box(ax,x,1230,w,75,fill,accent,r=4,lw=1)
        txt(ax,x+w/2,1240,title,16.5,"bold",accent,ha="center")
        txt(ax,x+w/2,1276,sub,14.5,ha="center")
    arrow(ax,(1045,1225),(1045,1163),lw=2)
    txt(ax,866,1327,"Teacher-forced",15.5,"bold",C["coral"])
    txt(ax,866,1358,"during training",15.5,color=C["coral"])
    txt(ax,55,1365,"Question text",18,"bold")
    arrow(ax,(265,1387),(424,1387),lw=2)
    box(ax,430,1355,340,70,C["navy_light"],r=10)
    txt(ax,600,1366,"Frozen OPT embedding",17,"bold",ha="center")
    txt(ax,600,1400,"T × 2560",15.5,ha="center")
    arrow(ax,(748,1350),(748,1312),lw=2)
    # Separate reverse-mode lane makes the freeze/gradient distinction explicit.
    box(ax,25,1470,1070,126,C["white"],C["edge"],r=14)
    txt(ax,45,1488,"Backward pass (M1)",18,"bold",C["teal"])
    txt(ax,605,1488,"Frozen OPT retains input gradients",16,color=C["muted"])
    labels=[(55,160,"ViT frozen",C["navy_light"],C["muted"]),
            (270,250,"Q-Former + queries",C["teal_light"],C["teal"]),
            (558,120,"FC",C["coral_light"],C["coral"]),
            (737,180,"OPT input",C["navy_light"],C["muted"]),
            (966,109,"Loss",C["coral_light"],C["coral"])]
    for x,w,label,fill,accent in labels:
        box(ax,x,1535,w,43,fill,r=8)
        txt(ax,x+w/2,1546,label,15.5,"bold",accent,ha="center")
    for a,b in [((960,1557),(923,1557)),((731,1557),(684,1557)),((552,1557),(526,1557))]:
        arrow(ax,a,b,C["teal"],lw=1.8,linestyle=(0,(4,3)))
    arrow(ax,(264,1557),(247,1557),C["teal"],lw=1.8,linestyle=(0,(4,3)))
    line(ax,240,1540,240,1574,C["muted"],lw=3)
    txt(ax,240,1580,"stop",13,color=C["muted"],ha="center")
    # Actual downstream training boundaries; pretraining losses are not shown.
    box(ax,25,1620,1070,69,C["canvas"],r=12)
    txt(ax,45,1641,"B0  No adaptation",17,"bold")
    txt(ax,365,1641,"B1  FC only",17,"bold",C["coral"])
    txt(ax,635,1641,"M1  Q-Former + queries + FC",17,"bold",C["teal"])
    txt(ax,30,1710,"Original redraw from BLIP-2 Fig. 2 / Fig. 3 (OPT path); dimensions match this checkpoint.",13,color=C["muted"])
    save(fig,directory,"method")


def method_mobile(directory):
    fig,ax=base(640,2880)
    box(ax,1,1,638,2878,C["white"],C["edge"],r=20)
    txt(ax,27,25,"BLIP-2 retinal\nadaptation",28,"bold")
    txt(ax,30,134,"OPT decoder · Query-transformer detail",16.5,color=C["muted"])
    # Frozen image pathway.
    box(ax,25,195,590,430,C["canvas"],C["edge"],r=14)
    txt(ax,45,217,"1  Frozen visual features",22,"bold")
    txt(ax,125,277,"RGB · 224²",18,ha="center")
    method_patch_icon(ax,70,321,110)
    arrow(ax,(186,376),(231,376),lw=2)
    box(ax,240,319,335,113,C["navy_light"],r=12)
    txt(ax,407,339,"ViT-g/14",25,"bold",ha="center")
    txt(ax,407,391,"Frozen · 39 blocks",18,ha="center",color=C["muted"])
    arrow(ax,(406,437),(406,481),lw=2)
    method_tokens(ax,112,493,n=8,w=42,h=32,gap=9,fill=C["navy_light"],edge=C["muted"])
    txt(ax,320,557,"257 × 1408",23,"bold",ha="center")
    txt(ax,320,598,"256 patches + CLS",18,ha="center",color=C["muted"])
    # Magnified Q-Former query stream.
    box(ax,25,665,590,710,C["white"],C["edge"],r=14)
    txt(ax,55,687,"2  Q-Former · trainable in M1",20,"bold")
    txt(ax,337,751,"Learned queries · 32 × 768",21,"bold",C["teal"],ha="center")
    method_tokens(ax,142,800,n=8,w=35,h=31,gap=9)
    txt(ax,559,860,"12 blocks",18,"bold",C["teal"],ha="right")
    box(ax,79,897,505,75,C["teal_light"],r=10)
    txt(ax,332,918,"Query self-attention",23,"bold",ha="center")
    arrow(ax,(332,836),(332,891),C["teal"],lw=2)
    arrow(ax,(332,974),(332,1014),C["teal"],lw=2)
    box(ax,79,1020,505,107,C["teal_light"],r=10)
    txt(ax,332,1036,"Cross-attention",23,"bold",ha="center")
    txt(ax,332,1084,"Q: queries · K,V: visual tokens",17.5,ha="center",color=C["muted"])
    method_path(ax,[(108,509),(45,509),(45,1073),(73,1073)],C["muted"],lw=2)
    txt(ax,60,990,"K,V",16,"bold",C["muted"])
    txt(ax,332,1148,"Every 2nd block · 6 of 12",18,ha="center",color=C["muted"])
    arrow(ax,(332,1186),(332,1209),C["teal"],lw=2)
    box(ax,79,1215,505,69,C["teal_light"],r=10)
    txt(ax,332,1231,"Feed-forward",23,"bold",ha="center")
    arrow(ax,(332,1289),(332,1309),C["teal"],lw=2)
    txt(ax,332,1331,"Z · 32 query states × 768",20,"bold",C["teal"],ha="center")
    # Prefix-only conditioning of OPT; the question bypasses Q-Former.
    box(ax,25,1415,590,1005,C["canvas"],C["edge"],r=14)
    txt(ax,45,1437,"3  Prefix-conditioned OPT",22,"bold")
    box(ax,65,1506,510,95,C["coral_light"],C["coral"],r=12,lw=1)
    txt(ax,320,1521,"FC projection",24,"bold",C["coral"],ha="center")
    txt(ax,320,1567,"768 → 2560",20,ha="center")
    arrow(ax,(320,1607),(320,1638),C["coral"],lw=2)
    method_tokens(ax,130,1644,n=8,w=35,h=31,gap=9)
    txt(ax,320,1694,"32 visual prefix tokens · width 2560",18.5,"bold",C["teal"],ha="center")
    box(ax,65,1770,227,76,C["white"],C["edge"],r=9)
    txt(ax,178,1789,"Question text",19,"bold",ha="center")
    arrow(ax,(298,1808),(331,1808),lw=2)
    box(ax,338,1766,237,84,C["navy_light"],r=10)
    txt(ax,456,1779,"Frozen OPT",18,"bold",ha="center")
    txt(ax,456,1815,"embedding",18,ha="center")
    txt(ax,65,1870,"Concatenate embeddings once",19,"bold")
    for x,w,title,sub,fill,accent in [(65,170,"Visual\nprefix","32 tokens",C["teal_light"],C["teal"]),(235,170,"Question","T tokens",C["navy_light"],C["muted"]),(405,170,"Answer*","A + EOS",C["coral_light"],C["coral"])]:
        box(ax,x,1950,w,119,fill,accent,r=3,lw=1)
        txt(ax,x+w/2,1965,title,20.5,"bold",accent,ha="center")
        txt(ax,x+w/2,2037,sub,17,ha="center")
    method_path(ax,[(124,1659),(44,1659),(44,2009),(59,2009)],C["teal"],lw=2)
    method_path(ax,[(456,1855),(605,1855),(605,1920),(320,1920),(320,1944)],C["muted"],lw=2)
    txt(ax,65,2095,"*Teacher-forced answer: training only",18,color=C["coral"])
    arrow(ax,(320,2134),(320,2160),lw=2)
    box(ax,65,2166,510,110,C["navy_light"],r=12)
    txt(ax,320,2184,"OPT-2.7B · frozen",25,"bold",ha="center")
    txt(ax,320,2238,"32-block causal decoder · width 2560",17,ha="center",color=C["muted"])
    arrow(ax,(320,2280),(320,2305),lw=2)
    box(ax,65,2311,510,86,C["coral_light"],C["coral"],r=11,lw=1)
    txt(ax,320,2325,"Answer-only NLL (+ EOS)",21,"bold",C["coral"],ha="center")
    txt(ax,320,2366,"Prefix, question and padding masked",16.5,ha="center")
    # Reverse-mode and training-boundary legend.
    box(ax,25,2460,590,175,C["white"],C["edge"],r=13)
    txt(ax,45,2481,"Backward pass (M1)",22,"bold",C["teal"])
    txt(ax,45,2531,"Loss → OPT input → FC",21,"bold",C["teal"])
    txt(ax,45,2571,"→ Q-Former + queries",21,"bold",C["teal"])
    txt(ax,45,2610,"Stops before frozen ViT",17,color=C["muted"])
    for y,title,fill,accent in [(2670,"B0  No adaptation",C["navy_light"],C["muted"]),(2721,"B1  FC only",C["coral_light"],C["coral"]),(2772,"M1  Q-Former + queries + FC",C["teal_light"],C["teal"])]:
        box(ax,25,y,590,43,fill,r=8)
        txt(ax,45,y+9,title,18,"bold",accent)
    txt(ax,30,2844,"Original redraw · BLIP-2 Fig. 2 / Fig. 3 (OPT)",14,color=C["muted"])
    save(fig,directory,"method-mobile")


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
                "script": "tools/build_figures.py", "files": files,
                "architecture_reference": {"paper": "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models",
                    "url": "https://proceedings.mlr.press/v202/li23q/li23q.pdf", "figures": ["Figure 2: query-transformer structure", "Figure 3 (top): decoder-only OPT path"],
                    "drawing": "Original editable-vector redraw for this retinal adaptation; no paper pixels or retinal images reproduced.",
                    "checkpoint": "Salesforce/blip2-opt-2.7b", "revision": "59a1ef6c1e5117b3f65523d1c6066825bcf315e3",
                    "scope": "Actual Q-Former/projection downstream adaptation and answer-only loss; paper pretraining objectives omitted."}}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built %d SVG/PNG files." % len(files))


if __name__ == "__main__":
    main()
