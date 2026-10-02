#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive FIB-SEM milling-angle schematic (cryo-FIB lamella preparation).

Milling angle (grazing angle between ion beam and grid surface):

    theta = (90 - beta) + T - P

    beta : angle between electron and ion columns (typically 52 deg)
    T    : stage tilt
    P    : shuttle pre-tilt (typically 45 deg)

Net grid incline from horizontal is i = P - T, and the SEM views the grid
i degrees off its normal.

Usage:
    python FIB-SEM_angle_widget.py                 # interactive window
    python FIB-SEM_angle_widget.py -T 18 -P 45     # start at given values
    python FIB-SEM_angle_widget.py --save out.png  # render a static figure

Requires: matplotlib, numpy

Note: all non-ASCII characters are written as \\u escapes so the file
survives any editor / transfer encoding.
"""
import argparse

import matplotlib

import numpy as np

DEG = "\u00b0"
BETA = "\u03b2"
THETA = "\u03b8"


def rot(p, deg):
    """Rotate 2D point(s) counter-clockwise (y up) by deg degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y = p[..., 0], p[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def pol(c, r, deg):
    a = np.radians(deg)
    return np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def rect(x0, y0, w, h):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)


def build_scene(T, P, beta):
    """Return all geometry in world coordinates (y up, grid centre at origin)."""
    G = np.zeros(2)
    hinge_local = np.array([-60.0, 0.0])      # shuttle hinge on stage top surface
    grid_in_shuttle = np.array([65.0, 12.0])  # grid centre in shuttle frame

    # Stage frame is rotated clockwise by T (right side goes down).
    # Shuttle frame is rotated counter-clockwise by P about the hinge.
    grid_local = hinge_local + rot(grid_in_shuttle, P)
    S = G - rot(grid_local, -T)

    def stage_to_world(pts):
        return S + rot(np.asarray(pts, float), -T)

    def shuttle_to_world(pts):
        pts = np.asarray(pts, float)
        return stage_to_world(hinge_local + rot(pts, P))

    return {
        "G": G,
        "hinge": stage_to_world(hinge_local),
        "plate": stage_to_world(rect(-130, -10, 260, 10)),
        "shuttle": shuttle_to_world(rect(0, 0, 130, 12)),
        "grid": shuttle_to_world(rect(50, 12, 30, 3)),
        "shuttle_lead": shuttle_to_world([25.0, 6.0]),
        "plate_left": stage_to_world([-130.0, -5.0]),
    }


def draw(ax, T, P, beta):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-330, 330)
    ax.set_ylim(-190, 260)
    ax.axis("off")

    from matplotlib.patches import Arc, Polygon

    sc = build_scene(T, P, beta)
    G, H = sc["G"], sc["hinge"]
    elev = 90.0 - beta
    inc = P - T
    mill = elev - inc

    gray, teal, amber = "#B4B2A9", "#5DCAA5", "#EF9F27"
    blue, coral, ref = "#378ADD", "#D85A30", "#888780"

    ax.add_patch(Polygon(sc["plate"], fc=gray, ec="#5F5E5A", lw=0.8, zorder=2))
    ax.add_patch(Polygon(sc["shuttle"], fc=teal, ec="#0F6E56", lw=0.8, zorder=3))
    ax.add_patch(Polygon(sc["grid"], fc=amber, ec="#854F0B", lw=0.8, zorder=4))

    # Reference lines: horizontal at grid, grid-plane extension, horizontal at hinge
    dash = dict(color=ref, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot([G[0], G[0] + 170], [G[1], G[1]], **dash)
    pe = pol(G, 170, inc)
    ax.plot([G[0], pe[0]], [G[1], pe[1]], **dash)
    ax.plot([H[0] - 85, H[0]], [H[1], H[1]], **dash)

    # Beams
    es, ee = np.array([0.0, 210.0]), np.array([0.0, 7.0])
    is_, ie = pol(G, 240, elev), pol(G, 7, elev)
    arrow = dict(arrowstyle="-|>", mutation_scale=14, lw=2, zorder=5)
    ax.annotate("", xy=ee, xytext=es, arrowprops=dict(color=blue, **arrow))
    ax.annotate("", xy=ie, xytext=is_, arrowprops=dict(color=coral, **arrow))
    ax.text(es[0] + 12, es[1], "Electron beam (SEM)", va="center", fontweight="bold", fontsize=10)
    ax.text(is_[0] + 10, is_[1], "Ion beam (FIB)", va="center", fontweight="bold", fontsize=10)

    # Angle arcs
    def arc(c, r, a1, a2, color="#444441", lw=1.2):
        lo, hi = sorted((a1, a2))
        if hi - lo > 0.3:
            ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=lo, theta2=hi, color=color, lw=lw, zorder=6))

    arc(G, 130, elev, 90)
    bl = pol(G, 148, (90 + elev) / 2)
    ax.text(*bl, f"{BETA} {beta:g}{DEG}", va="center", fontsize=10)

    arc(G, 50, 0, inc)
    il = pol(G, 66, -18 if inc >= 0 else 18)
    if abs(inc) > 0.5:
        ax.text(*il, f"i {inc:.1f}{DEG}", va="center", fontsize=10)

    arc(G, 90, inc, elev, color=coral, lw=2.5)
    ml = pol(G, 110, inc - (10 if mill >= 0 else -10))
    ax.text(*ml, f"{THETA} {mill:.1f}{DEG}", va="center", fontweight="bold", fontsize=11)

    arc(H, 45, 180 - T, 180)
    if T > 0.5:
        ax.text(*pol(H, 62, 192), f"T {T:g}{DEG}", ha="center", va="center", fontsize=10)
    arc(H, 72, -T, P - T)
    if P > 0.5:
        ax.text(*pol(H, 92, -T + P / 2), f"P {P:g}{DEG}", va="center", fontsize=10)

    # Part labels with leaders
    def label(text, xy, target):
        ax.text(xy[0], xy[1], text, va="center", fontsize=10, color="#444441")
        ax.plot([xy[0] + 118, target[0]], [xy[1], target[1]], color=ref, lw=0.6, ls=(0, (3, 2)), zorder=1)

    label("Grid / lamella plane", (-320, 110), G + np.array([-3, 3]))
    label("Shuttle (pre-tilt P)", (-320, 45), sc["shuttle_lead"])
    pl = sc["plate_left"]
    label("Stage (tilt T)", (-320, pl[1] - 40), pl)

    # Readout
    txt = (f"{THETA} = (90{DEG} - {BETA}) + T - P = (90{DEG} - {beta:g}{DEG}) + {T:g}{DEG} - {P:g}{DEG} = {mill:.1f}{DEG}\n"
           f"Net grid incline i = P - T = {inc:.1f}{DEG}   |   SEM view {abs(inc):.1f}{DEG} off grid normal")
    if mill <= 0.5:
        txt += "\nIon beam is parallel to or behind the grid surface: increase stage tilt."
    ax.text(0, -178, txt, ha="center", va="bottom", fontsize=9.5, color="#444441")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-T", "--tilt", type=float, default=18.0, help="stage tilt in degrees (default 18)")
    ap.add_argument("-P", "--pretilt", type=float, default=45.0, help="shuttle pre-tilt in degrees (default 45)")
    ap.add_argument("-b", "--beam-angle", type=float, default=52.0, help="angle between FIB and SEM columns (default 52; Incidence between SEM and FIB in Aquilos and Hydra = 52)")
    ap.add_argument("--save", metavar="FILE", help="save a static figure instead of opening a window")
    args = ap.parse_args()

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(9, 8))
    if not args.save:
        fig.canvas.manager.set_window_title("FIB-SEM milling angle")
    ax = fig.add_axes([0.02, 0.26, 0.96, 0.72])

    s_T = Slider(fig.add_axes([0.25, 0.17, 0.55, 0.03]), f"Stage tilt T ({DEG})", 0, 50, valinit=args.tilt, valstep=1)
    s_P = Slider(fig.add_axes([0.25, 0.12, 0.55, 0.03]), f"Shuttle pre-tilt P ({DEG})", 0, 60, valinit=args.pretilt, valstep=1)
    s_B = Slider(fig.add_axes([0.25, 0.07, 0.55, 0.03]), f"FIB-SEM angle {BETA} ({DEG})", 40, 65,
                 valinit=args.beam_angle, valstep=0.5)
    fig.text(0.5, 0.025, f"Note: Incidence between SEM and FIB in Aquilos and Hydra = 52{DEG}", ha="center", fontsize=9.5,
             color="#444441", style="italic")

    def refresh(_=None):
        draw(ax, s_T.val, s_P.val, s_B.val)
        fig.canvas.draw_idle()

    s_T.on_changed(refresh)
    s_P.on_changed(refresh)
    s_B.on_changed(refresh)
    refresh()

    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
