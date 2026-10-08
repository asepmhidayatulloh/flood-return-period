"""
Reference implementation (Python/NumPy) of the model used in index.html.

It rebuilds the same synthetic valley, solves the water surface with Manning's
equation (strip-wise conveyance), keeps only cells connected to the channel,
and writes a results table plus a preview figure. Use it to cross-check the
browser results or as a starting point for working with a real DEM.

Usage:
    python tools/reference_model.py                 # writes docs/preview.png and data/results.csv
    python tools/reference_model.py --mean 450 --cv 0.6 --no-figure
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np

NX, NY, DX = 260, 150, 10.0
T_LIST = [2, 5, 10, 25, 50, 100, 200]


def gumbel_k(T):
    return -(np.sqrt(6) / np.pi) * (0.5772157 + np.log(np.log(T / (T - 1.0))))


def _hash(i, j):
    i = i.astype(np.uint32)
    j = j.astype(np.uint32)
    h = (i * np.uint32(374761393) + j * np.uint32(668265263)) ^ np.uint32(0x5BD1E995)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    return ((h ^ (h >> np.uint32(16))).astype(np.float64)) / 4294967295.0


def _vnoise(x, y):
    i, j = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - i, y - j
    s = lambda t: t * t * (3 - 2 * t)
    a, b = _hash(i, j), _hash(i + 1, j)
    c, d = _hash(i, j + 1), _hash(i + 1, j + 1)
    return a + (b - a) * s(fx) + (c - a) * s(fy) + (a - b - c + d) * s(fx) * s(fy)


def build_terrain(S0):
    """Returns z[j, i], channel mask, centreline yc[i], thalweg elevation and row."""
    x = (np.arange(NX) + 0.5) * DX
    y = (np.arange(NY) + 0.5) * DX
    X, Y = np.meshgrid(x, y)
    yc = 750 + 120 * np.sin(2 * np.pi * x / 1300 + 0.6) + 35 * np.sin(2 * np.pi * x / 520)
    d = Y - yc[None, :]
    ad = np.abs(d)
    bw = 18.0
    z = 100 - S0 * X
    z = z - np.where(ad < bw, 3.2 * (1 - (ad / bw) ** 2), 0.0)
    z = z + 0.0042 * np.maximum(ad - bw, 0)
    ws = np.where(d < 0, 480.0, 320.0)
    z = z + np.where(ad > ws, 2.2 * np.power(np.maximum(ad - ws, 0) / 100, 1.6), 0.0)
    z = z + 1.1 * (_vnoise(X / 140, Y / 100) - 0.5)
    levee = (d > 0) & (X > 600) & (X < 1100)
    z = z + np.where(levee, 1.1 * np.exp(-((ad - 55) ** 2) / (2 * 22 ** 2)) * np.sin(np.pi * (X - 600) / 500), 0.0)
    z = z - 1.3 * np.exp(-(((X - 1700) ** 2) / (2 * 160 ** 2) + ((Y - (yc[None, :] - 230)) ** 2) / (2 * 70 ** 2)))
    is_ch = ad < bw + 6
    jth = z.argmin(axis=0)
    thal = z.min(axis=0)
    return z, is_ch, yc, thal, jth


def stage(Q, z, is_ch, thal, S0, nc, nf):
    n = np.where(is_ch, nc, nf)
    lo, hi = thal.copy(), thal + 30.0
    sq = np.sqrt(S0)
    for _ in range(42):
        m = 0.5 * (lo + hi)
        h = m[None, :] - z
        K = (np.where(h > 0, np.power(np.maximum(h, 0), 5 / 3) / n, 0.0) * DX).sum(axis=0)
        below = K * sq < Q
        lo = np.where(below, m, lo)
        hi = np.where(below, hi, m)
    w = 0.5 * (lo + hi)
    out = np.empty(NX)
    for i in range(NX):  # moving average, +-6 columns, truncated at the edges
        out[i] = w[max(0, i - 6): min(NX, i + 7)].mean()
    return out


def connected_mask(w, z, jth):
    wet = (w[None, :] - z) > 0.05
    mask = np.zeros_like(wet)
    q = deque()
    for i in range(NX):
        if wet[jth[i], i]:
            mask[jth[i], i] = True
            q.append((jth[i], i))
    while q:
        j, i = q.popleft()
        for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            jj, ii = j + dj, i + di
            if 0 <= jj < NY and 0 <= ii < NX and wet[jj, ii] and not mask[jj, ii]:
                mask[jj, ii] = True
                q.append((jj, ii))
    return mask


def run(mean=300.0, cv=0.55, S0=0.002, nc=0.035, nf=0.07):
    z, is_ch, yc, thal, jth = build_terrain(S0)
    results = []
    for T in T_LIST:
        Q = max(0.05 * mean, mean * (1 + gumbel_k(T) * cv))
        w = stage(Q, z, is_ch, thal, S0, nc, nf)
        mask = connected_mask(w, z, jth)
        depth = np.where(mask, np.maximum(w[None, :] - z, 0), 0.0)
        depth[depth <= 0.05] = 0.0
        wet = depth > 0
        area = wet.sum() * DX * DX
        results.append(dict(T=T, Q=Q, w=w, depth=depth, area_ha=area / 1e4,
                            mean_depth=depth[wet].mean() if wet.any() else 0.0,
                            max_depth=depth.max(), volume_m3=depth.sum() * DX * DX,
                            vel=np.where(wet, np.power(np.maximum(depth, 1e-9), 2 / 3) * np.sqrt(S0) / np.where(is_ch, nc, nf), 0.0)))
    return dict(z=z, yc=yc, thal=thal, is_ch=is_ch, results=results, S0=S0)


def save_csv(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["return_period_years", "discharge_m3s", "area_ha", "mean_depth_m", "max_depth_m", "volume_m3"])
        for r in model["results"]:
            wr.writerow([r["T"], f"{r['Q']:.1f}", f"{r['area_ha']:.2f}", f"{r['mean_depth']:.3f}",
                         f"{r['max_depth']:.3f}", f"{r['volume_m3']:.0f}"])


def save_figure(model, path, cap=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    z, yc, res = model["z"], model["yc"], model["results"]
    ext = [0, NX * DX, NY * DX, 0]
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(z, vert_exag=6, dx=DX, dy=DX)
    land = LinearSegmentedColormap.from_list("land", ["#d0dcc4", "#bea882"])
    base = land(np.clip((z - (100 - model["S0"] * (np.arange(NX) + .5) * DX)[None, :]) / 9, 0, 1))
    base[..., :3] *= (0.55 + 0.6 * shade)[..., None]
    base = np.clip(base, 0, 1)
    wcm = LinearSegmentedColormap.from_list("depth", ["#bee8fa", "#7dcef0", "#3298d8", "#1c66b6", "#283e98", "#481870"])

    fig = plt.figure(figsize=(15, 9), dpi=110, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    pick = [0, 3, 6]  # T = 2, 25, 200
    xs = (np.arange(NX) + .5) * DX
    ys = (np.arange(NY) + .5) * DX
    sx, sy = slice(4, NX, 9), slice(4, NY, 9)
    s = np.gradient(yc, DX)
    dirx, diry = 1 / np.sqrt(1 + s ** 2), s / np.sqrt(1 + s ** 2)
    for col, idx in enumerate(pick):
        ax = fig.add_subplot(gs[0, col])
        r = res[idx]
        ax.imshow(base, extent=ext, aspect="equal")
        dm = np.ma.masked_where(r["depth"] <= 0, r["depth"])
        im = ax.imshow(dm, extent=ext, cmap=wcm, vmin=0, vmax=cap, alpha=0.9, aspect="equal")
        V = r["vel"][sy, sx]
        U = V * dirx[sx][None, :]
        Vv = V * diry[sx][None, :]
        m = V > 0
        Xg, Yg = np.meshgrid(xs[sx], ys[sy])
        ax.quiver(Xg[m], Yg[m], U[m], Vv[m], color="#0b1d33", angles="xy", scale_units="xy", scale=0.012, width=0.0028, headwidth=4)
        ax.set_title(f"T = {r['T']} years   Q = {r['Q']:.0f} m³/s   area = {r['area_ha']:.1f} ha", fontsize=11)
        ax.set_xlabel("x (m)")
        if col == 0:
            ax.set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=fig.axes[:3], shrink=0.8, pad=0.01)
    cb.set_label("Depth (m)")

    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(base, extent=ext, aspect="equal")
    colors = ["#1b3a8c", "#2a6fd0", "#17a9c4", "#5cc45a", "#f0b429", "#ee5a36", "#a21caf"]
    for t in range(len(res) - 1, -1, -1):
        wet = np.ma.masked_where(res[t]["depth"] <= 0, np.ones_like(z))
        ax.imshow(wet, extent=ext, cmap=LinearSegmentedColormap.from_list("c", [colors[t], colors[t]]), alpha=0.8, aspect="equal")
    ax.set_title("Flood extent by return period (first flooded at)", fontsize=11)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels=[f"T {r['T']}" for r in res],
              loc="lower left", ncol=7, fontsize=8, framealpha=0.9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = fig.add_subplot(gs[1, 2])
    Ts = [str(r["T"]) for r in res]
    ax.bar(Ts, [r["area_ha"] for r in res], color="#4c96c8")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Flood area (ha)")
    ax2 = ax.twinx()
    ax2.plot(Ts, [r["mean_depth"] for r in res], "o-", color="#9a4a08")
    ax2.set_ylabel("Mean depth (m)", color="#9a4a08")
    ax.set_title("Flood area and mean depth", fontsize=11)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mean", type=float, default=300.0, help="mean annual peak discharge (m3/s)")
    ap.add_argument("--cv", type=float, default=0.55, help="coefficient of variation")
    ap.add_argument("--slope", type=float, default=0.002, help="bed slope (m/m)")
    ap.add_argument("--n-channel", type=float, default=0.035)
    ap.add_argument("--n-floodplain", type=float, default=0.07)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    model = run(a.mean, a.cv, a.slope, a.n_channel, a.n_floodplain)
    save_csv(model, root / "data" / "results.csv")
    print(f"{'T':>4} {'Q (m3/s)':>9} {'area (ha)':>10} {'mean h (m)':>11} {'max h (m)':>10}")
    for r in model["results"]:
        print(f"{r['T']:>4} {r['Q']:>9.0f} {r['area_ha']:>10.1f} {r['mean_depth']:>11.2f} {r['max_depth']:>10.2f}")
    if not a.no_figure:
        save_figure(model, root / "docs" / "preview.png")
        print("wrote docs/preview.png")
