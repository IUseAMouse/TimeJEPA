"""
Communication demo: TimeJEPA forecasts a VIDEO, pixel by pixel.

    python scripts/forecast_video.py --scene pendulum \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_v3_zs/pretrain_False/<champion>.ckpt \\
        --model-config lotsa_tiny_v3_eval

Principle (designed 2026-08-31): each pixel is a univariate intensity series,
handled EXACTLY like the harness handles multivariate data (per-channel split,
zero spatial coupling). The finetuned model produces the median (forecast
video) AND the q10-q90 fan (per-pixel uncertainty heatmap - the forecaster
drawing its own doubt). No model change, no training: the GIFT checkpoint as is.

Scenes generated in this script (no codec dependency, and the ground truth of
the continuation is available by construction; slow simulations are cached):
  * pendulum   - non-linear pendulum, RK4 (period ~128 frames).
  * vortex     - von Karman street behind a cylinder, D2Q9 lattice-Boltzmann,
    Re ~ 220; rendered field = vorticity. Periodic shedding: the nominal case.
  * ripples    - three-source ripple tank, analytic (periods 48/60 frames).
  * spirals    - Barkley reaction-diffusion spiral waves (period ~60 frames).
  * letters    - the same LBM channel with a TEXT-shaped obstacle (--text).
  * chaos      - double pendulum: genuinely chaotic; the fan is expected to
    WIDEN past the Lyapunov horizon (the calibrated-doubt scene).
  * convection - Rayleigh-Benard thermal LBM, unsteady plumes (~150 frames).

Output: a side-by-side GIF [truth | forecast median | fan width], snapshot
PNGs, and two honest numbers: model MAE vs persistence MAE (last context
frame frozen) over the forecast horizon.

Status: demo, never an official number. Forecast runs WITHOUT TTA by default
(--tta-flip to enable, official E19b formula).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydra import compose, initialize_config_dir              # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402


# ---------------------------------------------------------------- scenes

def render_disc(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Anti-aliased disc via a distance function - clean sub-pixel rendering."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(r + 0.5 - d, 0.0, 1.0)


def scene_pendulum(n_frames: int, h: int, w: int) -> np.ndarray:
    """NON-linear pendulum (RK4) - the period depends on amplitude, so the
    per-pixel pattern is not a textbook sinusoid: a real test."""
    # Period 128 frames and disc h/7: the transit over one pixel lasts ~8-15
    # frames - ABOVE the patch resolution (16/8). Measured 2026-08-31: at
    # period 64 / radius h/12, per-pixel pulses last 2-3 frames and the
    # pinball-optimal median is ~0 everywhere (max 0.03) - the bizitobs case
    # in video. The demo must stay in the regime the model resolves.
    g_over_l = (2 * np.pi / 128.0) ** 2
    theta, omega = np.deg2rad(40.0), 0.0
    dt, sub = 1.0, 8                          # 8 RK4 sub-steps per frame

    pivot = (w / 2.0, h * 0.12)
    rod = h * 0.62
    radius = max(2.0, h / 7.0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    for t in range(n_frames):
        for _ in range(sub):
            def deriv(th, om):
                return om, -g_over_l * np.sin(th)
            k1 = deriv(theta, omega)
            k2 = deriv(theta + 0.5 * dt / sub * k1[0], omega + 0.5 * dt / sub * k1[1])
            k3 = deriv(theta + 0.5 * dt / sub * k2[0], omega + 0.5 * dt / sub * k2[1])
            k4 = deriv(theta + dt / sub * k3[0], omega + dt / sub * k3[1])
            theta += dt / sub / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            omega += dt / sub / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        cx = pivot[0] + rod * np.sin(theta)
        cy = pivot[1] + rod * np.cos(theta)
        frames[t] = render_disc(h, w, cx, cy, radius)
    return frames


def scene_vortex(n_frames: int, h: int, w: int, cache: Path,
                 record_every: int = 35, warmup: int = 20000) -> np.ndarray:
    """Von Karman street - D2Q9 LBM past a cylinder, rendered field = vorticity.

    Deliberately minimal solver (BGK, simple bounce-back on the obstacle,
    approximate Zou/He at the borders): the goal is a visually correct
    PERIODIC shedding, not production CFD. Simulated at 3x the output
    resolution then averaged - free anti-aliasing.

    Measured calibration (2026-08-31, failed v1): at warmup 4000 the shedding
    NEVER started (no autocorr peak, persistence MAE 0.0069 - near-static
    field, the model was forecasting an aperiodic transient). The instability
    takes ~15-25k steps to establish; the initial state is perturbed on top
    of the off-centering. Target period: St~0.2 -> T ~ D/(St*ulb) lattice
    steps; with D=2*(3h//9) and record_every=35, ~70-80 frames/period -> ~13
    periods in the 1024 context.
    """
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"vortex: cache {cache}")
            return arr

    nx, ny = w * 3, h * 3
    re, ulb = 220.0, 0.04
    cyl_r = ny // 9
    cx, cy = nx // 4, ny // 2 + 2              # +2: breaks symmetry, seeds the shedding
    nulb = ulb * 2 * cyl_r / re
    omega = 1.0 / (3 * nulb + 0.5)

    # D2Q9
    v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
                  [0, -1], [-1, 1], [-1, 0], [-1, -1]])
    t9 = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1], dtype=np.float64) / 36.0
    col_left, col_mid, col_right = [0, 1, 2], [3, 4, 5], [6, 7, 8]

    yy, xx = np.mgrid[0:ny, 0:nx]
    obstacle = (xx - cx) ** 2 + (yy - cy) ** 2 < cyl_r ** 2

    def equilibrium(rho, u):
        cu = 3.0 * np.einsum('qd,dyx->qyx', v, u)
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
        return rho * t9[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usqr)

    vel_in = np.zeros((2, ny, nx)); vel_in[0] = ulb
    # Perturbed initial state (not just off-centered): without it the
    # metastable symmetric wake survives the whole warmup (measured, v1).
    vel_0 = vel_in.copy()
    vel_0[0] *= 1.0 + 0.04 * np.sin(2 * np.pi * yy / ny) \
                    * np.sin(2 * np.pi * xx / nx)
    fin = equilibrium(np.ones((ny, nx)), vel_0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        fin[col_right, :, -1] = fin[col_right, :, -2]          # free outlet
        rho = fin.sum(axis=0)
        u = np.einsum('qd,qyx->dyx', v, fin) / rho
        u[:, :, 0] = vel_in[:, :, 0]                           # imposed inlet
        rho[:, 0] = 1.0 / (1.0 - u[0, :, 0]) * (
            fin[col_mid, :, 0].sum(axis=0) + 2 * fin[col_right, :, 0].sum(axis=0))
        feq = equilibrium(rho, u)
        fin[col_left, :, 0] = feq[col_left, :, 0] \
            + fin[col_right[::-1], :, 0] - feq[col_right[::-1], :, 0]
        fout = fin - omega * (fin - feq)
        for q in range(9):                                     # obstacle bounce-back
            fout[q, obstacle] = fin[8 - q, obstacle]
        for q in range(9):                                     # streaming
            fin[q] = np.roll(np.roll(fout[q], v[q, 0], axis=1), v[q, 1], axis=0)

        k = step - warmup
        if k >= 0 and k % record_every == 0:
            vort = (np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)
                    - np.roll(u[0], -1, axis=0) + np.roll(u[0], 1, axis=0))
            vort[obstacle] = 0.0
            small = vort.reshape(h, 3, w, 3).mean(axis=(1, 3))
            frames[k // record_every] = small
        if step % 5000 == 0:
            print(f"vortex: step {step}/{total}")

    # Signed vorticity -> [0,1] through a robust GLOBAL scale (q99): the same
    # for all frames, otherwise normalization would destroy the dynamics.
    s = np.quantile(np.abs(frames), 0.99)
    frames = np.clip(frames / (2 * max(s, 1e-9)) + 0.5, 0.0, 1.0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames



def scene_ripples(n_frames: int, h: int, w: int) -> np.ndarray:
    """Ripple tank: three point sources, two temporal periods (48 and 60
    frames - both inside the model's band), 1/sqrt(r) decay. Analytic, so the
    per-pixel truth is a sum of two sines: the cleanest possible showcase of
    the fan staying razor thin when the scene is genuinely predictable."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lam = w / 6.0
    srcs = [(h * 0.35, w * 0.30, 48.0, 0.0),
            (h * 0.65, w * 0.70, 48.0, np.pi / 2),
            (h * 0.25, w * 0.75, 60.0, 0.0)]
    t = np.arange(n_frames, dtype=np.float32)[:, None, None]
    z = np.zeros((n_frames, h, w), dtype=np.float32)
    for cy, cx, period, phase in srcs:
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        env = 1.0 / np.sqrt(1.0 + r / lam)
        z += env[None] * np.sin(2 * np.pi * (r[None] / lam - t / period) + phase)
    sc = np.quantile(np.abs(z), 0.99)
    return np.clip(z / (2 * max(sc, 1e-9)) + 0.5, 0.0, 1.0).astype(np.float32)


def scene_spirals(n_frames: int, h: int, w: int, cache: Path,
                  record_every: int = 2, warmup: int = 6000) -> np.ndarray:
    """Rotating spiral waves - Barkley reaction-diffusion (the
    Belousov-Zhabotinsky aesthetic). Every pixel is a relaxation oscillator
    with a stable period: the model's nominal regime, in chemistry instead of
    fluid dynamics. Spiral seeded by crossed half-plane initial conditions."""
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"spirals: cache {cache}")
            return arr
    ny, nx = h * 2, w * 2
    # eps=0.06 is sub-excitable (wave dies in 60 time units - measured);
    # canonical Barkley eps=0.02 rotates at ~115 steps/period at dt=0.05.
    a, b, eps, dt = 0.75, 0.06, 0.02, 0.05
    yy, xx = np.mgrid[0:ny, 0:nx]
    u = (yy > ny // 2).astype(np.float64)
    v = np.where(xx > nx // 2, a / 2.0, 0.0)

    def lap(f):
        # No-flux (Neumann) edges: with periodic wrap the broken wavefront
        # annihilates itself and the medium dies (measured in the smoke).
        g = np.pad(f, 1, mode="edge")
        return (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
                - 4 * f)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        uth = (v + b) / a
        u = u + dt * (u * (1 - u) * (u - uth) / eps + lap(u))
        v = v + dt * (u - v)
        u = np.clip(u, 0.0, 1.0)
        k = step - warmup
        if k >= 0 and k % record_every == 0:
            frames[k // record_every] = u.reshape(h, 2, w, 2).mean(axis=(1, 3))
        if step % 10000 == 0:
            print(f"spirals: step {step}/{total}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


def _lbm_periodic_shear(n_frames: int, h: int, w: int, cache: Path,
                        record_every: int = 100, warmup: int = 25000) -> np.ndarray:
    """Kelvin-Helmholtz billows - D2Q9 LBM, fully periodic double shear layer
    (a +u band between two -u bands: periodic-compatible). Rendered field =
    vorticity. NOT exposed in the CLI: three measured calibrations found no
    per-pixel periodicity at any sampling stride (the roll-up is a one-shot
    transient, the merged state a slow aperiodic drift). Kept for
    experimentation per the no-delete policy."""
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"shear: cache {cache}")
            return arr
    ny, nx = h * 3, w * 3
    ulb, tau = 0.06, 0.56
    omega = 1.0 / tau
    v9 = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
                   [0, -1], [-1, 1], [-1, 0], [-1, -1]])
    t9 = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1], dtype=np.float64) / 36.0

    def equilibrium(rho, u):
        cu = 3.0 * np.einsum('qd,dyx->qyx', v9, u)
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
        return rho * t9[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usqr)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    d = ny / 40.0
    u0 = np.zeros((2, ny, nx))
    u0[0] = ulb * (np.tanh((yy - ny / 4) / d) - np.tanh((yy - 3 * ny / 4) / d) - 1.0)
    u0[1] = 0.02 * ulb * np.sin(2 * np.pi * 2 * xx / nx)
    fin = equilibrium(np.ones((ny, nx)), u0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        rho = fin.sum(axis=0)
        u = np.einsum('qd,qyx->dyx', v9, fin) / rho
        feq = equilibrium(rho, u)
        fout = fin - omega * (fin - feq)
        for q in range(9):
            fin[q] = np.roll(np.roll(fout[q], v9[q, 0], axis=1), v9[q, 1], axis=0)
        k = step - warmup
        if k >= 0 and k % record_every == 0:
            vort = (np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)
                    - np.roll(u[0], -1, axis=0) + np.roll(u[0], 1, axis=0))
            frames[k // record_every] = vort.reshape(h, 3, w, 3).mean(axis=(1, 3))
        if step % 10000 == 0:
            print(f"shear: step {step}/{total}")
    sc = np.quantile(np.abs(frames), 0.99)
    frames = np.clip(frames / (2 * max(sc, 1e-9)) + 0.5, 0.0, 1.0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


def _letters_mask(ny: int, nx: int, text: str) -> np.ndarray:
    """Obstacle mask from rendered text (default PIL bitmap font, scaled)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (nx, ny), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    # Target: letters ~30% of the channel height. 45% chokes the channel and
    # suppresses the shedding (smoke: 3% active pixels, period 2 = noise);
    # the default-font scale of v1 gave 0.2% of the domain - invisible wake.
    scale = max(1, int(ny * 0.30 / th))
    small = Image.new("L", (tw + 2, th + 2), 0)
    ImageDraw.Draw(small).text((1 - x0, 1 - y0), text, fill=255, font=font)
    big = small.resize((small.width * scale, small.height * scale), Image.NEAREST)
    img.paste(big, (nx // 6, (ny - big.height) // 2))
    return np.array(img) > 127


def scene_flow(n_frames: int, h: int, w: int, cache: Path, re: float,
               obstacle: np.ndarray = None, record_every: int = 35,
               warmup: int = 20000, char_len: int = None) -> np.ndarray:
    """Channel flow past an ARBITRARY obstacle - the scene_vortex numerics,
    parameterized. Used by the 'letters' scene (text-shaped obstacle) and the
    'turbulence' scene (cylinder at high Re: the wake loses periodicity and
    the fan is expected to WIDEN - the calibrated-doubt demo)."""
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"flow: cache {cache}")
            return arr
    nx, ny = w * 3, h * 3
    ulb = 0.04
    if obstacle is None:
        cyl_r = ny // 9
        cy, cx = ny // 2 + 2, nx // 4
        yy, xx = np.mgrid[0:ny, 0:nx]
        obstacle = (xx - cx) ** 2 + (yy - cy) ** 2 < cyl_r ** 2
        char_len = 2 * cyl_r
    char_len = char_len or max(int(obstacle.any(axis=1).sum()), 8)
    nulb = ulb * char_len / re
    omega = 1.0 / (3 * nulb + 0.5)
    v9 = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
                   [0, -1], [-1, 1], [-1, 0], [-1, -1]])
    t9 = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1], dtype=np.float64) / 36.0
    col_mid, col_right, col_left = [3, 4, 5], [6, 7, 8], [0, 1, 2]

    def equilibrium(rho, u):
        cu = 3.0 * np.einsum('qd,dyx->qyx', v9, u)
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
        return rho * t9[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usqr)

    yy, xx = np.mgrid[0:ny, 0:nx]
    vel_in = np.zeros((2, ny, nx)); vel_in[0] = ulb
    vel_0 = vel_in.copy()
    vel_0[0] *= 1.0 + 0.04 * np.sin(2 * np.pi * yy / ny) \
                    * np.sin(2 * np.pi * xx / nx)
    fin = equilibrium(np.ones((ny, nx)), vel_0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        fin[col_right, :, -1] = fin[col_right, :, -2]
        rho = fin.sum(axis=0)
        u = np.einsum('qd,qyx->dyx', v9, fin) / rho
        u[:, :, 0] = vel_in[:, :, 0]
        rho[:, 0] = 1.0 / (1.0 - u[0, :, 0]) * (
            fin[col_mid, :, 0].sum(axis=0) + 2 * fin[col_right, :, 0].sum(axis=0))
        feq = equilibrium(rho, u)
        fin[col_left, :, 0] = feq[col_left, :, 0] \
            + fin[col_right[::-1], :, 0] - feq[col_right[::-1], :, 0]
        fout = fin - omega * (fin - feq)
        for q in range(9):
            fout[q, obstacle] = fin[8 - q, obstacle]
        for q in range(9):
            fin[q] = np.roll(np.roll(fout[q], v9[q, 0], axis=1), v9[q, 1], axis=0)
        k = step - warmup
        if k >= 0 and k % record_every == 0:
            vort = (np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)
                    - np.roll(u[0], -1, axis=0) + np.roll(u[0], 1, axis=0))
            vort[obstacle] = 0.0
            frames[k // record_every] = vort.reshape(h, 3, w, 3).mean(axis=(1, 3))
        if step % 5000 == 0:
            print(f"flow: step {step}/{total}")
    sc = np.quantile(np.abs(frames), 0.99)
    frames = np.clip(frames / (2 * max(sc, 1e-9)) + 0.5, 0.0, 1.0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


def scene_chaos(n_frames: int, h: int, w: int) -> np.ndarray:
    """Double pendulum (RK4) - genuine chaos. The point of this scene is the
    OPPOSITE of the others: past the Lyapunov horizon no forecaster can know
    the trajectory, and an honest one must say so - the fan is expected to
    WIDEN into a probability cloud where the periodic scenes keep it razor
    thin. (Replaces the high-Re "turbulence" idea: at this LBM resolution
    tau hits 0.5 and blows up, and a 2D wake stays periodic far past
    Re 1000 anyway - the smoke killed it.)"""
    m1 = m2 = 1.0
    l1 = l2 = 1.0
    g = (2 * np.pi / 90.0) ** 2 * l1     # ~90-frame small-swing timescale

    def deriv(st):
        t1, w1, t2, w2 = st
        d = t1 - t2
        den = 2 * m1 + m2 - m2 * np.cos(2 * d)
        a1 = (-g * (2 * m1 + m2) * np.sin(t1) - m2 * g * np.sin(t1 - 2 * t2)
              - 2 * np.sin(d) * m2 * (w2 ** 2 * l2 + w1 ** 2 * l1 * np.cos(d))) \
            / (l1 * den)
        a2 = (2 * np.sin(d) * (w1 ** 2 * l1 * (m1 + m2)
                               + g * (m1 + m2) * np.cos(t1)
                               + w2 ** 2 * l2 * m2 * np.cos(d))) / (l2 * den)
        return np.array([w1, a1, w2, a2])

    st = np.array([np.deg2rad(120.0), 0.0, np.deg2rad(-10.0), 0.0])
    dt, sub = 1.0, 16
    pivot = (w / 2.0, h * 0.28)
    arm = h * 0.30
    radius = max(2.0, h / 9.0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    for t in range(n_frames):
        for _ in range(sub):
            k1 = deriv(st)
            k2 = deriv(st + 0.5 * dt / sub * k1)
            k3 = deriv(st + 0.5 * dt / sub * k2)
            k4 = deriv(st + dt / sub * k3)
            st = st + dt / sub / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x1 = pivot[0] + arm * np.sin(st[0]); y1 = pivot[1] + arm * np.cos(st[0])
        x2 = x1 + arm * np.sin(st[2]);       y2 = y1 + arm * np.cos(st[2])
        frames[t] = np.maximum(0.6 * render_disc(h, w, x1, y1, radius * 0.8),
                               render_disc(h, w, x2, y2, radius))
    return frames


def scene_convection(n_frames: int, h: int, w: int, cache: Path,
                     record_every: int = 60, warmup: int = 12000) -> np.ndarray:
    """Rayleigh-Benard convection - thermal LBM (D2Q9 flow + D2Q5
    temperature, Boussinesq buoyancy via velocity shift). Hot floor, cold
    ceiling, periodic sides; rendered field = temperature (the lava-lamp
    look). Parameters aimed at the UNSTEADY plume regime: above the
    oscillatory onset, below chaos."""
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"convection: cache {cache}")
            return arr
    ny, nx = h * 2, w * 2
    tau_f, tau_g = 1.0, 0.8
    g_beta = 0.0001             # bisected: 3e-4 blows up (NaN), 3e-5 stays
                                # conductive; 1e-4 gives unsteady plumes with a
                                # ~150-frame period (95% active pixels, smoke)
    v9 = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
                   [0, -1], [-1, 1], [-1, 0], [-1, -1]])
    t9 = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1], dtype=np.float64) / 36.0
    v5 = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    t5 = np.array([2, 1, 1, 1, 1], dtype=np.float64) / 6.0

    def eq9(rho, u):
        cu = 3.0 * np.einsum('qd,dyx->qyx', v9, u)
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
        return rho * t9[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usqr)

    def eq5(T, u):
        cu = 3.0 * np.einsum('qd,dyx->qyx', v5, u)
        return T * t5[:, None, None] * (1 + cu)

    yy = np.mgrid[0:ny, 0:nx][0].astype(np.float64)
    T = 1.0 - yy / (ny - 1)
    rng0 = np.random.default_rng(0)
    T += 0.02 * rng0.standard_normal(T.shape)
    fin = eq9(np.ones((ny, nx)), np.zeros((2, ny, nx)))
    gin = eq5(T, np.zeros((2, ny, nx)))

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        rho = fin.sum(axis=0)
        u = np.einsum('qd,qyx->dyx', v9, fin) / rho
        T = gin.sum(axis=0)
        # Boussinesq: buoyancy enters as a velocity shift in the equilibrium.
        ub = u.copy()
        ub[1] += np.clip(tau_f * g_beta * (T - 0.5) * ny / rho, -0.05, 0.05)
        fout = fin - (fin - eq9(rho, ub)) / tau_f
        gout = gin - (gin - eq5(T, u)) / tau_g
        # walls: no-slip floor/ceiling (bounce-back), fixed T (anti-bounce-back)
        fout[[3, 0, 6], 0, :] = fin[[5, 8, 2], 0, :]
        fout[[5, 2, 8], -1, :] = fin[[3, 6, 0], -1, :]
        for q in range(9):
            fin[q] = np.roll(np.roll(fout[q], v9[q, 0], axis=1), v9[q, 1], axis=0)
        for q in range(5):
            gin[q] = np.roll(np.roll(gout[q], v5[q, 0], axis=1), v5[q, 1], axis=0)
        gin[3, 0, :] = t5[3] * 2 * 1.0 - gin[4, 0, :]          # hot floor T=1
        gin[4, -1, :] = t5[4] * 2 * 0.0 - gin[3, -1, :]        # cold ceiling T=0
        k = step - warmup
        if k >= 0 and k % record_every == 0:
            frames[k // record_every] = np.clip(
                T.reshape(h, 2, w, 2).mean(axis=(1, 3)), 0.0, 1.0)
        if step % 10000 == 0:
            print(f"convection: step {step}/{total} (T in [{T.min():.2f},{T.max():.2f}])")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


# ---------------------------------------------------------------- forecast

def forecast_series(model, series: np.ndarray, horizon: int,
                    device, chunk: int, tta_flip: bool) -> np.ndarray:
    """[N, ctx] -> fan [N, horizon, 9]. The core shared by both modes."""
    n = series.shape[0]
    fans = np.empty((n, horizon, 9), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = torch.from_numpy(series[i:i + chunk, :, None]).float().to(device)
            q = model.forecast(x, n=horizon)["quantiles_denorm"]
            if tta_flip:
                # q_tau(x) <- (q_tau(x) - q_{1-tau}(-x)) / 2  (E19b, official)
                q_neg = model.forecast(-x, n=horizon)["quantiles_denorm"]
                q = 0.5 * (q - torch.flip(q_neg, dims=[-1]))
            fans[i:i + chunk] = q.cpu().numpy()
    return fans


def forecast_pixels(model, video: np.ndarray, ctx_len: int, horizon: int,
                    device, chunk: int, tta_flip: bool):
    """Pixel mode: each pixel = one series. -> (med, q10, q90, mean) [h,H,W]."""
    t_all, h, w = video.shape
    assert t_all >= ctx_len + horizon, "video too short for context+horizon"
    ctx = video[t_all - horizon - ctx_len: t_all - horizon]      # [ctx, H, W]
    q = forecast_series(model, ctx.reshape(ctx_len, h * w).T,
                        horizon, device, chunk, tta_flip)         # [P, h, 9]
    shape = (horizon, h, w)
    # Fan mean ~ E[x]: for a near-binary pixel this is the presence
    # probability - the "ghost ball", the communication pane.
    return (q[..., 4].T.reshape(shape), q[..., 0].T.reshape(shape),
            q[..., 8].T.reshape(shape), q.mean(-1).T.reshape(shape))


def forecast_pca(model, video: np.ndarray, ctx_len: int, horizon: int,
                 device, chunk: int, tta_flip: bool,
                 n_modes: int, mc_samples: int, seed: int = 0):
    """POD/PCA mode: forecast the MODAL COEFFICIENTS, not the pixels.

    The scene has few degrees of freedom (pendulum: ~1, vortex street: a
    traveling wave, ~2 modes in quadrature); pixel by pixel we ask 1600
    intermittent questions where the scene only asks k. PCA on the CONTEXT
    frames ONLY (modes never see the future - zero leakage): video ~ mu +
    sum a_j(t)*mode_j. Each a_j is a SMOOTH, periodic series - the model's
    nominal regime - and we forecast k series instead of H*W. The paper's
    philosophy applied to the demo: predict in a representation space,
    decode for display.

    Pixel uncertainty via Monte Carlo: one quantile level drawn PER MODE
    (inter-mode independence - PCA decorrelates on the context, stated
    assumption), keeping that level's whole trajectory (intra-mode temporal
    coherence, same logic as the comonotone fan rollout).
    """
    t_all, h, w = video.shape
    ctx = video[t_all - horizon - ctx_len: t_all - horizon].reshape(ctx_len, -1)
    mu = ctx.mean(axis=0)                                        # [P]
    x0 = ctx - mu
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    k = min(n_modes, vt.shape[0])
    modes = vt[:k]                                               # [k, P]
    coeffs = x0 @ modes.T                                        # [ctx, k]
    evr = float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-12))
    print(f"PCA: {k} modes, context variance explained {evr:.1%}")

    q = forecast_series(model, coeffs.T.astype(np.float32),
                        horizon, device, chunk, tta_flip)        # [k, h, 9]
    med = mu + q[..., 4].T @ modes                               # [h, P]
    mean = mu + q.mean(-1).T @ modes

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 9, size=(mc_samples, k))
    samples = np.empty((mc_samples, horizon, mu.size), dtype=np.float32)
    for si in range(mc_samples):
        traj = q[np.arange(k), :, idx[si]]                       # [k, h]
        samples[si] = mu + traj.T @ modes
    q10 = np.quantile(samples, 0.1, axis=0)
    q90 = np.quantile(samples, 0.9, axis=0)

    shape = (horizon, h, w)
    return (med.reshape(shape).astype(np.float32),
            q10.reshape(shape).astype(np.float32),
            q90.reshape(shape).astype(np.float32),
            mean.reshape(shape).astype(np.float32))


# ---------------------------------------------------------------- rendering

PANE_LABELS = ("ground truth", "TimeJEPA median", "TimeJEPA mean",
               "uncertainty q90-q10")


def to_gif(path: Path, truth: np.ndarray, med: np.ndarray, mean: np.ndarray,
           width: np.ndarray, ctx_tail: np.ndarray, upscale: int, fps: int):
    """Labeled 2x2 GRID GIF (post format):
        [ truth | median ]       context first, then horizon
        [ mean  | uncertainty ]  (red border = we are forecasting).
    Bug fixed 2026-08-31: the strip v1 only assembled 3 of the 4 panes."""
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    magma = colormaps["magma"]

    wmax = max(float(width.max()), 1e-9)
    frames = []
    n_tail = len(ctx_tail)
    pane_w = truth.shape[2] * upscale
    pane_h = truth.shape[1] * upscale
    cap_h, gap = 14, 2
    grid_w = 2 * pane_w + gap
    grid_h = 2 * (cap_h + pane_h) + gap
    caption_rows = None
    for t in range(n_tail + len(truth)):
        if t < n_tail:                                   # context phase: same
            g = np.clip(ctx_tail[t], 0, 1)               # video in every pane,
            panes = [g, g, g, np.zeros_like(g)]          # zero uncertainty
        else:
            k = t - n_tail
            panes = [np.clip(truth[k], 0, 1), np.clip(med[k], 0, 1),
                     np.clip(mean[k], 0, 1), width[k] / wmax]
        cells = []
        for j, p in enumerate(panes):
            rgb = (magma(p)[..., :3] if j == 3
                   else np.repeat(p[..., None], 3, axis=-1))
            a = (rgb * 255).astype(np.uint8)
            a = np.repeat(np.repeat(a, upscale, axis=0), upscale, axis=1)
            cells.append(Image.fromarray(a))
        if caption_rows is None:                         # captions rendered once
            caption_rows = []
            for row in (PANE_LABELS[:2], PANE_LABELS[2:]):
                strip = Image.new("RGB", (grid_w, cap_h), (28, 28, 28))
                d = ImageDraw.Draw(strip)
                for j, lab in enumerate(row):
                    x0 = j * (pane_w + gap)
                    tw = d.textlength(lab)
                    d.text((x0 + max(0, (pane_w - tw) // 2), 2), lab,
                           fill=(220, 220, 220))
                caption_rows.append(strip)
        full = Image.new("RGB", (grid_w, grid_h), (90, 90, 90))
        for j, cell in enumerate(cells):
            cx = (j % 2) * (pane_w + gap)
            cy = (j // 2) * (cap_h + pane_h + gap)
            full.paste(caption_rows[j // 2], (0, cy)) if j % 2 == 0 else None
            full.paste(cell, (cx, cy + cap_h))
        if t >= n_tail:                                  # red frame: forecasting
            d = ImageDraw.Draw(full)
            d.rectangle([0, 0, grid_w - 1, grid_h - 1],
                        outline=(255, 60, 60), width=2)
        frames.append(full)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=1000 // fps, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="pendulum",
                    choices=["pendulum", "vortex", "ripples", "spirals",
                             "letters", "chaos", "convection"])
    ap.add_argument("--re", type=float, default=220.0,
                    help="Reynolds for the letters scene (LBM stability caps "
                         "usable values around ~240 at this resolution)")
    ap.add_argument("--text", default="TJ",
                    help="obstacle text for the letters scene")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="lotsa_tiny_v3_eval")
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024, help="series per forward")
    ap.add_argument("--mode", choices=["pixel", "pca"], default="pixel",
                    help="pixel = 1 series per pixel; pca = forecast the modal "
                         "coefficients (POD), the \"predict in a "
                         "representation\" demo")
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--mc-samples", type=int, default=128,
                    help="MC samples for pixel uncertainty (pca mode)")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--tail", type=int, default=96,
                    help="context frames shown before the red border")
    ap.add_argument("--out", default="evaluation/video_forecast")
    args = ap.parse_args()

    n_frames = args.context + args.horizon

    def _cache(name):
        return Path(args.out) / f"{name}_{args.height}x{args.width}_{n_frames}.npy"

    if args.scene == "pendulum":
        video = scene_pendulum(n_frames, args.height, args.width)
    elif args.scene == "vortex":
        # v2 in the name: the solver calibration is part of the cache
        # identity (the non-periodic v1 has the same array shape).
        video = scene_vortex(n_frames, args.height, args.width, _cache("vortex_v2"))
    elif args.scene == "ripples":
        video = scene_ripples(n_frames, args.height, args.width)
    elif args.scene == "spirals":
        video = scene_spirals(n_frames, args.height, args.width, _cache("spirals"))
    elif args.scene == "letters":
        re = args.re
        ny, nx = args.height * 3, args.width * 3
        mask = _letters_mask(ny, nx, args.text)
        video = scene_flow(n_frames, args.height, args.width,
                           _cache(f"letters-{args.text}-re{re:.0f}"), re,
                           obstacle=mask)
    elif args.scene == "chaos":
        video = scene_chaos(n_frames, args.height, args.width)
    else:
        video = scene_convection(n_frames, args.height, args.width,
                                 _cache("convection"))

    # Guard (measured 2026-08-31, one evening lost): a PRETRAIN checkpoint
    # carries a decoder that exists but was NEVER trained (JEPA gives it no
    # gradient), and the P3.2 contract only covers the core - loading
    # succeeds, outputs are plausible noise (corr 0.00 on a bare sinusoid,
    # vs 1.00 for the finetuned one). Reliable discriminant: the Lightning
    # hyper_parameters of the training module.
    hp = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False).get("hyper_parameters", {})
    if "finetune_mode" not in hp and "contextualized_targets" in hp:
        raise SystemExit(
            f"REFUSED: {args.checkpoint} is a PRETRAIN checkpoint "
            "(JEPA hyper_parameters, no finetune_mode) - its quantile head "
            "is at initialization. The video demo requires a FINETUNED "
            "checkpoint (*_zs/pretrain_False or champions/ directories).")

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    if args.mode == "pca":
        med, q10, q90, mean = forecast_pca(
            model, video, args.context, args.horizon, device, args.chunk,
            args.tta_flip, args.n_modes, args.mc_samples)
    else:
        med, q10, q90, mean = forecast_pixels(
            model, video, args.context, args.horizon, device, args.chunk,
            args.tta_flip)
    truth = video[-args.horizon:]
    persist = np.repeat(video[-args.horizon - 1][None], args.horizon, axis=0)

    mae_model = float(np.abs(med - truth).mean())
    mae_persist = float(np.abs(persist - truth).mean())
    cov80 = float(((truth >= q10) & (truth <= q90)).mean())
    print(f"model MAE       : {mae_model:.4f}")
    print(f"persistence MAE : {mae_persist:.4f}  (ratio {mae_model / mae_persist:.3f})")
    print(f"80% coverage    : {cov80:.3f} (nominal 0.800)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scene}_{Path(args.checkpoint).stem}" \
          + (f"_pca{args.n_modes}" if args.mode == "pca" else "") \
          + ("_flip" if args.tta_flip else "")
    to_gif(out / f"{tag}.gif", truth, med, mean, q90 - q10,
           video[-args.horizon - args.tail:-args.horizon],
           args.upscale, args.fps)
    for k in (0, args.horizon // 4, args.horizon // 2, args.horizon - 1):
        from PIL import Image
        row = np.concatenate([truth[k], med[k], mean[k]], axis=1)
        Image.fromarray((np.clip(row, 0, 1) * 255).astype(np.uint8)) \
             .resize((row.shape[1] * args.upscale, row.shape[0] * args.upscale),
                     Image.NEAREST).save(out / f"{tag}_frame{k:03d}.png")
    print(f"Outputs: {out}/{tag}.gif")


if __name__ == "__main__":
    main()
