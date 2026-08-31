#!/usr/bin/env python
"""
Démo de communication : TimeJEPA forecast une VIDÉO, pixel par pixel.

    python scripts/forecast_video.py --scene pendulum \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_v3_zs/pretrain_False/<champion>.ckpt \\
        --model-config lotsa_tiny_v3_eval

Principe (conçu 2026-08-31) : chaque pixel est une série univariée d'intensité,
traitée EXACTEMENT comme le harnais traite le multivarié (éclatement par canal,
zéro couplage spatial). Le modèle finetuné produit la médiane (vidéo forecast)
ET le fan q10-q90 (heatmap d'incertitude par pixel — l'image que personne ne
montre : le forecaster qui dessine son propre doute). Aucune modification du
modèle, aucun entraînement : c'est le checkpoint GIFT tel quel.

Deux scènes générées ici même (pas de dépendance codec, et la vérité terrain
de la continuation est disponible par construction) :
  * pendulum — pendule non linéaire intégré en RK4 (période ~64 frames à
    l'amplitude par défaut : ~16 périodes dans le contexte de 1024 frames).
  * vortex   — allée de von Kármán derrière un cylindre, solveur
    lattice-Boltzmann D2Q9 minimal (Re ~ 220) ; le champ rendu est la
    vorticité. Le lâcher tourbillonnaire est périodique : cas nominal pour un
    forecast par pixel. La simulation est mise en cache (.npy) — la partie
    lente ne tourne qu'une fois.

Sortie : un GIF côte à côte [vérité | médiane forecastée | largeur du fan]
(la barre verticale marque la frontière contexte→forecast), des PNG
d'instantanés, et deux chiffres honnêtes : MAE du modèle vs MAE de la
persistance (dernière frame du contexte figée) sur l'horizon forecasté.

Statut : démo, jamais un chiffre officiel. Le forecast est fait SANS TTA par
défaut (--tta-flip pour l'activer, formule officielle E19b).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydra import compose, initialize_config_dir              # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402


# ---------------------------------------------------------------- scènes

def render_disc(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Disque anti-aliasé par fonction de distance — rendu sub-pixel propre."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(r + 0.5 - d, 0.0, 1.0)


def scene_pendulum(n_frames: int, h: int, w: int) -> np.ndarray:
    """Pendule NON linéaire (RK4) — la période dépend de l'amplitude, donc le
    motif par pixel n'est pas une sinusoïde de manuel : un vrai test."""
    # Période 128 frames et disque h/7 : le transit sur un pixel dure ~8-15
    # frames — AU-DESSUS de la résolution du patch (16/8). Mesuré 2026-08-31 :
    # à période 64 / rayon h/12, les impulsions par pixel font 2-3 frames et la
    # médiane pinball-optimale est ~0 partout (max 0.03) — le cas bizitobs en
    # vidéo. La démo doit rester dans le régime que le modèle résout.
    g_over_l = (2 * np.pi / 128.0) ** 2
    theta, omega = np.deg2rad(40.0), 0.0
    dt, sub = 1.0, 8                          # 8 sous-pas RK4 par frame

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
    """Allée de von Kármán — LBM D2Q9 sur cylindre, champ rendu = vorticité.

    Solveur volontairement minimal (BGK, rebond simple sur l'obstacle,
    Zou/He approximé aux bords) : l'objectif est un lâcher PÉRIODIQUE
    visuellement juste, pas une CFD de production. Simulation à 3x la
    résolution de sortie puis moyennage — l'anti-aliasing gratuit.

    Calibration mesurée (2026-08-31, v1 ratée) : à warmup 4000 le lâcher n'a
    JAMAIS démarré (autocorr sans pic, persistance MAE 0.0069 — champ quasi
    statique, le modèle forecastait un transitoire apériodique). L'instabilité
    met ~15-25k pas à s'établir ; on perturbe l'état initial en plus du
    décentrage. Période visée : St≈0.2 → T ≈ D/(St·ulb) pas de lattice ; avec
    D=2·(3h//9) et record_every=35, ~70-80 frames/période → ~13 périodes dans
    le contexte de 1024.
    """
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_frames, h, w):
            print(f"vortex : cache {cache}")
            return arr

    nx, ny = w * 3, h * 3
    re, ulb = 220.0, 0.04
    cyl_r = ny // 9
    cx, cy = nx // 4, ny // 2 + 2              # +2 : brise la symétrie, amorce le lâcher
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
    # État initial perturbé (pas seulement décentré) : sans cela le sillage
    # symétrique métastable survit à tout le warmup (mesuré, v1).
    vel_0 = vel_in.copy()
    vel_0[0] *= 1.0 + 0.04 * np.sin(2 * np.pi * yy / ny) \
                    * np.sin(2 * np.pi * xx / nx)
    fin = equilibrium(np.ones((ny, nx)), vel_0)

    frames = np.empty((n_frames, h, w), dtype=np.float32)
    total = warmup + n_frames * record_every
    for step in range(total):
        fin[col_right, :, -1] = fin[col_right, :, -2]          # sortie libre
        rho = fin.sum(axis=0)
        u = np.einsum('qd,qyx->dyx', v, fin) / rho
        u[:, :, 0] = vel_in[:, :, 0]                           # entrée imposée
        rho[:, 0] = 1.0 / (1.0 - u[0, :, 0]) * (
            fin[col_mid, :, 0].sum(axis=0) + 2 * fin[col_right, :, 0].sum(axis=0))
        feq = equilibrium(rho, u)
        fin[col_left, :, 0] = feq[col_left, :, 0] \
            + fin[col_right[::-1], :, 0] - feq[col_right[::-1], :, 0]
        fout = fin - omega * (fin - feq)
        for q in range(9):                                     # rebond obstacle
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
            print(f"vortex : pas {step}/{total}")

    # Vorticité signée -> [0,1] par une échelle GLOBALE robuste (q99) : la même
    # pour toutes les frames, sinon la normalisation détruirait la dynamique.
    s = np.quantile(np.abs(frames), 0.99)
    frames = np.clip(frames / (2 * max(s, 1e-9)) + 0.5, 0.0, 1.0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


# ---------------------------------------------------------------- forecast

def forecast_series(model, series: np.ndarray, horizon: int,
                    device, chunk: int, tta_flip: bool) -> np.ndarray:
    """[N, ctx] -> fan [N, horizon, 9]. Le cœur commun aux deux modes."""
    n = series.shape[0]
    fans = np.empty((n, horizon, 9), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = torch.from_numpy(series[i:i + chunk, :, None]).float().to(device)
            q = model.forecast(x, n=horizon)["quantiles_denorm"]
            if tta_flip:
                # q_tau(x) <- (q_tau(x) - q_{1-tau}(-x)) / 2  (E19b, officiel)
                q_neg = model.forecast(-x, n=horizon)["quantiles_denorm"]
                q = 0.5 * (q - torch.flip(q_neg, dims=[-1]))
            fans[i:i + chunk] = q.cpu().numpy()
    return fans


def forecast_pixels(model, video: np.ndarray, ctx_len: int, horizon: int,
                    device, chunk: int, tta_flip: bool):
    """Mode pixel : chaque pixel = une série. -> (med, q10, q90, mean) [h,H,W]."""
    t_all, h, w = video.shape
    assert t_all >= ctx_len + horizon, "vidéo trop courte pour contexte+horizon"
    ctx = video[t_all - horizon - ctx_len: t_all - horizon]      # [ctx, H, W]
    q = forecast_series(model, ctx.reshape(ctx_len, h * w).T,
                        horizon, device, chunk, tta_flip)         # [P, h, 9]
    shape = (horizon, h, w)
    # Moyenne du fan ≈ E[x] : pour un pixel quasi binaire c'est la probabilité
    # de présence — la « balle fantôme », le pane de com.
    return (q[..., 4].T.reshape(shape), q[..., 0].T.reshape(shape),
            q[..., 8].T.reshape(shape), q.mean(-1).T.reshape(shape))


def forecast_pca(model, video: np.ndarray, ctx_len: int, horizon: int,
                 device, chunk: int, tta_flip: bool,
                 n_modes: int, mc_samples: int, seed: int = 0):
    """Mode POD/PCA : forecaster les COEFFICIENTS MODAUX, pas les pixels.

    La scène a peu de degrés de liberté (le pendule : ~1, l'allée de vortex :
    une onde progressive, ~2 modes en quadrature) ; pixel par pixel, on pose
    1600 questions intermittentes là où la scène n'en pose que k. PCA sur les
    frames du CONTEXTE SEUL (les modes ne voient jamais le futur — zéro
    fuite) : vidéo ≈ mu + Σ a_j(t)·mode_j. Chaque a_j est une série LISSE et
    périodique — le régime nominal du modèle — et on forecast k séries au
    lieu de H·W. C'est la philosophie du papier appliquée à la démo :
    prédire dans un espace de représentation, décoder pour l'affichage.

    Incertitude pixel par Monte-Carlo : un niveau de quantile tiré PAR MODE
    (indépendance inter-modes — la PCA décorrèle sur le contexte, hypothèse
    déclarée), la trajectoire entière de ce niveau étant gardée (cohérence
    temporelle intra-mode, même logique que le rollout comonotone du fan).
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
    print(f"PCA : {k} modes, variance du contexte expliquée {evr:.1%}")

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


# ---------------------------------------------------------------- rendu

PANE_LABELS = ("ground truth", "TimeJEPA median", "TimeJEPA mean",
               "uncertainty q90-q10")


def to_gif(path: Path, truth: np.ndarray, med: np.ndarray, mean: np.ndarray,
           width: np.ndarray, ctx_tail: np.ndarray, upscale: int, fps: int):
    """GIF en GRILLE 2x2 étiquetée (format post) :
        [ vérité | médiane ]      contexte d'abord, puis horizon
        [ moyenne | incertitude ] (liseré rouge en haut = on forecast).
    Bug corrigé 2026-08-31 : la v1 en bande n'assemblait que 3 panes sur 4."""
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
        if t < n_tail:                                   # phase contexte : la
            g = np.clip(ctx_tail[t], 0, 1)               # même vidéo partout,
            panes = [g, g, g, np.zeros_like(g)]          # incertitude nulle
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
        if caption_rows is None:                         # bandeaux rendus 1 fois
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
        if t >= n_tail:                                  # cadre rouge : on forecast
            d = ImageDraw.Draw(full)
            d.rectangle([0, 0, grid_w - 1, grid_h - 1],
                        outline=(255, 60, 60), width=2)
        frames.append(full)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=1000 // fps, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", choices=["pendulum", "vortex"], default="pendulum")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="lotsa_tiny_v3_eval")
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024, help="séries par forward")
    ap.add_argument("--mode", choices=["pixel", "pca"], default="pixel",
                    help="pixel = 1 série par pixel ; pca = forecast des "
                         "coefficients modaux (POD), la démo « prédire dans "
                         "une représentation »")
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--mc-samples", type=int, default=128,
                    help="échantillons MC pour l'incertitude pixel (mode pca)")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--tail", type=int, default=96,
                    help="frames de contexte montrées avant le liseré rouge")
    ap.add_argument("--out", default="evaluation/video_forecast")
    args = ap.parse_args()

    n_frames = args.context + args.horizon
    if args.scene == "pendulum":
        video = scene_pendulum(n_frames, args.height, args.width)
    else:
        # v2 dans le nom : la calibration du solveur fait partie de l'identité
        # du cache (la v1 non périodique a la même forme de tableau).
        cache = Path(args.out) / f"vortex_v2_{args.height}x{args.width}_{n_frames}.npy"
        video = scene_vortex(n_frames, args.height, args.width, cache)

    # Garde-fou (mesuré 2026-08-31, une soirée perdue) : un checkpoint de
    # PRETRAIN porte un décodeur présent mais JAMAIS entraîné (le JEPA ne lui
    # donne aucun gradient), et le contrat P3.2 ne couvre que le cœur — le
    # chargement passe, les sorties sont du bruit plausible (corr 0.00 sur une
    # sinusoïde nue, contre 1.00 pour le finetuné). Discriminant fiable : les
    # hyper_parameters Lightning du module d'entraînement.
    hp = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False).get("hyper_parameters", {})
    if "finetune_mode" not in hp and "contextualized_targets" in hp:
        raise SystemExit(
            f"REFUS : {args.checkpoint} est un checkpoint de PRETRAIN "
            "(hyper_parameters JEPA, pas de finetune_mode) — sa tête quantile "
            "est à l'initialisation. La démo vidéo exige un checkpoint "
            "FINETUNÉ (répertoires *_zs/pretrain_False ou champions/).")

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
    print(f"MAE modèle      : {mae_model:.4f}")
    print(f"MAE persistance : {mae_persist:.4f}  (ratio {mae_model / mae_persist:.3f})")
    print(f"couverture 80%  : {cov80:.3f} (nominal 0.800)")

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
    print(f"Sorties : {out}/{tag}.gif")


if __name__ == "__main__":
    main()
