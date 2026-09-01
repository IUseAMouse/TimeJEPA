# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round
# (G13-T1/T2 control-as-EBM probe, E20c); kept per the no-delete policy.
"""
G13-T1/T2 - the "control in EBM mode" hypothesis, tested WITHOUT a_film or GPU.

    python scripts/control_ebm_probe.py \\
        --checkpoint checkpoints/timejepa_tiny_lotsa_mix/last.ckpt \\
        --model-config lotsa_tiny_mix_eval --standalone-targets

Principle: we OWN the simulator (linear thermostat, known dynamics), so every
claim of the judge and every plan is verifiable against the true dynamics -
which neither LOTSA nor GIFT allows.

Simulator (observable univariate, hidden action as in LOTSA):
    x_{t+1} = x_t + alpha*(amb_t - x_t) + beta*u_t + sigma*eps_t
    amb_t   = 24-step sinusoid + level; u_t in [0, 1] (heating)
    history generated under a thermostat POLICY (smoothed bang-bang) => u
    correlated with the state = deliberate observational confounding, the G13
    wall in miniature.

T1 - does the judge know the dynamics?
    h=256 candidates: coherent (new seeds + varied setpoints) against three
    violation families - time reversal, initial-state jump, INVERTED action
    response (beta -> -beta). Energy AUC per family. Predictions written
    beforehand: jump/reversal separated (AUC > 0.7); the action sign is the
    hard case - if it is NOT separated, that directly measures the
    confounding wall (the observational judge does not know the effect of u,
    only the morphologies).

T2 - planning by backprop, ground truth in hand.
    Goal: keep x inside a band [c_lo, c_hi] over h steps. Command u in
    [0,1]^h optimized by gradient THROUGH the differentiable simulator
    (noise-free), under two objectives:
        (a) cost alone;
        (b) cost + lambda_E * Energy(ctx, x(u)) - the uncertainty
            regularizer a la Henaff/Canziani/LeCun (MPUR).
    VERDICT BY EXECUTION: 200 noisy rollouts under the found u -> realized
    violation rate + "optimism gap" (noise-free planned cost vs realized
    cost). Prediction: (a) over-promises (smooth-simulator Goodhart), (b)
    closes the gap; if energy does not help, negative result recorded.

Read-only, CPU (~2-4 min). Console output + JSON evaluation/control_ebm/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402

# ------------------------------------------------------------------ simulator
ALPHA, BETA, SIGMA = 0.10, 0.50, 0.05
PERIOD, AMB_AMP, AMB_LVL = 24, 1.0, 0.0
CTX_LEN, H = 1024, 256


def ambient(t0: int, n: int) -> torch.Tensor:
    t = torch.arange(t0, t0 + n, dtype=torch.float32)
    return AMB_LVL + AMB_AMP * torch.sin(2 * np.pi * t / PERIOD)


def simulate(x0: float, t0: int, u: torch.Tensor, noise: torch.Tensor | None,
             beta: float = BETA) -> torch.Tensor:
    """Rollout, differentiable if u requires it. u, noise: [n]. Returns x [n]."""
    amb = ambient(t0, len(u))
    xs, x = [], torch.as_tensor(float(x0))
    for i in range(len(u)):
        eps = noise[i] if noise is not None else 0.0
        x = x + ALPHA * (amb[i] - x) + beta * u[i] + SIGMA * eps
        xs.append(x)
    return torch.stack(xs)


def thermostat_policy(x: torch.Tensor, setpoint: float, gain: float = 2.0) -> torch.Tensor:
    return torch.clamp(gain * (setpoint - x), 0.0, 1.0)


def gen_history(rng: torch.Generator, n: int = CTX_LEN, setpoints=(1.0, 2.0)):
    """History under a thermostat policy, alternating day/night setpoints."""
    noise = torch.randn(n, generator=rng)
    xs, us, x = [], [], torch.tensor(0.5)
    for t in range(n):
        sp = setpoints[(t // PERIOD) % len(setpoints)]
        u = thermostat_policy(x, sp)
        amb = AMB_LVL + AMB_AMP * float(np.sin(2 * np.pi * t / PERIOD))
        x = x + ALPHA * (amb - x) + BETA * u + SIGMA * noise[t]
        xs.append(float(x)); us.append(float(u))
    return np.array(xs, dtype=np.float32), np.array(us, dtype=np.float32)


# -------------------------------------------------------------- judge energy
def energy(model, ctx: torch.Tensor, cands: torch.Tensor, standalone: bool,
           grad: bool = False) -> torch.Tensor:
    """E = 1 - cos(z_pred, enc(cand)) averaged over patches. ctx [L], cands [Nc, h].
    Differentiable in `cands` if grad=True (probe_energy path, without no_grad)."""
    n_tgt = (cands.shape[1] - model.patching.patch_size) // model.patching.stride + 1
    x_ctx = ctx.reshape(1, -1, 1)
    xc = cands.unsqueeze(-1)
    if model.robust_scaler is not None:
        model.robust_scaler.fit(x_ctx)
        x_ctx = model.robust_scaler.transform(x_ctx)
        xc = model.robust_scaler.transform(xc)
    ctx_n = model.revin(x_ctx, mode='norm') if model.revin is not None else x_ctx
    xc = (xc - model.revin.mean) / model.revin.std if model.revin is not None else xc
    with torch.set_grad_enabled(grad):
        ctx_emb = model.online_encoder(model.patching(ctx_n))
        z_pred = model.predictor.forward_simple(
            context_embeddings=ctx_emb, num_targets=model.num_target_patches,
            w=(torch.ones(1) if hasattr(model.predictor, 'w_film') else None),
        )[:, :n_tgt, :]
        if standalone:
            z_cand = model.online_encoder(model.patching(xc))
        else:
            full = torch.cat([ctx_n.expand(xc.shape[0], -1, -1), xc], dim=1)
            z_cand = model.online_encoder(model.patching(full))[:, -n_tgt:, :]
        cos = torch.nn.functional.cosine_similarity(
            z_cand.flatten(1), z_pred.expand_as(z_cand).flatten(1), dim=1)
    return 1.0 - cos                                                  # [Nc]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(E_violation > E_coherent) - 1.0 = perfect separation."""
    return float((pos[None, :] > neg[:, None]).mean())


# ------------------------------------------------------------------------ T1
def run_t1(model, standalone, n_inst, n_cand, seed):
    g = torch.Generator().manual_seed(seed)
    fams = {"coherent": [], "reversal": [], "state_jump": [], "beta_flip": []}
    for _ in range(n_inst):
        hist, _ = gen_history(g)
        x0, t0 = float(hist[-1]), CTX_LEN
        cohs, revs, jmps, flips = [], [], [], []
        for _ in range(n_cand):
            sp = float(torch.empty(1).uniform_(0.5, 2.5, generator=g))
            noise = torch.randn(H, generator=g)
            u_traj, xs, x = [], [], torch.tensor(x0)
            for i in range(H):
                u = thermostat_policy(x, sp)
                amb = AMB_LVL + AMB_AMP * float(np.sin(2 * np.pi * (t0 + i) / PERIOD))
                x = x + ALPHA * (amb - x) + BETA * u + SIGMA * float(noise[i])
                xs.append(float(x)); u_traj.append(u)
            coh = torch.tensor(xs)
            cohs.append(coh)
            revs.append(torch.flip(coh, [0]))
            jump = float(torch.empty(1).uniform_(1.5, 2.5, generator=g)) * \
                (1 if torch.rand(1, generator=g) > 0.5 else -1)
            jmps.append(simulate(x0 + jump, t0, torch.stack(u_traj), noise))
            # same policy, INVERTED action response (heating cools)
            xs_f, x = [], torch.tensor(x0)
            for i in range(H):
                u = thermostat_policy(x, sp)
                amb = AMB_LVL + AMB_AMP * float(np.sin(2 * np.pi * (t0 + i) / PERIOD))
                x = x + ALPHA * (amb - x) - BETA * u + SIGMA * float(noise[i])
                xs_f.append(float(x))
            flips.append(torch.tensor(xs_f))
        ctx = torch.from_numpy(hist)
        for name, cs in (("coherent", cohs), ("reversal", revs),
                         ("state_jump", jmps), ("beta_flip", flips)):
            fams[name].append(energy(model, ctx, torch.stack(cs).float(),
                                     standalone).numpy())
    out = {}
    coh = np.concatenate(fams["coherent"])
    print("\nT1 - energy per candidate family "
          f"(coherent: {coh.mean():.4f} +- {coh.std():.4f})")
    for name in ("reversal", "state_jump", "beta_flip"):
        e = np.concatenate(fams[name])
        a = auc(e, coh)
        out[name] = {"mean_energy": float(e.mean()), "auc_vs_coherent": a}
        print(f"  {name:11s} E {e.mean():.4f} +- {e.std():.4f}  AUC vs coherent {a:.3f}")
    out["coherent_mean_energy"] = float(coh.mean())
    return out


# ------------------------------------------------------------------------ T2
def plan(model, ctx, x0, t0, standalone, lambda_e, c_lo, c_hi,
         steps=120, lr=0.08, lambda_u=0.02, plan_beta=BETA):
    """plan_beta != BETA = T2b: the PLANNER believes a wrong dynamics
    (controlled model error); execution (realized) always uses the true one.
    That is the real MPUR setting - the regularizer only has work to do when
    the planning model is wrong."""
    raw = torch.zeros(H, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        u = torch.sigmoid(raw)
        x = simulate(x0, t0, u, noise=None, beta=plan_beta)
        cost = (torch.relu(x - c_hi) + torch.relu(c_lo - x)).mean() \
            + lambda_u * (u ** 2).mean()
        # The energy lives at ~0.7-1.0, the cost at ~0.005: regularize on the
        # energy EXCESS above the coherent-futures level (E20c: raw lambda_E
        # at 0.5 drowned the cost at 99% - a rigged trial). lambda_e stays
        # the weight of that excess.
        if lambda_e > 0:
            e = energy(model, ctx, x.unsqueeze(0), standalone, grad=True)[0]
            loss = cost + lambda_e * torch.relu(e - plan.e_ref)
        else:
            loss = cost
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u = torch.sigmoid(raw)
        x_plan = simulate(x0, t0, u, noise=None, beta=plan_beta)
        cost_plan = float((torch.relu(x_plan - c_hi) + torch.relu(c_lo - x_plan)).mean())
    return u.detach(), cost_plan


plan.e_ref = 0.80   # "coherent future" energy level (T1, mix checkpoint)


def realized(u, x0, t0, c_lo, c_hi, n_mc, g):
    costs, viols = [], []
    for _ in range(n_mc):
        x = simulate(x0, t0, u, noise=torch.randn(H, generator=g))
        costs.append(float((torch.relu(x - c_hi) + torch.relu(c_lo - x)).mean()))
        viols.append(float(((x < c_lo) | (x > c_hi)).float().mean()))
    return float(np.mean(costs)), float(np.mean(viols))


def run_t2(model, standalone, n_inst, lambda_e, n_mc, seed, plan_beta=BETA):
    g = torch.Generator().manual_seed(seed + 1)
    rows = []
    print(f"\nT2 - planning by backprop, target band [1.2, 1.8], lambda_E={lambda_e}, "
          f"planner beta {plan_beta} (true: {BETA})")
    for k in range(n_inst):
        hist, _ = gen_history(g)
        ctx, x0, t0 = torch.from_numpy(hist), float(hist[-1]), CTX_LEN
        row = {}
        for name, le in (("cost_only", 0.0), ("cost+energy", lambda_e)):
            u, cost_plan = plan(model, ctx, x0, t0, standalone, le, 1.2, 1.8,
                                plan_beta=plan_beta)
            cost_real, viol = realized(u, x0, t0, 1.2, 1.8, n_mc, g)
            row[name] = {"cost_plan": cost_plan, "cost_real": cost_real,
                         "optimism_gap": cost_real - cost_plan,
                         "violation_rate": viol, "u_mean": float(u.mean())}
        rows.append(row)
        a, b = row["cost_only"], row["cost+energy"]
        print(f"  inst{k}  cost_only: plan {a['cost_plan']:.4f} real {a['cost_real']:.4f} "
              f"(gap {a['optimism_gap']:+.4f}, viol {a['violation_rate']:.1%}) | "
              f"+energy: plan {b['cost_plan']:.4f} real {b['cost_real']:.4f} "
              f"(gap {b['optimism_gap']:+.4f}, viol {b['violation_rate']:.1%})")
    agg = {}
    for name in ("cost_only", "cost+energy"):
        agg[name] = {k: float(np.mean([r[name][k] for r in rows]))
                     for k in rows[0][name]}
    print(f"  => mean optimism gap: cost_only {agg['cost_only']['optimism_gap']:+.4f} "
          f"| +energy {agg['cost+energy']['optimism_gap']:+.4f} ; "
          f"realized violation: {agg['cost_only']['violation_rate']:.1%} "
          f"| {agg['cost+energy']['violation_rate']:.1%}")
    return {"instances": rows, "aggregate": agg}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="lotsa_tiny_mix_eval")
    ap.add_argument("--standalone-targets", action="store_true")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=16, help="per family (T1)")
    ap.add_argument("--lambda-e", type=float, default=0.5,
                    help="weight of the energy EXCESS above e_ref (rebalanced E20c)")
    ap.add_argument("--plan-beta", type=float, default=BETA,
                    help="T2b: beta BELIEVED by the planner (true=0.5) - controlled model error")
    ap.add_argument("--e-ref", type=float, default=0.80,
                    help="reference energy of coherent futures (T1 of the probed checkpoint)")
    ap.add_argument("--mc", type=int, default=200, help="verification rollouts (T2)")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)
    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    model.eval()

    results = {"checkpoint": args.checkpoint, "standalone": args.standalone_targets,
               "sim": {"alpha": ALPHA, "beta": BETA, "sigma": SIGMA}}
    results["t1"] = run_t1(model, args.standalone_targets, args.instances,
                           args.candidates, args.seed)
    if not args.skip_t2:
        plan.e_ref = args.e_ref
        results["t2"] = run_t2(model, args.standalone_targets,
                               max(3, args.instances // 2),
                               args.lambda_e, args.mc, args.seed,
                               plan_beta=args.plan_beta)

    out = Path("evaluation/control_ebm")
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem + ("_standalone" if args.standalone_targets else "")
    (out / f"{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nJSON: {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
