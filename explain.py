"""
Explainable-AI module: which observations drive each thruster/torque command?
Uses SHAP (KernelExplainer) if installed (pip install shap), otherwise permutation importance.
  python explain.py --model models/ppo_residual.zip --mode residual
Outputs results/xai_importance.png and results/xai_importance.csv
"""
import argparse, os
import numpy as np
import pandas as pd
from osam_env import OSAMInspectionEnv
from controllers import ResidualEnv

ACTS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def collect_states(policy_fn, mode, n=400, seed=777):
    env = OSAMInspectionEnv({"curriculum_level": 2})
    env = ResidualEnv(env) if mode == "residual" else env
    X, o = [], None
    ep = 0
    while len(X) < n:
        o, _ = env.reset(seed=seed + ep); ep += 1
        while True:
            X.append(o.copy())
            o, _, te, tr, _ = env.step(policy_fn(o))
            if te or tr or len(X) >= n:
                break
    return np.array(X), OSAMInspectionEnv().obs_names


def importance(f, X, method="auto", n_explain=100):
    bg = X[np.random.default_rng(0).choice(len(X), 40, replace=False)]
    xs = X[np.random.default_rng(1).choice(len(X), min(n_explain, len(X)), replace=False)]
    if method in ("auto", "shap"):
        try:
            import shap
            sv = np.array(shap.KernelExplainer(f, bg).shap_values(xs, silent=True))
            sv = np.abs(sv.transpose(2, 0, 1) if sv.shape[0] == len(xs) else sv)    # -> (out, samples, feat)
            return sv.mean(axis=1), "SHAP (mean |value|)"
        except ImportError:
            print("[info] shap not installed -> permutation importance")
    base = f(xs); rng = np.random.default_rng(2); imp = np.zeros((base.shape[1], xs.shape[1]))
    for j in range(xs.shape[1]):
        for _ in range(5):
            xp = xs.copy(); xp[:, j] = rng.permutation(xp[:, j])
            imp[:, j] += np.abs(f(xp) - base).mean(axis=0) / 5
    return imp, "permutation importance (mean |change in action|)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_residual.zip")
    ap.add_argument("--mode", choices=["residual", "pure"], default="residual")
    ap.add_argument("--method", default="auto")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    m = PPO.load(a.model, device="cpu")
    f = lambda X: m.predict(np.atleast_2d(X).astype(np.float32), deterministic=True)[0]
    X, names = collect_states(lambda o: f(o)[0], a.mode)
    imp, label = importance(f, X, a.method)
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(imp.T, index=names, columns=ACTS); df.to_csv("results/xai_importance.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7)); im = ax.imshow(df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(6)); ax.set_xticklabels(ACTS); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, label=label); ax.set_title("What drives each thruster/torque command?")
    plt.tight_layout(); plt.savefig("results/xai_importance.png", dpi=150); print(df.round(4).to_string())
