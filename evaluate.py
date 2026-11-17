"""
Monte-Carlo evaluation: PD baseline vs trained PPO, RTA on/off, all curriculum levels.
Eval seeds (10000+) never overlap training seeds.   python evaluate.py --episodes 100
"""
import argparse, os
import numpy as np
import pandas as pd
from osam_env import OSAMInspectionEnv
from controllers import ResidualEnv


def load_ppo(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def run_method(name, make_wrapped, act_fn, level, episodes, rta):
    rows = []
    for ep in range(episodes):
        base = OSAMInspectionEnv({"curriculum_level": level, "rta": rta, "realistic": True})
        env = make_wrapped(base)
        obs, _ = env.reset(seed=10_000 + 97 * level + ep)
        while True:
            obs, r, te, tr, info = env.step(act_fn(obs))
            if te or tr:
                break
        rows.append(dict(method=name, level=level, rta=rta, episode=ep, success=info["is_success"],
                         time_s=info["sim_time"], delta_v=info["delta_v"], fuel_kg=info["fuel_used_kg"],
                         final_pos_err=info["pos_err"], final_point_deg=info["point_err_deg"],
                         koz_violation=info["koz_violation"], sun_violation=info["sun_violation"],
                         min_range=info["min_range"], rta_trans_frac=info["rta_trans_frac"],
                         rta_att_frac=info["rta_att_frac"], duty=info["duty_cycle"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--residual_model", default="models/ppo_residual.zip")
    ap.add_argument("--pure_model", default="models/ppo_pure.zip")
    a = ap.parse_args()

    zeros = lambda obs: np.zeros(6, dtype=np.float32)
    methods = [("PD baseline", lambda e: ResidualEnv(e), zeros)]
    if os.path.exists(a.residual_model):
        m = load_ppo(a.residual_model)
        methods.append(("PPO residual", lambda e: ResidualEnv(e), lambda o: m.predict(o, deterministic=True)[0]))
    if os.path.exists(a.pure_model):
        m2 = load_ppo(a.pure_model)
        methods.append(("PPO pure", lambda e: e, lambda o: m2.predict(o, deterministic=True)[0]))
    if len(methods) == 1:
        print("[info] no trained PPO model found in models/ - evaluating the PD baseline only. Run train.py first.")

    rows = []
    for name, mk, act in methods:
        for level in (1, 2, 3):
            for rta in (True, False):
                rows += run_method(name, mk, act, level, a.episodes, rta)
                print(f"done: {name} L{level} RTA={rta}")
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/mc_episodes.csv", index=False)

    ok = df[df.success]
    summ = df.groupby(["method", "level", "rta"]).agg(
        success_pct=("success", lambda x: 100 * x.mean()), koz_viol=("koz_violation", "sum"),
        sun_viol=("sun_violation", "sum"), min_range_m=("min_range", "min"),
        rta_trans_pct=("rta_trans_frac", lambda x: 100 * x.mean()), duty=("duty", "mean")).round(2)
    dv = ok.groupby(["method", "level", "rta"]).agg(dv_mean=("delta_v", "mean"), dv_std=("delta_v", "std"),
                                                    time_mean=("time_s", "mean")).round(2)
    summ = summ.join(dv)
    summ.to_csv("results/mc_summary.csv")
    print("\n", summ.to_string())

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    s = df[df.rta].groupby(["level", "method"]).success.mean().unstack() * 100
    s.plot.bar(ax=ax[0]); ax[0].set_ylabel("success %"); ax[0].set_title("Docking/inspection success (RTA on)")
    d = df[df.rta & df.success].pivot_table(index="level", columns="method", values="delta_v")
    d.plot.bar(ax=ax[1]); ax[1].set_ylabel("Delta-V [m/s]"); ax[1].set_title("Propellant use (successful runs)")
    v = df[~df.rta].groupby(["level", "method"]).apply(lambda g: (g.koz_violation | g.sun_violation).mean() * 100).unstack()
    v.plot.bar(ax=ax[2]); ax[2].set_ylabel("episodes with a violation %"); ax[2].set_title("Constraint violations (RTA OFF)")
    plt.tight_layout(); plt.savefig("results/mc_summary.png", dpi=150)
    print("saved results/mc_episodes.csv, mc_summary.csv, mc_summary.png")