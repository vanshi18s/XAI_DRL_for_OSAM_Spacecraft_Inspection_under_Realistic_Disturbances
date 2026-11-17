"""Adversarial RTA test: policies that deliberately try to violate constraints. Run: python test_rta.py"""
import numpy as np
from osam_env import OSAMInspectionEnv, q2R, unit

def trans_kamikaze(env):   # thrust straight at chief, attitude held by damping
    r, v, q, w = env.get_measurement(); R = q2R(q)
    f = R.T @ unit(-r); f = np.clip(f / (np.abs(f).max() + 1e-9), -1, 1)
    w_rel = w - R.T @ np.array([0, 0, env.n])
    return np.concatenate([f, np.clip(-2*env.I @ w_rel/env.Tmax*1.0, -1, 1)]).astype(np.float32)

def sun_kamikaze(env):     # torque boresight toward the sun, no thrust
    r, v, q, w = env.get_measurement(); R = q2R(q)
    s_b = R.T @ env._sun_lvlh()
    return np.concatenate([np.zeros(3), np.sign(np.cross([1.0, 0, 0], s_b))]).astype(np.float32)

def rand(env): return env.np_random.uniform(-1, 1, 6).astype(np.float32)

def run(pol, rta, n=40, inward=0.0):
    kv = sv = 0; mr = 1e9; ms = 1e9
    for lvl in (1, 2):
        env = OSAMInspectionEnv({"rta": rta, "curriculum_level": lvl, "max_time": 300})
        for ep in range(n // 2):
            env.reset(seed=500 + ep)
            if inward: env.v = env.v + (-unit(env.r)) * inward; env._meas = env._measure()
            while True:
                _, _, te, tr, info = env.step(pol(env))
                if te or tr: break
            kv += info["koz_violation"]; sv += info["sun_violation"]
            mr = min(mr, info["min_range"]); ms = min(ms, info["min_sun_angle_deg"])
    return kv, sv, mr, ms, n

for name, pol, inward in (("ram target (1 m/s inbound)", trans_kamikaze, 1.0), ("point at sun", sun_kamikaze, 0.0),
                          ("random actions", rand, 0.0)):
    for rta in (False, True):
        kv, sv, mr, ms, n = run(pol, rta, inward=inward)
        print(f"{name:26s} RTA={rta!s:5}: KOZ viol {kv:2d}/{n} (min range {mr:6.2f} m; KOZ=4) | "
              f"sun viol {sv:2d}/{n} (min angle {ms:5.1f} deg; limit 25)")
