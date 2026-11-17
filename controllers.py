"""Classical baseline controller (PD + CW feed-forward) and residual-RL wrapper."""
import numpy as np
import gymnasium as gym
from osam_env import q2R, unit


def pd_policy(env, vmax=1.5, a_dec_frac=0.35, kv=0.5, kp_att=0.012, kd_att=0.20):
    """Guidance: speed profile toward goal. Control: velocity loop with CW feed-forward,
    pointing PD on boresight error. Uses (noisy) on-board measurements only."""
    r, v, q, w = env.get_measurement()
    m = env.m_dry + env.fuel
    R = q2R(q)
    a_max = env.Fmax / m
    e = env.goal - r
    d = np.linalg.norm(e)
    v_des = unit(e) * min(vmax, np.sqrt(2*a_dec_frac*a_max*d), 0.15*d)
    a_cmd = kv * (v_des - v) - env._a_cw(r, v)
    F = m * R.T @ a_cmd
    # attitude: point +X body at chief
    los_b = R.T @ unit(-r)
    err = np.cross(np.array([1.0, 0, 0]), los_b)
    w_rel = w - R.T @ np.array([0, 0, env.n])
    tau = env.I @ (kp_att * err - kd_att * w_rel) + np.cross(w, env.I @ w)
    return np.clip(np.concatenate([F / env.Fmax, tau / env.Tmax]), -1, 1).astype(np.float32)


class CurriculumWrapper(gym.Wrapper):
    """Exposes set_curriculum_level through the wrapper chain (gymnasium>=1.0 does not forward attributes)."""

    def set_curriculum_level(self, level):
        self.env.unwrapped.set_curriculum_level(level)


class ResidualEnv(CurriculumWrapper):
    """action_applied = clip(PD(state) + res_scale * policy_action).
    With policy_action = 0 the agent equals the PD baseline, so PPO starts from a
    working controller and only has to learn improvements (fuel, robustness)."""

    def __init__(self, env, res_scale=0.3):
        super().__init__(env)
        self.res_scale = res_scale

    def step(self, action):
        base = pd_policy(self.env.unwrapped)
        a = np.clip(base + self.res_scale * np.asarray(action), -1, 1)
        return self.env.step(a)