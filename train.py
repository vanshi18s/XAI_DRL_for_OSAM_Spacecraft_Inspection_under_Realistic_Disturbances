"""
PPO training for the OSAM 6-DOF inspection task (Stable-Baselines3).

  python train.py --mode residual --steps 600000     # recommended: PPO learns a correction on top of PD baseline
  python train.py --mode pure     --steps 2000000    # PPO from scratch (harder, needs more steps)

Curriculum (auto): level 1 (25-50 m) -> 2 (50-100 m) -> 3 (100-180 m) when rolling success >= 80%.
The RTA safety filter is ON during training, so the agent learns inside the safe set.
"""
import argparse, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from osam_env import OSAMInspectionEnv
from controllers import ResidualEnv, CurriculumWrapper


def make_env(mode, rank, level):
    def _init():
        env = OSAMInspectionEnv({"curriculum_level": level, "rta": True, "realistic": True})
        return ResidualEnv(env, res_scale=0.15) if mode == "residual" else CurriculumWrapper(env)
    return _init


class AutoCurriculum(BaseCallback):
    def __init__(self, window=100, thresh=0.8, verbose=1):
        super().__init__(verbose)
        self.window, self.thresh, self.level, self.hist = window, thresh, 1, []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.hist.append(float(info.get("is_success", False)))
        self.hist = self.hist[-self.window:]
        if len(self.hist) >= self.window:
            sr = float(np.mean(self.hist))
            self.logger.record("curriculum/success_rate", sr)
            self.logger.record("curriculum/level", self.level)
            if sr >= self.thresh and self.level < 3:
                self.level += 1
                self.training_env.env_method("set_curriculum_level", self.level)
                self.hist = []
                if self.verbose:
                    print(f"\n>>> step {self.num_timesteps}: success {sr:.0%} -> CURRICULUM LEVEL {self.level}\n")
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["residual", "pure"], default="residual")
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--envs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    VecCls = SubprocVecEnv if a.envs > 1 and (os.cpu_count() or 1) > 1 else DummyVecEnv
    venv = VecMonitor(VecCls([make_env(a.mode, i, 1) for i in range(a.envs)]))

    model = PPO(
        "MlpPolicy", venv, seed=a.seed, verbose=1, tensorboard_log="./tb/",
        learning_rate=lambda p: 3e-4 * max(p, 0.1),       # linear decay
        n_steps=max(64, 4096 // a.envs), batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                           log_std_init=-2.0 if a.mode == "residual" else -0.5),
    )
    model.learn(a.steps, callback=AutoCurriculum(), progress_bar=False)
    model.save(f"models/ppo_{a.mode}")
    print(f"saved models/ppo_{a.mode}.zip")