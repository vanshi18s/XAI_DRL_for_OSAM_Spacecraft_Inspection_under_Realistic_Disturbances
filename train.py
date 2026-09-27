import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from satellite_env import Satellite6DOFProximityEnv


class AutoCurriculumCallback(BaseCallback):
    """
    Monitors docking success rate over rolling windows.
    Advances curriculum level when the success rate exceeds 70%.
    """
    def __init__(self, check_freq: int = 10_000, window_size: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window_size = window_size
        self.recent_successes = []
        self.current_level = 1

    def _on_step(self) -> bool:
        # Collect episode info from monitors
        for info in self.locals.get("infos", []):
            if "is_docked" in info:
                self.recent_successes.append(1 if info["is_docked"] else 0)
                if len(self.recent_successes) > self.window_size:
                    self.recent_successes.pop(0)

        # Evaluate and advance level
        if self.n_calls % self.check_freq == 0 and len(self.recent_successes) >= self.window_size:
            success_rate = np.mean(self.recent_successes)
            if self.verbose:
                print(f"\n[Step {self.num_timesteps}] Level {self.current_level} | Success Rate: {success_rate * 100:.1f}%")

            if success_rate >= 0.70 and self.current_level < 3:
                self.current_level += 1
                print(f"\n>>> ADVANCING TO CURRICULUM LEVEL {self.current_level} <<<\n")
                # Broadcast level update to all sub-environments
                self.training_env.env_method("set_curriculum_level", self.current_level)
                self.recent_successes.clear()

        return True


def make_env(rank: int, seed: int = 0):
    def _init():
        env = Satellite6DOFProximityEnv(config={"enable_noise": False})
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    num_cpu = 4  # Adjust based on available CPU cores
    log_dir = "./satellite_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Creating {num_cpu} vectorized environments...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # VecNormalize keeps network inputs within standard normal distribution
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Policy setup tailored for satellite GNC
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    callback = AutoCurriculumCallback(check_freq=5_000, window_size=40)

    print("\nStarting Curriculum DRL Training (500,000 timesteps)...")
    model.learn(total_timesteps=500_000, callback=callback, progress_bar=True)

    # Save model and observation normalization statistics
    model.save("ppo_6dof_satellite_docking")
    vec_env.save("vec_normalize_stats.pkl")
    print("\nModel and normalization statistics saved successfully.")