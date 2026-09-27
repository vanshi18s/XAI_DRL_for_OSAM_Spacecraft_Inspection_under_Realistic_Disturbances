import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from satellite_env import Satellite6DOFProximityEnv


def evaluate_agent(test_level: int = 1, episodes: int = 3):
    print(f"\n==========================================")
    print(f"  EVALUATING ON CURRICULUM LEVEL {test_level}")
    print(f"==========================================")

    # 1. Instantiate test environment
    base_env = Satellite6DOFProximityEnv(config={"enable_noise": False})
    base_env.set_curriculum_level(test_level)
    monitored_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    # 2. Load saved normalization statistics and model weights
    try:
        vec_env = VecNormalize.load("vec_normalize_stats.pkl", vec_env)
        vec_env.training = False       # Freeze running statistics
        vec_env.norm_reward = False   # Preserve raw evaluation rewards
    except FileNotFoundError:
        print("[Warning] vec_normalize_stats.pkl not found. Running unnormalized.")

    model = PPO.load("ppo_6dof_satellite_docking")

    successes = 0
    for ep in range(1, episodes + 1):
        obs = vec_env.reset()
        print(f"\n--- Episode {ep} ---")
        ep_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += rewards[0]
            base_env.render()

            if dones[0]:
                info = infos[0]
                is_docked = info["is_docked"]
                if is_docked:
                    successes += 1
                print(f"\nEpisode {ep} Result:")
                print(f"  Docked:       {is_docked}")
                print(f"  Final Dist:   {info['pos_error']:.3f} m")
                print(f"  Attitude Err: {np.rad2deg(info['att_error_rad']):.2f}°")
                print(f"  Fuel Mass:    {info['fuel_mass_kg']:.2f} kg")
                print(f"  Total Reward: {ep_reward:.2f}")
                break

    print(f"\nFinal Score for Level {test_level}: {successes}/{episodes} Dockings Successful ({(successes/episodes)*100:.1f}%)")


if __name__ == "__main__":
    # Test across all distance stages
    evaluate_agent(test_level=1, episodes=2)
    evaluate_agent(test_level=2, episodes=2)