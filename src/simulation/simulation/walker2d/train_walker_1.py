import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import time


def train_first_iteration():
    # Configuration
    env_id = "Walker2d-v5"
    num_envs = 12  # Number of parallel environments (adjust based on CPU cores)
    total_timesteps = 4_000_000  # 4 Million steps (needed for a good walker)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    # Create Vectorized Environment
    # We use SubprocVecEnv so each env runs on a different CPU core.
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

    # Initialize the Agent
    # We use Proximal Policy Optimization(PPO) algorithm here. Earlier 
    # we used simple Q-learning in Cartpole Agent.
    # I've attached the research paper to PPO here:
    # https://arxiv.org/abs/1707.06347
    # MlpPolicy: Standard Dense Neural Network
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        batch_size=2048,  # Larger batch size for GPU efficiency
        n_steps=2048 // 12,  # Divided by 12 because my machine has 12 cores.
    )

    # Train
    start_time = time.time()
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()

    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes")

    # Save
    model.save("walker2d_policy_1")
    print("Model saved as 'walker2d_policy.zip'")

    # Close the parallel environments
    env.close()


if __name__ == "__main__":
    train_first_iteration()
