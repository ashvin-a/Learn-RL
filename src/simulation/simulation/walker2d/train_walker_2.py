import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import torch
import time


# Building a Custom wrapper for controlling the x-velocity of the agent
class TargetVelocityWrapper(gym.RewardWrapper):
    def __init__(self, env, target_vel=1.0):
        super().__init__(env)
        self.target_vel = target_vel

    def reward(self, reward):
        # Extract velocity from MuJoCo data
        x_vel = self.env.unwrapped.data.qvel[0]

        # Calculate penalty for deviation
        velocity_error = abs(x_vel - self.target_vel)

        # New Reward: Survival (+1) minus Error (*2)
        return 1.0 - (2.0 * velocity_error)


def make_env():
    env = gym.make("Walker2d-v5")
    return TargetVelocityWrapper(env=env, target_vel=1.5)


def train_second_iteration():
    # Configuration
    env_id = "Walker2d-v5"
    total_timesteps = 2_000_000  # 4 Million steps (needed for a good walker)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    # Create Vectorized Environment
    # Here, we try to stack the last 4 framees for better learning of where
    # the legs are.
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    # Initialize the Agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        batch_size=2048,  # Larger batch size for GPU efficiency
        n_steps=2048,
    )

    # Train
    start_time = time.time()
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()

    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes")

    # Save
    model.save("walker2d_policy_2")
    print("Model saved as 'walker2d_policy_2.zip'")

    # Close the parallel environments
    env.close()


if __name__ == "__main__":
    train_second_iteration()
