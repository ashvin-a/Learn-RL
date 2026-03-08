import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import time

def train_sac_vectorized():
    # Configuration
    env_id = "Walker2d-v5"
    
    # 1. Reduced from 12 to 4 to save RAM for the SAC Replay Buffer
    num_envs = 12  
    
    # 2. SAC is much more sample-efficient than PPO. 
    # 3 Million steps is usually plenty for SAC to master Walker2D.
    total_timesteps = 1_000_000  

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    # Create Vectorized Environment on multiple CPU cores
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

    # Initialize the SAC Agent
    print("Building the SAC Neural Network...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        
        # --- SAC SPECIFIC HYPERPARAMETERS ---
        batch_size=256,
        buffer_size=1_000_000,    # Stores 1 million past steps in memory
        learning_starts=10_000,   # Takes 10k random steps before training starts
        ent_coef='auto',          # Automatically tunes the "curiosity" 
        
        # CRITICAL for vectorized SAC: Update the network exactly as many 
        # times as the number of parallel environments we just stepped.
        train_freq=1,
        gradient_steps=num_envs,  
    )

    # Train
    start_time = time.time()
    print("Starting SAC training loop...")
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()

    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes")

    # Save
    model.save("walker2d_sac_vectorized")
    print("Model saved as 'walker2d_sac_vectorized.zip'")

    # Close the parallel environments
    env.close()

if __name__ == "__main__":
    train_sac_vectorized()