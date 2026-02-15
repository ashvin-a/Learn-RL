import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load Env with Human Render Mode
env = gym.make("Walker2d-v5", render_mode="human")

# 2. Load Model
model = PPO.load("walker2d_policy_1")

# 3. Enjoy Loop
obs, _ = env.reset()
while True:
    # deterministic=True makes the robot use its BEST action, not a random one
    action, _ = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()
