import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from train_walker_2 import TargetVelocityWrapper

def make_env():
    env = gym.make("Walker2d-v5", render_mode="human")
    return TargetVelocityWrapper(env=env, target_vel=1.5)

# 1. Load Env with Human Render Mode
env = DummyVecEnv([make_env])    
# We stack 4 frames, exactly like in training
env = VecFrameStack(env, n_stack=4)

model = PPO.load("walker2d_policy_2")

# 3. Enjoy Loop
obs, _ = env.reset()
while True:
    # deterministic=True makes the robot use its BEST action, not a random one
    action, _ = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, _ = env.step(action)
    
    if done[0]:
        obs, _ = env.reset()
