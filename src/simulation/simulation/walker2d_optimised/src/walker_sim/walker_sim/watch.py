import os
import time
from stable_baselines3 import SAC
from walker_sim.walker_env import ROS2WalkerEnv

def main():
    print("Loading ROS 2 Walker2D Environment...")
    env = ROS2WalkerEnv()

    # Locate the saved model we just trained
    model_path = os.path.join(os.getcwd(), "sac_walker2d_v1.zip")
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print("Loading Trained SAC Model...")
    model = SAC.load(model_path)

    print("Starting Inference Loop...")
    obs, _ = env.reset()
    
    try:
        while True:
            # 1. The brain decides the best action
            action, _states = model.predict(obs, deterministic=True)
            
            # 2. Send the action to the ROS 2 Server
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()
                
            # 3. Slow down the loop so it plays in real-time
            # (Otherwise the ROS 2 server will simulate it too fast to see!)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Inference stopped manually.")
    finally:
        env.close()

if __name__ == '__main__':
    main()