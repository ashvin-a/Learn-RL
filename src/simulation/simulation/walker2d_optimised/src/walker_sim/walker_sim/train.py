import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from walker_sim.walker_env import ROS2WalkerEnv

def main():
    print("Initializing ROS 2 Walker2D Environment...")
    
    # 1. Instantiate the environment
    env = ROS2WalkerEnv()
    
    # Optional but highly recommended: SB3 has a built-in checker to ensure 
    # your custom environment obeys all Gymnasium rules (spaces, bounds, etc.)
    check_env(env)
    
    # 2. Initialize the SAC Agent
    # MlpPolicy means it uses a standard Multi-Layer Perceptron neural network
    print("Building the SAC Neural Network...")
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1000000,
        tensorboard_log="./walker2d_tensorboard/"
    )
    
    # 3. Train the Agent!
    # For a complex task like Walker2D, you usually need 1M to 3M timesteps.
    # We will set it to 100,000 here just to verify the pipeline works.
    print("Starting Training Loop...")
    try:
        model.learn(total_timesteps=100000, log_interval=4)
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")
    
    # 4. Save the trained policy
    model_path = os.path.join(os.getcwd(), "sac_walker2d_v1")
    model.save(model_path)
    print(f"Model saved successfully to {model_path}.zip")
    
    # 5. Clean up ROS 2 nodes
    env.close()

if __name__ == '__main__':
    main()