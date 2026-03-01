import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node

# Import our custom ROS 2 blueprints
from walker_interfaces.srv import StepPhysics, ResetPhysics

class ROS2WalkerEnv(gym.Env):
    """ Custom Gymnasium Environment that bridges SB3 to ROS 2 MuJoCo 

    A mental map to help understand the workflow:

        1. Brain (SB3) generates an Action.
        2. Client (Gym Wrapper) packages the action into a ROS Request and pauses.
        3. Server (MuJoCo) receives the action, steps the physics, and sends back the State.
        4. Client (Gym Wrapper) wakes up, unpacks the state, calculates the Reward, and returns it.
        5. Brain (SB3) updates its neural network and loop repeats!
    """

    
    def __init__(self):
        super(ROS2WalkerEnv, self).__init__()
        
        # Initialize ROS 2 Node inside the environment
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('rl_gym_client')
        
        # Define Spaces
        # Walker2D has 6 actuators (thigh, leg, foot for both sides)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Walker2D observation space: 17 dimensions (8 positional + 9 velocity)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        # Create Service Clients
        self.step_cli = self.node.create_client(StepPhysics, 'step_physics')
        self.reset_cli = self.node.create_client(ResetPhysics, 'reset_physics')
        
        # Wait for the MuJoCo server to wake up
        while not self.step_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for MuJoCo Physics Server...')

    def step(self, action):
        # Create Request
        req = StepPhysics.Request()
        req.action = action.astype(float).tolist()
        
        # Send Request and BLOCK until the server replies
        future = self.step_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        
        # Unpack Response
        obs = np.array(response.observation, dtype=np.float32)
        terminated = response.terminated
        truncated = response.truncated
        right_contact, left_contact = response.foot_contacts
        
        # Calculate the Reward (Anti-Hopping Logic)
        # Forward velocity is typically the first element of qvel (index 8 in our obs array)
        forward_vel = obs[8] 
        reward = float(forward_vel)  # Primary goal: move forward!
        
        # Symmetry / Bipedal Shaping
        if right_contact and left_contact:
            # Both feet on the ground (standing/double support) - small bonus
            reward += 0.1
        elif (right_contact and not left_contact) or (left_contact and not right_contact):
            # One foot on the ground (healthy walking gait) - larger bonus
            reward += 0.3
        elif not right_contact and not left_contact:
            # Both feet in the air (jumping/hopping excessively) - PENALTY
            reward -= 0.5 
            
        # Survival bonus (don't fall over!)
        reward += 1.0
        
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        req = ResetPhysics.Request()
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        
        obs = np.array(response.observation, dtype=np.float32)
        info = {}
        return obs, info

    def close(self):
        """ Clean up ROS 2 nodes when training finishes """
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()