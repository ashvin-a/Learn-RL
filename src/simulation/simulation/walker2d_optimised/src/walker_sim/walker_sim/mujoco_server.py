import rclpy
from rclpy.node import Node
import mujoco
import numpy as np
import os

from walker_interfaces.srv import StepPhysics, ResetPhysics

class MujocoWalkerServer(Node):

    def __init__(self):
        super().__init__("mujoco_server")

        xml_path = "./walker_sim/resource/walker2d_fixed.xml"

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            self.get_logger().info("Mujoco model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Mujoco model moonnjich gooys!! {e}")
        
        # Creating ROS services
        self.step_srv = self.create_service(StepPhysics, "step_physics", self.step_callback)
        self.reset_srv = self.create_service(ResetPhysics, "reset_physics", self.reset_callback)

        # Caching Geom IDs for Foot Contact Detection
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot')
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_left')

        self.get_logger().info("MuJoCo Physics Server is UP and waiting for actions...")

    def step_callback(self, request, response):
        """ Triggered every time SB3 takes an action. """
        
        # Apply actions (torques) to actuators
        self.data.ctrl[:] = request.action

        # Step the physics engine forward exactly one timestep
        mujoco.mj_step(self.model, self.data)

        # Populate Response
        response.observation = self.get_observation()
        response.foot_contacts = self.check_foot_contacts()

        # Check Termination condition (Did the Walker fall over?)
        # Typically, qpos[1] is the Z-height of the torso.
        torso_height = self.data.qpos[1]
        torso_angle = self.data.qpos[2] # Pitch angle
        
        #* Terminate if it crouches too low or leans too far forward/backward
        response.terminated = bool(torso_height < 0.8 or abs(torso_angle) > 1.0)
        response.truncated = False

        return response

    def check_foot_contacts(self):
        """ Checks the MuJoCo contact array to see if the feet are touching the floor. """
        right_contact = False
        left_contact = False

        # Iterate through all active contacts in the physics step
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check Right Foot
            if (geom1 == self.floor_geom_id and geom2 == self.right_foot_geom_id) or \
               (geom2 == self.floor_geom_id and geom1 == self.right_foot_geom_id):
                right_contact = True

            # Check Left Foot
            if (geom1 == self.floor_geom_id and geom2 == self.left_foot_geom_id) or \
               (geom2 == self.floor_geom_id and geom1 == self.left_foot_geom_id):
                left_contact = True

        return [right_contact, left_contact]

    def get_observation(self, ):
        """Gets info from Mujoco and send it to RL"""
        # In walker2D, we skip the X-translation (qpos[0]) so the agent 
        # doesn't memorise its global position, making the policy 
        # translation-invariant
        qpos = self.data.qpos[1:]
        qvel = self.data.qvel
        return np.concatenate([qpos, qvel]).astype(float).tolist()

    def reset_callback(self, request, response):
        """ Triggered when the episode ends. """
        mujoco.mj_resetData(self.model, self.data)
        
        # Add slight random noise to initial state to prevent deterministic overfitting
        self.data.qpos[:] += np.random.uniform(low=-0.005, high=0.005, size=self.model.nq)
        self.data.qvel[:] += np.random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        
        mujoco.mj_forward(self.model, self.data) # Update kinematic tree
        
        response.observation = self.get_observation()
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MujocoWalkerServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()