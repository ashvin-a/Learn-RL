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

    def step_callback(self, ):
        pass

    def reset_callback(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = MujocoWalkerServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()