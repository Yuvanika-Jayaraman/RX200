import rclpy
from rclpy.node import Node
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

class pickupnode(Node):
    def __init__(self):
        self.bot=InterbotixManipulatorXS(robot_model='rx200', group_name='arm', gripper_name='gripper',)
        robot_startup()
        self.command()
        robot_shutdown()

    def command(self):
        self.bot.arm.go_to_home_pose()
        time.sleep(1)
        self.bot.arm.set_single_joint_position(joint_name='waist', position=-0.832)
        time.sleep(1)
        self.bot.gripper.release(2.0)
        time.sleep(1)
        self.bot.arm.set_single_joint_position(joint_name='elbow', position=0.865)
        time.sleep(1)
        self.bot.arm.set_single_joint_position(joint_name='wrist_angle', position=-0.062)
        time.sleep(1)
        self.bot.gripper.grasp()
        time.sleep(1)
        self.bot.arm.set_single_joint_position(joint_name='wrist_angle', position=0.0)
        time.sleep(1)
        self.bot.arm.set_single_joint_position(joint_name='elbow', position=0.0)
        time.sleep(1)
        self.bot.gripper.release()
        time.sleep(1)
        self.bot.arm.go_to_home_pose()
        time.sleep(1)
        self.bot.arm.go_to_sleep_pose()
        time.sleep(1)
 

def main(args=None):
     rclpy.init(args=args)
     node = pickupnode()
     node.destroy_node()
     rclpy.shutdown()

if __name__ == '__main__':
    main()
