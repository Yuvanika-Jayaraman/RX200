import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand

class rx200goopp(Node):
    def __init__(self):
        super().__init__('rx200_go_opposite')
        self.publisher1 = self.create_publisher(
            JointGroupCommand,
            '/rx200_1/commands/joint_group',
            10
        )
        self.publisher2 = self.create_publisher(
            JointGroupCommand,
            '/rx200_2/commands/joint_group',
            10
        )
        cmd1 = JointGroupCommand()
        cmd1.name = "arm"
        cmd1.cmd = [-0.5, 0.0, 0.0, 0.0, 0.0]
        self.publisher1.publish(cmd1)
        cmd2=JointGroupCommand()
        cmd2.name="arm"
        cmd2.cmd=[0.5, 0.0, 0.0, 0.0, 0.0]
        self.publisher2.publish(cmd2)
        print("rx200_1 moved to right and rx200_2 moved to left")

def main(args=None):
    rclpy.init(args=args)
    node = rx200goopp()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
