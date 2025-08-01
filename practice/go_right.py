import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand

class rx200goright(Node):
    def __init__(self):
        super().__init__('rx200_go_right')
        self.publisher = self.create_publisher(
            JointGroupCommand,
            '/rx200/commands/joint_group',
            10
        )
        cmd = JointGroupCommand()
        cmd.name = "arm"
        cmd.cmd = [-0.5, 0.0, 0.0, 0.0, 0.0]
        self.publisher.publish(cmd)
        print("Robot arm moved to right")

def main(args=None):
    rclpy.init(args=args)
    node = rx200goright()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
