import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand
from sensor_msgs.msg import JointState

class rx200goopp(Node):
    def __init__(self):
        super().__init__('rx200_go_opposite')
        self.publisher1=self.create_publisher(JointGroupCommand, '/rx200_1/commands/joint_group', 10)
        self.publisher2=self.create_publisher(JointGroupCommand, '/rx200_2/commands/joint_group', 10)
        cmd1=JointGroupCommand()
        cmd1.name="arm"
        cmd1.cmd=[-0.5, 0.8, -1.2, 1.0, -0.6]
        self.publisher1.publish(cmd1)
        self.new = [] 
        self.subscriber2=self.create_subscription(JointState, '/rx200_1/joint_states', self.sub_callback, 10)
        self.timer=self.create_timer(0.5, self.t_callback)
        
    def sub_callback(self, msg):
        a=[]
        for i in range(5):
            if msg.name[i]=='waist':
                a.append(-msg.position[i])
            elif msg.name[i]=='wrist_rotate':
                a.append(-msg.position[i])
            else:
                a.append(msg.position[i])
        self.new = a
    
    def t_callback(self):
        cmd2=JointGroupCommand()
        cmd2.name="arm"
        cmd2.cmd=self.new
        self.publisher2.publish(cmd2)
        print("rx200_1 moved to right and rx200_2 moved to left")

def main(args=None):
    rclpy.init(args=args)
    node=rx200goopp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
