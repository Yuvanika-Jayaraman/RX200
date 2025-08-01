import numpy as np
import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand
from scipy.linalg import expm
from gym.spaces import Box
import threading
import time

def get_pose_from_T(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    return [x, y, z]

def to_s_matrix(w, v):
    w_x, w_y, w_z = w
    v_x, v_y, v_z = v
    return np.array([
        [0, -w_z, w_y, v_x],
        [w_z, 0, -w_x, v_y],
        [-w_y, w_x, 0, v_z],
        [0, 0, 0, 0]
    ])

def FK_pox(joint_angles, M, S):
    T = np.eye(4)
    for i in range(len(joint_angles)):
        S_i = to_s_matrix(S[i, :3], S[i, 3:])
        T = T @ expm(S_i * joint_angles[i])
    T = T @ M
    return T

S = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, -0.10457, 0, 0],
    [0, 1, 0, -0.30457, 0, 0.05],
    [0, 1, 0, -0.30457, 0, 0.25],
    [1, 0, 0, 0, 0.30457, 0]
])

M = np.array([
    [1, 0, 0, 0.408575],
    [0, 1, 0, 0],
    [0, 0, 1, 0.30457],
    [0, 0, 0, 1]
])


def forward_kinematics(joint_angles):
    T = FK_pox(joint_angles, M, S)
    return np.array(get_pose_from_T(T))

class dualarmROSnode(Node):
    def __init__(self):
        super().__init__('dual_arm_rl_node')
        self.publisher1 = self.create_publisher(JointGroupCommand, '/rx200_1/commands/joint_group', 10)
        self.publisher2 = self.create_publisher(JointGroupCommand, '/rx200_2/commands/joint_group', 10)
        self.cmd1=JointGroupCommand()
        self.cmd2=JointGroupCommand()
        self.cmd1.name='arm'
        self.cmd2.name='arm'
    
    def publish(self, command1, command2):
        self.cmd1.cmd=list(command1)
        self.cmd2.cmd=list(command2)
        self.publisher1.publish(self.cmd1)
        self.publisher2.publish(self.cmd2)
        time.sleep(0.5)

class rx200dualenv:
    def __init__(self, use_ros=True):
        self.n_joints = 5
        self.state1 = np.zeros(self.n_joints)
        self.state2 = np.zeros(self.n_joints)
        self.goal1 = np.array([0.3, 0.1, 0.2])
        self.goal2 = np.array([0.3, 0.1, 0.2])
        self.current_step = 0
        self.max_steps = 100
        self.success_threshold = 0.05
        self.use_ros = use_ros
        self.joint_limits = {'lower': np.array([-3.14159, -1.88496, -1.5708, -1.745, -2.617]),'upper': np.array([3.14159, 1.98968, 1.5708, 1.745, 2.617])}
        self.observation_space = Box(low=np.tile(self.joint_limits['lower'], 2), high=np.tile(self.joint_limits['upper'], 2), dtype=np.float32)
        self.action_space = Box(low=-0.1, high=0.1, shape=(self.n_joints*2,), dtype=np.float32)
        self.action_dim = self.n_joints*2
        self.state_dim = self.n_joints*2
        self.action_bound = [-0.1, 0.1]  
        self.reward_range = (-np.inf, 0)
        self.ros_node = None
        self.ros_thread = None
        self.ros_spinning = False
        
        if self.use_ros:
            self._init_ros()

    def _init_ros(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.ros_node = dualarmROSnode()

            self.ros_spinning = True
            self.ros_thread = threading.Thread(target=self._spin_ros)
            self.ros_thread.daemon = True
            self.ros_thread.start()
            
            print("ROS node initialized successfully")
        except Exception as e:
            print(f"Failed to initialize ROS: {e}")
            self.use_ros = False

    def _spin_ros(self):
        while self.ros_spinning and rclpy.ok():
            try:
                rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            except Exception as e:
                print(f"Error in ROS spinning: {e}")
                break


    def step(self, action1, action2):
        try:
            self.current_step += 1
            action1 = (np.clip(action1, self.action_bound[0], self.action_bound[1]))
            action2 = (np.clip(action2, self.action_bound[0], self.action_bound[1]))
            self.state1 = self.state1 + action1
            self.state1 = np.clip(self.state1, self.joint_limits['lower'], self.joint_limits['upper'])
            self.state2 = self.state2 + action2
            self.state2 = np.clip(self.state2, self.joint_limits['lower'], self.joint_limits['upper'])
            if self._is_safe_pose(self.state1) and self._is_safe_pose(self.state2):
                if self.use_ros and self.ros_node:
                    self.ros_node.publish(self.state1, self.state2)
                    time.sleep(0.5)
            else:
                return self._get_observation(), -10, True, {'error': 'unsafe_pose'}
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -10, True, {'error': str(e)}

        ee_pos1 = forward_kinematics(self.state1)
        ee_pos2 = forward_kinematics(self.state2)
        dist1 = np.linalg.norm(ee_pos1 - self.goal1)
        dist2 = np.linalg.norm(ee_pos2-self.goal2)
        reward = -(dist1+dist2)
        arm_distance = np.linalg.norm(ee_pos1 - ee_pos2)
        done = self.current_step >= self.max_steps or (dist1 < self.success_threshold and dist2 < self.success_threshold)
        return self._get_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.state1 = np.zeros(self.n_joints)
        self.state2 = np.zeros(self.n_joints)

        if self.use_ros and self.ros_node:
            self.ros_node.publish(self.state1, self.state2)  

        return self._get_observation()


    def render(self):
        ee1= forward_kinematics(self.state1)
        ee2=forward_kinematics(self.state2)
        print("End-effector Position 1 and 2:", ee1, ' ',ee2)

    def close(self):
        self.current_step = 0
        self.state1 = np.zeros(self.n_joints)
        self.state2 = np.zeros(self.n_joints)

        if self.use_ros and self.ros_node:
            self.ros_node.publish(self.state1, self.state2)  

    def _get_observation(self):
        return np.concatenate([self.state1, self.state2])

    def _is_safe_pose(self, joint_angles):
        ee_pos = forward_kinematics(joint_angles)
        if ee_pos[2] < 0.05:  
            return False
        if np.linalg.norm(ee_pos[:2]) > 0.5:
            return False
        return True
