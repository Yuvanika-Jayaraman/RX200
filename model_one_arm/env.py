import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from scipy.linalg import expm
import time
from gym.spaces import Box

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


class rx200armenv:
    def __init__(self):
        self.n_joints = 5
        self.state = np.zeros(self.n_joints)
        self.goal = np.array([0.3, 0.1, 0.2])
        self.current_step = 0
        self.max_steps = 100
        self.success_threshold = 0.025
        self.bot = InterbotixManipulatorXS(robot_model='rx200', group_name='arm', gripper_name='gripper', robot_name='rx200_1')
        self.joint_limits = {'lower': np.array([-3.14159, -1.88496, -1.5708, -1.745, -2.617]),'upper': np.array([3.14159, 1.98968, 1.5708, 1.745, 2.617])}
        self.observation_space = Box(low=self.joint_limits['lower'], high=self.joint_limits['upper'], shape=(self.n_joints,), dtype=np.float32)
        self.action_space = Box(low=-0.1, high=0.1, shape=(self.n_joints,), dtype=np.float32)
        self.action_dim = self.n_joints
        self.state_dim = self.n_joints
        self.action_bound = [-0.1, 0.1]  
        self.reward_range = (-np.inf, 0)

    def step(self, action):
        try:
            self.current_step += 1
            action = (np.clip(action, self.action_bound[0], self.action_bound[1]))
            self.state = self.state + action
            self.state = np.clip(self.state, self.joint_limits['lower'], self.joint_limits['upper'])
            if self._is_safe_pose(self.state):
                self.bot.arm.set_joint_positions(self.state)
                time.sleep(0.1)
            else:
                return self._get_observation(), -10, True, {'error': 'unsafe_pose'}
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -10, True, {'error': str(e)}

        ee_pos = forward_kinematics(self.state)
        dist = np.linalg.norm(ee_pos - self.goal)
        reward = -dist
        done = self.current_step >= self.max_steps or dist < self.success_threshold
        return self._get_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.bot.arm.go_to_home_pose()
        self.state = np.zeros(self.n_joints)
        return self._get_observation()

    def render(self):
        ee_pos = forward_kinematics(self.state)
        print("End-effector Position:", ee_pos)

    def close(self):
        self.bot.arm.go_to_sleep_pose()

    def _get_observation(self):
        return self.state.copy()

    def _is_safe_pose(self, joint_angles):
        ee_pos = forward_kinematics(joint_angles)
        if ee_pos[2] < 0.05:  
            return False
        if np.linalg.norm(ee_pos[:2]) > 0.5:
            return False
        return True
