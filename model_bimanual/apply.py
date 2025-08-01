import numpy as np
from rl3.env import rx200dualenv, forward_kinematics
from rl3.ddpg import ddpg
import time

def main():
    env = rx200dualenv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    agent = ddpg(a_dim, s_dim, a_bound)
    agent.restore()
    state = env.reset()
    done = False
    max_steps = 100
    for step in range(max_steps):
        action = agent.choose_action(state)
        action1 = action[:5]
        action2 = action[5:]

        state, reward, done, _ = env.step(action1, action2)

        ee1 = forward_kinematics(env.state1)
        ee2 = forward_kinematics(env.state2)
        print(f"Step {step}: EE1: {ee1}, EE2: {ee2}, Reward: {reward:.4f}")

        if done:
            print("Task complete or max steps reached.")
            break

    env.close()

if __name__ == '__main__':
    main()
