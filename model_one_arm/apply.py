import tensorflow as tf 
from rl2.ddpg import ddpg
from rl2.env import rx200armenv
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np 
import time 

def main():
    env = rx200armenv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    agent = ddpg(a_dim, s_dim, a_bound)
    agent.restore()
    state=env.reset()
    try:
        for i in range(env.max_steps):
            action=agent.choose_action(state)
            next_state, reward, done, info= env.step(action)
            print(f"Step {i}: Reward={reward:.4f}, Done={done}, EE Distance={-reward:.4f}")
            state=next_state
            if done:
                print("Successful")
                break
    except KeyboardInterrupt:
        print("user interrupt")
    finally:
        print("shutting down")
        env.close()

if __name__=="__main__":
    main()

