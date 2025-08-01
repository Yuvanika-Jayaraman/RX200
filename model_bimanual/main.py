from rl3.env import rx200dualenv
from rl3.ddpg import ddpg
import numpy as np
import logging
import os
from rl3.env import forward_kinematics  

log_dir = os.path.join(os.path.expanduser("~"), "rx200_rl3_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "dual_training1.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

max_episode = 1000 
save_every = 50     

def main():
    env = rx200dualenv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    agent = ddpg(a_dim, s_dim, a_bound)
    agent.load_single_arm_weights("models/actor_model")

    episode_rewards = []

    for i in range(max_episode):
        episode_reward = 0
        step_count = 0
        logger.info(f"Episode {i}")
        s = env.reset()
        logger.info(f"Initial joint angles: {s}")
        done = False

        while not done:
            action = agent.choose_action(s)
            action = agent.add_exploration_noise(action)
            action1 = action[:5]
            action2 = action[5:]
            s_, reward, done, _ = env.step(action1, action2)
            agent.store_transition(s, action, reward, s_)
            episode_reward += reward
            step_count += 1
            logger.info(f"Step {step_count}: Action1={action1}, Action2={action2}, New State={s_}")

            if agent.memory_full:
                agent.learn()

            s = s_

        episode_rewards.append(episode_reward)

        ee1 = forward_kinematics(env.state1)
        ee2 = forward_kinematics(env.state2)
        dist1 = np.linalg.norm(ee1 - env.goal1)
        dist2 = np.linalg.norm(ee2 - env.goal2)
        success = dist1 < env.success_threshold and dist2 < env.success_threshold


        logger.info(f"Episode Summary:")
        logger.info(f"  Reward: {episode_reward:.4f}")

        logger.info(f"  Final EE Pos of 1: {ee1}")
        logger.info(f"  Goal1: {env.goal1}")
        logger.info(f"  Final Distance: {dist1:.4f}")

        logger.info(f"  Final EE Pos of 2: {ee2}")
        logger.info(f"  Goal2: {env.goal2}")
        logger.info(f"  Final Distance: {dist2:.4f}")

        logger.info(f"  Success: {success}")
        
        if (i + 1) % save_every == 0:
            logger.info("Saving model...")
            agent.save()

    env.close()
    logger.info("Training completed.")


if __name__ == '__main__':
    main()
