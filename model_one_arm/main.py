from rl2.env import rx200armenv
from rl2.ddpg import ddpg
import numpy as np
import logging
import os
from rl2.env import forward_kinematics  # if it's a top-level function

log_dir = os.path.join(os.path.expanduser("~"), "rx200_rl2_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_improve1.log")
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
    env = rx200armenv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound
    agent = ddpg(a_dim, s_dim, a_bound)
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
            s_, reward, done, _ = env.step(action)
            agent.store_transition(s, action, reward, s_)
            episode_reward += reward
            step_count += 1
            logger.info(f"Step {step_count}: Action={action}, New State={s_}")

            if agent.memory_full:
                agent.learn()

            s = s_

        episode_rewards.append(episode_reward)

        ee_pos = forward_kinematics(env.state)
        final_dist = np.linalg.norm(ee_pos - env.goal)
        success = final_dist < env.success_threshold

        logger.info(f"Episode Summary:")
        logger.info(f"  Reward: {episode_reward:.4f}")
        logger.info(f"  Final EE Pos: {ee_pos}")
        logger.info(f"  Goal: {env.goal}")
        logger.info(f"  Final Distance: {final_dist:.4f}")
        logger.info(f"  Success: {success}")

        # Save models periodically
        if (i + 1) % save_every == 0:
            logger.info("Saving model...")
            agent.save()

    env.close()
    logger.info("Training completed.")


if __name__ == '__main__':
    main()
