import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


class Actor(tf.keras.Model):
    def __init__(self, a_dim, a_bound):
        super().__init__()
        self.a_bound = a_bound
        self.l1 = layers.Dense(300, activation='relu', name='actor_dense1')
        self.l2 = layers.Dense(200, activation='relu', name='actor_dense2')
        self.out = layers.Dense(a_dim, activation='tanh', name='actor_output')

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.out(x) * self.a_bound


class Critic(tf.keras.Model):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.state_dense = layers.Dense(150, activation='relu', name='critic_state_dense')
        self.action_dense = layers.Dense(150, activation='relu', name='critic_action_dense')
        self.concat_dense = layers.Dense(300, activation='relu', name='critic_concat_dense')
        self.l2 = layers.Dense(200, activation='relu', name='critic_dense2')
        self.out = layers.Dense(1, name='critic_output')

    def call(self, s, a):
        state_h = self.state_dense(s)
        action_h = self.action_dense(a)
        concat = tf.concat([state_h, action_h], axis=1)
        x = self.concat_dense(concat)
        x = self.l2(x)
        return self.out(x)


class ddpg:
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.actor = Actor(a_dim, self.a_bound)
        self.actor_target = Actor(a_dim, self.a_bound)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)
        dummy_state = tf.zeros((1, s_dim))
        dummy_action = tf.zeros((1, a_dim))
        self.actor(dummy_state)
        self.actor_target(dummy_state)
        self.critic(dummy_state, dummy_action)
        self.critic_target(dummy_state, dummy_action)
        self.update_target(self.actor_target, self.actor, tau=1.0)
        self.update_target(self.critic_target, self.critic, tau=1.0)
        self.actor_opt = tf.keras.optimizers.Adam(LR_A)
        self.critic_opt = tf.keras.optimizers.Adam(LR_C)

    def update_target(self, target_net, source_net, tau=TAU):
        for target_param, source_param in zip(target_net.trainable_variables, source_net.trainable_variables):
            target_param.assign((1 - tau) * target_param + tau * source_param)

    def choose_action(self, s):
        s = tf.convert_to_tensor(s[np.newaxis, :], dtype=tf.float32)
        return self.actor(s).numpy()[0]

    def learn(self):
        if self.pointer < BATCH_SIZE:
            return

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        bs = tf.convert_to_tensor(bs, dtype=tf.float32)
        ba = tf.convert_to_tensor(ba, dtype=tf.float32)
        br = tf.convert_to_tensor(br, dtype=tf.float32)
        bs_ = tf.convert_to_tensor(bs_, dtype=tf.float32)
        with tf.GradientTape() as tape:
            a_target = self.actor_target(bs_)
            q_target = self.critic_target(bs_, a_target)
            y = br + GAMMA * q_target
            q = self.critic(bs, ba)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            a_pred = self.actor(bs)
            actor_loss = -tf.reduce_mean(self.critic(bs, a_pred))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.update_target(self.actor_target, self.actor)
        self.update_target(self.critic_target, self.critic)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True

    def save(self):
        try:
            import os
            os.makedirs("models1", exist_ok=True)
            self.actor.save("models1/actor_model", save_format="tf")
            self.critic.save("models1/critic_model", save_format="tf")
            print("Models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")
            try:
                self.actor.save_weights("models1/actor_weights.ckpt")
                self.critic.save_weights("models1/critic_weights.ckpt")
                print("Model weights saved successfully!")
            except Exception as e2:
                print(f"Error saving weights: {e2}")

    def restore(self):
        try:
            self.actor = tf.keras.models.load_model("models1/actor_model")
            self.critic = tf.keras.models.load_model("models1/critic_model")
        except:
            try:
                self.actor.load_weights("models1/actor_weights.ckpt")
                self.critic.load_weights("models1/critic_weights.ckpt")
            except Exception as e:
                print(f"Error loading models: {e}")
                return
        self.update_target(self.actor_target, self.actor, tau=1.0)
        self.update_target(self.critic_target, self.critic, tau=1.0)

    def add_exploration_noise(self, action, noise_scale=0.1):
        noise = np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action + noise, -self.a_bound, self.a_bound)

    def close(self):
        pass 

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state