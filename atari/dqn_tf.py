import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 80
K = 4

def downsample_image(A):
  B = A[31:195]
  B = B.mean(axis=2)

  B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
  return B


def update_state(state, obs):
  obs_small = downsample_image(obs)
  return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope):

    self.K = K
    self.scope = scope

    with tf.variable_scope(scope):

      self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_SIZE, IM_SIZE), name='X')

      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

      Z = self.X / 255.0
      Z = tf.transpose(Z, [0, 2, 3, 1])
      for num_output_filters, filtersz, poolsz in conv_layer_sizes:
        Z = tf.contrib.layers.conv2d(
          Z,
          num_output_filters,
          filtersz,
          poolsz,
          activation_fn=tf.nn.relu
        )

      Z = tf.contrib.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.contrib.layers.fully_connected(Z, M)

      self.predict_op = tf.contrib.layers.fully_connected(Z, K)

      selected_action_values = tf.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )

      cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
      self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
      self.cost = cost

  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)

    self.session.run(ops)

  def set_session(self, session):
    self.session = session

  def predict(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    return c

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  samples = random.sample(experience_replay_buffer, batch_size)
  states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

  loss = model.update(states, actions, targets)
  return loss


def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  obs = env.reset()
  obs_small = downsample_image(obs)
  state = np.stack([obs_small] * 4, axis=0)
  assert(state.shape == (4, 80, 80))
  loss = None

  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))

    action = model.sample_action(state, epsilon)
    obs, reward, done, _ = env.step(action)
    obs_small = downsample_image(obs)
    next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)

    episode_reward += reward

    if len(experience_replay_buffer) == MAX_EXPERIENCES:
      experience_replay_buffer.pop(0)
    experience_replay_buffer.append((state, action, reward, next_state, done))
    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2
    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1
    state = next_state
    total_t += 1
    epsilon = max(epsilon - epsilon_change, epsilon_min)
  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon


if __name__ == '__main__':
  conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 10000
  total_t = 0
  experience_replay_buffer = []
  episode_rewards = np.zeros(num_episodes)

  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000

  env = gym.envs.make("Breakout-v0")

  model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="model")
  target_model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="target_model"
  )

  with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())

    print("Populating experience replay buffer...")
    obs = env.reset()
    obs_small = downsample_image(obs)
    state = np.stack([obs_small] * 4, axis=0)
    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        next_state = update_state(state, obs)
        experience_replay_buffer.append((state, action, reward, next_state, done))

        if done:
            obs = env.reset()
            obs_small = downsample_image(obs)
            state = np.stack([obs_small] * 4, axis=0)
        else:
            state = next_state

    for i in range(num_episodes):
      total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
        env,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        gamma,
        batch_sz,
        epsilon,
        epsilon_change,
        epsilon_min,
      )
      episode_rewards[i] = episode_reward

      last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
      print("Episode:", i,
        "Duration:", duration,
        "Num steps:", num_steps_in_episode,
        "Reward:", episode_reward,
        "Training time per step:", "%.3f" % time_per_step,
        "Avg Reward (Last 100):", "%.3f" % last_100_avg,
        "Epsilon:", "%.3f" % epsilon
      )
      sys.stdout.flush()