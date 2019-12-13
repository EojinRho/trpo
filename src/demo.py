# independent, pure numpy inference

import numpy as np
import random
import json
import os
import sys
import argparse

from env import make_env

render_mode = True

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple feedforward model '''
  def __init__(self, filename, mean_eval_mode=True):

    self.mean_eval_mode = mean_eval_mode

    with open(filename) as f:    
      data = json.load(f)

    self.env_name = data[0]
    self.scale = data[1][0]
    self.offset = data[1][1]
    self.scale[-1] = 1.0  # don't scale time step feature
    self.offset[-1] = 0.0  # don't offset time step feature

    self.w0 = np.array(data[2][1][0])
    self.b0 = np.array(data[2][1][1])
    self.w1 = np.array(data[2][1][2])
    self.b1 = np.array(data[2][1][3])
    self.w2 = np.array(data[2][1][4])
    self.b2 = np.array(data[2][1][5])
    self.w3 = np.array(data[2][1][6])
    self.b3 = np.array(data[2][1][7])

    log_vars = np.array(data[2][1][8])
    self.noise_bias = data[3]

    self.log_vars = np.sum(log_vars, axis=0) + self.noise_bias
    self.sigma = np.exp(self.log_vars / 2.0)

    self.input_size = self.w0.shape[0]
    self.output_size = self.w3.shape[1]

    self.shapes = [ self.w0.shape,
                    self.w1.shape,
                    self.w2.shape,
                    self.w3.shape]

    self.activations = [np.tanh, np.tanh, np.tanh, passthru]


    self.weight = [self.w0, self.w1, self.w2, self.w3]
    self.bias = [self.b0, self.b1, self.b2, self.b3]
    self.param_count = 0
    for shape in self.shapes:
      self.param_count += (np.product(shape) + shape[1])
    self.param_count += np.product(self.sigma.shape)

  def make_env(self, seed=-1, render_mode=False):
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, x, t=0):
    h = np.array(x).flatten()
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      h = self.activations[i](h)
    mu = h
    if self.mean_eval_mode:
      a = mu
    else:
      eps = np.random.randn(self.output_size)
      a = mu + self.sigma * eps
    return a

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1):

  reward_list = []
  t_list = []

  max_episode_length = 18000 # assume each episode is less than 5min @ 60fps

  scale = model.scale
  offset = model.offset

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    obs = model.env.reset()
    if obs is None:
      obs = np.zeros(model.input_size-1)

    total_reward = 0.0
    step = 0.0 # extra input

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")

      obs = obs.astype(np.float32).reshape((1, -1))
      obs = np.append(obs, [[step]], axis=1)  # add time step feature

      obs = (obs - offset) * scale  # center and scale observations

      action = model.get_action(obs.flatten())

      obs, reward, done, _, rewards = model.env.step(action)

      #if (render_mode):
      #  print("step reward", reward)

      total_reward += reward
      step += 1e-3  # increment time step feature

      if done:
        break

    if render_mode:
      print("reward", total_reward, "timesteps", t+1)

    reward_list.append(total_reward)
    t_list.append(t+1)

  return reward_list, t_list

def main(model_file, mean_eval):

  model = Model(model_file)

  model.make_env(render_mode=render_mode)
  model.mean_eval_mode=(mean_eval==1)

  while(1):
    reward, steps_taken = simulate(model, train_mode=False, render_mode=render_mode, num_episode=1)
    print ("terminal reward", reward, "average steps taken", np.mean(steps_taken))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Demo a pre-trained policy.'))
    parser.add_argument('model_file', type=str, help='full path to json file')
    parser.add_argument('-m', '--mean_eval', type=int, help='if 1, will use mean, rather than sampling from policy. 1 by default.', default=1)

    args = parser.parse_args()

    main(**vars(args))
