""" List of environments that should work

RoboschoolAnt-v1
RoboschoolHopper-v1
RoboschoolReacher-v1
RoboschoolWalker2d-v1
RoboschoolHumanoid-v1
RoboschoolHumanoidFlagrunHarder-v1
RoboschoolAtlasForwardWalk-v1
RoboschoolInvertedDoublePendulum-v1
RoboschoolInvertedPendulumSwingup-v1

HumanoidBulletEnv-v0
RacecarBulletEnv-v0
MinitaurBulletEnv-v0
HopperBulletEnv-v0
AntBulletEnv-v0
Walker2DBulletEnv-v0

BipedalWalker-v2
BipedalWalkerHardcore-v2

"""

import numpy as np
import gym

import roboschool

import pybullet as p
import pybullet_envs
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
import pybullet_envs.bullet.racecarGymEnv as racecarGymEnv

def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("RacecarBulletEnv")):
    print("bullet_racecar_started")
    env = racecarGymEnv.RacecarGymEnv(isDiscrete=False, renders=render_mode)
  elif (env_name.startswith("MinitaurBulletEnv")):
    print("bullet_minitaur_started")
    env = minitaur_gym_env.MinitaurBulletEnv(render=render_mode)
  else:
    env = gym.make(env_name)
  if render_mode and ("BulletEnv-v0" in env_name):
    env.render("human")
  if (seed >= 0):
    env.seed(seed)

  return env
