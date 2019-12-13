#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
from env import make_env
import numpy as np
from gym import wrappers
from policy import Policy
from policy_continue import PolicyContinue
from value_function_multi import NNValueFunction
from value_function_multi_continue import NNValueFunctionContinue
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import pickle

reward_dim = 6
alive_coef = 5.0
progress_coef = 1.0
change_rate = 0.003
threshold1 = 9 # I (junyong) changed this from 9 to 6 for the one running on the right terminal (Dec-06_05)
threshold2 = 1.5

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    #env = gym.make(env_name)
    env = make_env(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=True):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards_multi_list: shape = (episode len ,reward_dim)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards_multi_list, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        if isinstance(obs, list): obs = np.array(obs)
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        #print(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        #print(action)
        #assert False
        actions.append(action)
        obs, _, done, _, reward_multi = env.step(np.squeeze(action, axis=0))

        for i_tmp in range(len(reward_multi)):
            reward = reward_multi[i_tmp]
            if isinstance(reward, int):
                reward_multi[i_tmp] = float(reward)
            if not isinstance(reward, float):
                reward_multi[i_tmp] = np.asscalar(reward)

        rewards_multi_list.append(reward_multi)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards_multi_list, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : 2D NumPy array of (un-discounted) rewards(episode x num of rewards) from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        orig_rewards = np.copy(rewards)
        rewards[:,0] *= alive_coef
        rewards[:,1] *= progress_coef
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'orig_rewards': orig_rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1], axis=0)[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        #print("disc_sum_rew : {}".format(disc_sum_rew))
        #assert False,"disc check"
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value (same dim as reward dim)

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values_mul = val_func.predict(observes)
        trajectory['values'] = values_mul


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        # print(values.shape)
        # print(values)
        # print(values[1:,:])
        # print(np.append(values[1:,:] * gamma, [[0]*values.shape[1]], axis=0))
        # assert False
        tds = rewards - values + np.append(values[1:,:] * gamma, [[0]*values.shape[1]], axis=0)
        # print(rewards)
        # print(values)
        # print(np.append(values[1:,:] * gamma, [[0]*values.shape[1]], axis=0))
        # print(tds)
        # assert False
        advantages = discount(tds, gamma * lam)
        advantages = np.sum(advantages, axis=1)

        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """

    # print(np.concatenate([t['observes'] for t in trajectories]).shape)
    # print(np.concatenate([t['disc_sum_rew'] for t in trajectories]).shape)
    # assert False
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(np.sum(disc_sum_rew, axis=1)),
                '_min_discrew': np.min(np.sum(disc_sum_rew, axis=1)),
                '_max_discrew': np.max(np.sum(disc_sum_rew, axis=1)),
                '_std_discrew': np.var(np.sum(disc_sum_rew, axis=1)),
                '_Episode': episode
                })


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, net_size_factor, noise_bias, weight, use_ppoclip):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    global alive_coef, progress_coef
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    # now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    now = datetime.now().strftime("%b-%d_%H:%M:%S") + "_multi"
    logger = Logger(logname=env_name, now=now)
    aigym_path = os.path.join('/tmp', env_name, now)
    # env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    if weight == "None":
        val_func = NNValueFunction(obs_dim, net_size_factor=net_size_factor, alive_coef=alive_coef, progress_coef=progress_coef)
        policy = Policy(obs_dim, act_dim, kl_targ, net_size_factor=net_size_factor, noise_bias=noise_bias)

    else:
        token = weight.split(".")
        token[-3] = token[-3][:-5] + "value"
        weight_2 = ".".join(token)
        # assert False, "unreachable"
        val_func = NNValueFunctionContinue(weight_2, obs_dim, net_size_factor=net_size_factor, alive_coef=alive_coef, progress_coef=progress_coef)
        policy = PolicyContinue(weight, obs_dim, act_dim, kl_targ, net_size_factor=net_size_factor, noise_bias=noise_bias)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger, scaler)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break

            if input('Change alive_coef (y/[n])? ') == 'y':
                a = input("alive_coef value: ")
                alive_coef = float(a)
                val_func.alive_coef = float(a)
            if input('Change progress_coef (y/[n])? ') == 'y':
                a = input("progress_coef value: ")
                progress_coef = float(a)
                val_func.progress_coef = float(a)
            killer.kill_now = False
    logger.close()
    # with open("test_dump", 'w') as f:
    #     pickle.dump(policy, f)
    policy.close_sess()
    val_func.close_sess()

def main2(env_name, num_episodes, gamma, lam, kl_targ, batch_size, net_size_factor, noise_bias, weight, use_ppoclip):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    global alive_coef, progress_coef, threshold1, threshold2, change_rate
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    # now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    now = datetime.now().strftime("%b-%d_%H:%M:%S") + "_multi_hop_{},{},{}".format(change_rate,threshold1,threshold2)
    logger = Logger(logname=env_name, now=now)
    aigym_path = os.path.join('/tmp', env_name, now)
    # env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    if weight == "None":
        val_func = NNValueFunction(obs_dim, net_size_factor=net_size_factor, alive_coef=alive_coef, progress_coef=progress_coef, reward_dim=reward_dim)
        policy = Policy(obs_dim, act_dim, kl_targ, net_size_factor=net_size_factor, noise_bias=noise_bias)

    else:
        token = weight.split(".")
        token[-3] = token[-3][:-5] + "value"
        weight_2 = ".".join(token)
        # assert False, "unreachable"
        val_func = NNValueFunctionContinue(weight_2, obs_dim, net_size_factor=net_size_factor, alive_coef=alive_coef, progress_coef=progress_coef)
        policy = PolicyContinue(weight, obs_dim, act_dim, kl_targ, net_size_factor=net_size_factor, noise_bias=noise_bias)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    flag1 = False
    flag2 = False
    flag3 = False
    reward_queue = []
    queue_num = 100
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger, scaler)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False

        alive_sum = 0
        progr_sum = 0
        for t in trajectories:
            tmp_rewards = t['orig_rewards']
            tmp_rewards = np.sum(tmp_rewards, axis = 0)
            alive_sum += tmp_rewards[0]
            progr_sum += tmp_rewards[1]
        reward_queue.append(np.mean([t['rewards'].sum() for t in trajectories]))
        reward_queue = reward_queue[-queue_num:]
        reward_std = np.std(np.array(reward_queue))

        print("Reward std by {} episode : {}".format(queue_num, reward_std))

        if alive_sum >= 5000:
            flag3 = True

        if (flag3 and alive_sum > progr_sum * threshold1) or flag1:
            flag1 = True
            alive_coef -= change_rate
            progress_coef += change_rate
            val_func.alive_coef = float(alive_coef)
            val_func.progress_coef = float(progress_coef)
            if alive_sum < progr_sum * threshold2:
                flag1 = False

        if progr_sum > alive_sum * threshold1 or flag2:
            flag2 = True
            alive_coef += change_rate
            progress_coef -= change_rate
            val_func.alive_coef = float(alive_coef)
            val_func.progress_coef = float(progress_coef)

            if progr_sum < alive_sum * threshold2:
                flag2 = False

        print(alive_sum, progr_sum)

        logger.log_model_3({"alive_coef": alive_coef, "progress_coef": progress_coef,
                        "alive_sum": alive_sum, "progr_sum": progr_sum})

    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('--num_episodes', type=int, help='Number of episodes to run',
                        default=1000000000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-f', '--net_size_factor', type=int, help='Factor controlling size of network',
                        default=5)
    parser.add_argument('--noise_bias', type=float, help='noise bias', default=-1.0)
    parser.add_argument('--weight', type=str, help='File for pretrained weights', default="None")
    parser.add_argument('--use_ppoclip', type=str, help="PPO implementation of clipped surrogate", default="False")

    args = parser.parse_args()
    # main(**vars(args))
    main2(**vars(args))
