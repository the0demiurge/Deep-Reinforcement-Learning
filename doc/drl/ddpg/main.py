#!/usr/bin/env python3
import json
import pickle
import time
from collections import deque
from random import gauss, randint, sample

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import Tensor

name = 'baseline'
device = torch.device("cuda:{}".format(randint(0, torch.cuda.device_count() - 1)) if torch.cuda.is_available() else "cpu")

M = 2000  # Episodes
T = 200  # Trajectory lenth
N = 64  # Batch size
R = 1e+6  # Replay size
epsilon = 0.1  # Exploration constant
lr_critic = 1e-3  # Initial_learning rates
lr_actor = 1e-4
gamma = 0.99
tau = 0.001
activ1 = nn.RReLU
activ2 = F.rrelu

env = gym.make('Pendulum-v0')
replay_buffer = deque(maxlen=int(R))


class OrnsteinUhlenbeckProcess(object):
    # Ornstein–Uhlenbeck process
    def __init__(self, dt=1, theta=.15, sigma=1, nums=1):
        self.x = [0] * nums
        self.dt = dt
        self.theta = theta
        self.sigma = sigma
        self.nums = nums

    def __call__(self):
        dx = [-self.theta * self.x[i] * self.dt + gauss(0, self.sigma) for i in range(self.nums)]
        self.x = [self.x[i] + dx[i] for i in range(self.nums)]
        return self.x

    def reset(self):
        self.x = [0] * self.nums


def copy(source, destination):
    destination.load_state_dict(source.state_dict())


def tau_move_average(source, destination, tau=0.1):
    d_dict = destination.state_dict()
    s_dict = source.state_dict()
    for key in d_dict:
        d_dict[key] = s_dict[key] * tau + d_dict[key] * (1 - tau)
    destination.load_state_dict(d_dict)


class Recorder(object):
    def __init__(self, *args):
        self.episode = 0
        self.n = 0
        self.args = args
        self.hist = {arg: list() for arg in args}
        self._stat = {arg: list() for arg in args}
        self.stat_method = {
            'mean': np.mean,
            'var': np.var,
            'max': max,
            'min': min,
        }
        self.stats = {
            arg: {key: list() for key in self.stat_method}
            for arg in args
        }

    def __call__(self, *args):
        assert len(args) == len(self.args), 'args numbers not match'
        for arg, val in zip(self.args, args):
            self._stat[arg].append(val)
            self.hist[arg].append(val)
        self.n += 1

    def reset(self):
        for arg in self.args:
            for key in self.stat_method:
                if len(self._stat[arg]) > 0:
                    self.stats[arg][key].append(self.stat_method[key](self._stat[arg]))

        self.episode += 1
        self.n = 0
        self._stat = {arg: list() for arg in self.args}
        self.print()

    def plot(self, title=None, show=False):
        plt.clf()
        if title:
            print(title)
        for arg in self.args:
            plt.plot(self.moving_average(self.stats[arg]['mean']), 'x-', label=arg, linewidth=0.5, markersize=1)
        plt.legend()
        plt.title(title)
        if show:
            plt.show()
        if title:
            plt.savefig('{}.png'.format(title))

    def print(self):
        print(self.episode, end='\t')
        for arg in self.args:
            print(arg, ('{:7.4f} ' * 3).format(
                self.stats[arg]['mean'][-1],
                self.stats[arg]['min'][-1],
                self.stats[arg]['max'][-1]),
                end='\t')
        print()

    def save(self, file_name):
        json.dump({'hist': self.hist, 'stats': self.stats}, open(file_name, 'w'))

    @staticmethod
    def moving_average(values, ratio=0.9):
        result = list()
        val = values[0]
        for v in values:
            val = ratio * val + (1 - ratio) * v
            result.append(val)
        return result


def Actor(state_size):
    return nn.Sequential(
        nn.Linear(state_size, 400),
        activ1(),
        nn.Linear(400, 300),
        activ1(),
        nn.Linear(300, 1),
        activ1()
    )


class Critic(nn.Module):
    def __init__(self, state_size, act_size):
        super(Critic, self).__init__()
        self.fc_1 = nn.Linear(state_size + act_size, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, 1)

    def forward(self, state, actor):
        x = torch.cat([state, actor], 1)
        x = self.fc_1(x)
        x = activ2(x)
        x = self.fc_2(x)
        x = activ2(x)
        x = self.fc_3(x)
        return x


def play(env, policy, steps=200, hold=0.03):
    s = env.reset()
    sum_r = 0
    for i in range(steps):
        a = policy(Tensor(s).to(device)).cpu().detach().cpu().numpy()
        s, r, done, _ = env.step(a * 2)
        env.render()
        time.sleep(hold)
        sum_r += r
    return sum_r / steps


if __name__ == '__main__':
    print(device)

    state_size = 3
    act_size = 1
    mu = Actor(state_size).to(device)
    mu_ = Actor(state_size).to(device)
    copy(mu, mu_)
    Q = Critic(state_size, act_size).to(device)
    Q_ = Critic(state_size, act_size).to(device)
    copy(Q, Q_)
    random_process = OrnsteinUhlenbeckProcess(dt=1, theta=.15, sigma=0.2)
    recorder = Recorder('r', 'a')
    optim_critic = optim.Adam(Q.parameters(), lr=lr_critic)
    optim_actor = optim.Adam(mu.parameters(), lr=lr_actor)
    try:
        for episode in range(M):
            random_process.reset()
            s_1 = env.reset() / env.observation_space.high  # normalize state to [-1, 1]
            s_t = s_1
            if episode % 200 == 200 - 1:
                recorder.plot('{}'.format(name))
                play(env, mu)
            for t in range(T):
                a_t = mu(Tensor(s_t).to(device)).cpu().detach().numpy() + np.array(random_process())
                s_t_1, r_t, done, _ = env.step(a_t * 2)
                s_t_1 /= env.observation_space.high  # normalize state to [-1, 1]

                replay_buffer.append((s_t, a_t, [r_t], s_t_1))
                recorder(r_t, a_t[0])
                s_t = s_t_1

                if done:
                    break
                if len(replay_buffer) < N:
                    continue

                batch = zip(*sample(replay_buffer, N))
                s_i, a_i, r_i, s_i_1 = [np.array(i) for i in batch]
                r_i = (r_i - 6) / 6  # normalize reward to around [-1, 1]

                optim_critic.zero_grad()
                y_i = Tensor(r_i).to(device) + gamma * Q_(Tensor(s_i_1).to(device), mu_(Tensor(s_i_1).to(device)))
                loss_critic = F.mse_loss(y_i, Q(Tensor(s_i).to(device), Tensor(a_i).to(device)))
                loss_critic.backward()
                optim_critic.step()

                optim_actor.zero_grad()
                loss_actor = -Q(Tensor(s_i).to(device), mu(Tensor(s_i).to(device))).mean()
                loss_actor.backward()
                optim_actor.step()

                tau_move_average(Q, Q_, tau)
                tau_move_average(mu, mu_, tau)

            recorder.reset()
    except KeyboardInterrupt:
        pass

    recorder.reset()
    pickle.dump(mu.to('cpu').state_dict(), open('{}.pkl'.format(name), 'wb'))
    recorder.plot('{}'.format(name))
    recorder.save('{}.json'.format(name))
