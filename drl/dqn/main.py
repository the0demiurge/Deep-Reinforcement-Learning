#!/usr/bin/env python3
import json
import pickle
import time
from collections import deque
from random import randint, sample, random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import Tensor

name = 'baseline'
device = 'cuda:0'  # torch.device("cuda:{}".format(randint(0, torch.cuda.device_count() - 1)) if torch.cuda.is_available() else "cpu")

M = 2000  # Episodes
T = 200  # Trajectory lenth
N = 64  # Batch size
R = 1e+6  # Replay size
epsilon = 0.1  # Exploration constant
lr_critic = 1e-3  # Initial_learning rates
lr_actor = 1e-4
gamma = 0.99
tao = 0.001
activ1 = nn.RReLU
activ2 = F.rrelu

env = gym.make('CartPole-v0')
replay_buffer = deque(maxlen=int(R))


def copy(source, destination):
    destination.load_state_dict(source.state_dict())


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
            plt.plot(self.moving_average(self.stats[arg]['max']), 'x-', label=arg, linewidth=0.5, markersize=1)
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


def Q_Network(state_size, act_size):
    return nn.Sequential(
        nn.Linear(state_size, 400),
        activ1(),
        nn.Linear(400, 300),
        activ1(),
        nn.Linear(300, act_size),
        activ1()
    )


def play(env, Q, steps=200, hold=0.03):
    s = env.reset()
    sum_r = 0
    for i in range(steps):
        a = torch.argmax(Q(Tensor(s).to(device))).cpu().detach().numpy()
        s, r, done, _ = env.step(a)
        env.render()
        time.sleep(hold)
        sum_r += r
    return sum_r / steps


if __name__ == '__main__':
    print(device)

    state_size = 4
    act_size = 2

    Q = Q_Network(state_size, act_size).to(device)

    recorder = Recorder('t', 'a')
    optim = optim.Adam(Q.parameters(), lr=lr_critic)

    try:
        for episode in range(M):
            s_1 = env.reset()
            s_t = s_1
            if episode % 20 == 20 - 1:
                recorder.plot('{}'.format(name))
                # play(env, Q)
            for t in range(T):
                if random() < epsilon:
                    a_t = np.array(env.action_space.sample())  # choose a random step
                else:
                    a_t = torch.argmax(Q(Tensor(s_t).to(device))).cpu().detach().numpy()
                s_t_1, r_t, done, _ = env.step(a_t)

                replay_buffer.append((s_t, [a_t], [r_t], s_t_1))
                recorder(t, int(a_t))
                s_t = s_t_1

                if done:
                    replay_buffer[-1][2][0] = np.array(0)
                    break
                if len(replay_buffer) < N:
                    continue

                batch = zip(*sample(replay_buffer, N))
                s_i, a_i, r_i, s_i_1 = [np.array(i) for i in batch]

                optim.zero_grad()
                y_i = Tensor(r_i).to(device) + Tensor(r_i).to(device) * (gamma * torch.max(Q(Tensor(s_i_1).to(device)), dim=1)[0].view(-1, 1))
                loss = F.mse_loss(y_i, Q(Tensor(s_i).to(device))[list(range(N)), a_i.squeeze()].view(-1, 1))
                loss.backward()
                optim.step()

            recorder.reset()
    except KeyboardInterrupt:
        pass

    recorder.reset()
    pickle.dump(Q.to('cpu').state_dict(), open('{}.pkl'.format(name), 'wb'))
    recorder.plot('{}'.format(name))
    recorder.save('{}.json'.format(name))
