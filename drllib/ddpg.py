#!/usr/bin/env python3
from collections import deque
from random import sample

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from drllib.lib import copy, get_device, tao_move_average
from torch import Tensor


class DDPG(object):
    def __init__(
        self, Actor, Critic,
        random_process,
        replay_buffer_size=1e+6,
        lr_critic=1e-3,
        lr_actor=1e-4,
        gamma=0.99,
        tao=0.001,
        rnn_reset_function_name='reset',
        device=None,
    ):
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.tao = tao
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))
        self.random_process = random_process
        self.rnn_reset_function_name = rnn_reset_function_name

        self.mu = Actor().to(self.device)
        self.mu_ = Actor().to(self.device)
        copy(self.mu, self.mu_)
        self.Q = Critic().to(self.device)
        self.Q_ = Critic().to(self.device)
        copy(self.Q, self.Q_)
        self.optim_critic = optim.Adam(self.Q.parameters(), lr=lr_critic)
        self.optim_actor = optim.Adam(self.mu.parameters(), lr=lr_actor)

    def observe(self, state, action, reward):
        self.replay_buffer.append((self.prev_state, action, [reward], state))
        self.prev_state = state

    def plan(self, state, disturb=False):
        predict = self.mu(Tensor(state).to(self.device)).cpu().detach().numpy()
        if disturb:
            predict += np.array(self.random_process())
        return predict

    def train(self, batch_size=64):
        batch = zip(*sample(self.replay_buffer, batch_size))
        s_i, a_i, r_i, s_i_1 = [np.array(i) for i in batch]

        self.optim_critic.zero_grad()
        y_i = Tensor(r_i).to(self.device) + self.gamma * self.Q_(Tensor(s_i_1).to(self.device), self.mu_(Tensor(s_i_1).to(self.device)))
        loss_critic = F.mse_loss(y_i, self.Q(Tensor(s_i).to(self.device), Tensor(a_i).to(self.device)))
        loss_critic.backward()
        self.optim_critic.step()

        self.optim_actor.zero_grad()
        loss_actor = -self.Q(Tensor(s_i).to(self.device), self.mu(Tensor(s_i).to(self.device))).mean()
        loss_actor.backward()
        self.optim_actor.step()

        tao_move_average(self.Q, self.Q_, self.tao)
        tao_move_average(self.mu, self.mu_, self.tao)
        return loss_actor, loss_critic

    def reset(self, prev_state):
        self.random_process.reset()
        self.prev_state = prev_state
        # reset rnn hidden unit
        for network in [self.mu, self.mu_, self.Q, self.Q_]:
            reset_function = getattr(network, self.rnn_reset_function_name, None)
            if reset_function is not None:
                reset_function()

    def save(self, path=None, writer=None):
        if writer is not None:
            path = writer.file_writer.get_logdir() + '/model.pkl'
        assert path is not None, 'save path cannot be None'
        torch.save((self.Q), path)

    def load(self, path):
        (self.Q, self.Q_, self.mu, self.mu_) = torch.load(path)
