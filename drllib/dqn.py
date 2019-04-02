#!/usr/bin/env python3
from collections import deque
from random import sample, random, randint

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from drllib.lib import get_device
from torch import Tensor


class DQN(object):
    def __init__(
        self, Q_Network,
        replay_buffer_size=1e+6,
        lr=1e-3,
        gamma=0.99,
        rnn_reset_function_name='reset',
        device=None,
    ):
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.lr = lr
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))
        self.rnn_reset_function_name = rnn_reset_function_name

        self.Q = Q_Network().to(self.device)
        self.optimizor = optim.Adam(self.Q.parameters(), lr=lr)

    def observe(self, state, action, reward):
        self.replay_buffer.append((self.prev_state, [action], [reward], state))
        self.prev_state = state

    def plan(self, state, action_nums=None, epsilon=None):
        if epsilon is not None and random() < epsilon:
            predict = np.array(randint(0, action_nums - 1))
        else:
            predict = torch.argmax(self.Q(Tensor(state).to(self.device))).cpu().detach().numpy()
        return predict

    def train(self, batch_size=64):
        batch = zip(*sample(self.replay_buffer, batch_size))
        s_i, a_i, r_i, s_i_1 = [np.array(i) for i in batch]

        self.optimizor.zero_grad()
        y_i = Tensor(r_i).to(self.device) + Tensor(r_i).to(self.device) * (self.gamma * torch.max(self.Q(Tensor(s_i_1).to(self.device)), dim=1)[0].view(-1, 1))
        loss = F.mse_loss(y_i, self.Q(Tensor(s_i).to(self.device))[list(range(batch_size)), a_i.squeeze()].view(-1, 1))
        loss.backward()
        self.optimizor.step()

        return loss

    def reset(self, prev_state):
        self.prev_state = prev_state
        # reset rnn hidden unit
        for network in [self.Q]:
            reset_function = getattr(network, self.rnn_reset_function_name, None)
            if reset_function is not None:
                reset_function()

    def dump(self, path):
        torch.save((self.Q), path)

    def load(self, path):
        (self.Q) = torch.load(path)
