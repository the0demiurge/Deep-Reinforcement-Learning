import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from drllib import DDPG
from drllib.lib import OrnsteinUhlenbeckProcess
from tensorboardX import SummaryWriter

name = 'ddpg_pendulum'
state_size = 3
act_size = 1

M = 2000  # Episodes
T = 200  # Trajectory lenth
N = 64  # Batch size


def Actor(state_size):
    return nn.Sequential(
        nn.Linear(state_size, 400),
        nn.RReLU(),
        nn.Linear(400, 300),
        nn.RReLU(),
        nn.Linear(300, 1),
        nn.RReLU()
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
        x = F.rrelu(x)
        x = self.fc_2(x)
        x = F.rrelu(x)
        x = self.fc_3(x)
        return x


random_process = OrnsteinUhlenbeckProcess(dt=1, theta=.15, sigma=0.2)
agent = DDPG(lambda: Actor(state_size), lambda: Critic(state_size, act_size), random_process)
env = gym.make('Pendulum-v0')
writer = SummaryWriter(comment='_' + name)


try:
    for episode in range(M):
        state = env.reset() / env.observation_space.high  # normalize state to [-1, 1]
        agent.reset(state)
        accumulate_reward = 0
        for t in range(T):
            action = agent.plan(state, True)
            state, reward, done, _ = env.step(action * 2)
            state /= env.observation_space.high  # normalize state to [-1, 1]

            accumulate_reward += reward

            agent.observe(state, action, reward / 6)

            if done:
                break
            if len(agent.replay_buffer) < N:
                continue
            agent.train(N)
        writer.add_scalar('average_reward', accumulate_reward / t, global_step=episode)
        print(episode, accumulate_reward / t, sep='\t')

except KeyboardInterrupt:
    pass
agent.save(writer=writer)
