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

random_process = OrnsteinUhlenbeckProcess(dt=1, theta=.15, sigma=0.2)
writer = SummaryWriter(comment='_' + name)


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


env = gym.make('Pendulum-v0')

agent = DDPG(lambda: Actor(state_size), lambda: Critic(state_size, act_size), random_process)
try:
    for episode in range(2000):
        random_process.reset()
        s_1 = env.reset() / env.observation_space.high  # normalize state to [-1, 1]
        agent.reset(s_1)
        s_t = s_1
        rr = 0
        for t in range(200):
            a_t = agent.plan(s_t, True)
            s_t_1, r_t, done, _ = env.step(a_t * 2)
            s_t_1 /= env.observation_space.high  # normalize state to [-1, 1]
            rr += r_t
            agent.observe(s_t_1, a_t, r_t / 6)
            s_t = s_t_1

            if done:
                break
            if len(agent.replay_buffer) < 64:
                continue
            agent.train()
        writer.add_scalar('average_reward', rr / t)

except KeyboardInterrupt:
    pass
agent.save(name + '.pkl')
