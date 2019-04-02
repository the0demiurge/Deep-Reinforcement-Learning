import gym
import torch.nn as nn
from drllib import DQN
from tensorboardX import SummaryWriter

name = 'dqn_cartpole'
state_size = 4
act_size = 2

epsilon = 0.1  # Exploration constant
M = 2000  # Episodes
T = 200  # Trajectory lenth
N = 64  # Batch size

writer = SummaryWriter(comment='_' + name)
env = gym.make('CartPole-v0')


def Q_Network(state_size, act_size):
    return nn.Sequential(
        nn.Linear(state_size, 400),
        nn.RReLU(),
        nn.Linear(400, 300),
        nn.RReLU(),
        nn.Linear(300, act_size),
        nn.RReLU()
    )


agent = DQN(lambda: Q_Network(state_size, act_size))


try:
    for episode in range(M):
        state = env.reset()
        agent.reset(state)
        length = 0
        for t in range(T):
            action = agent.plan(state, 2, epsilon)
            state, reward, done, _ = env.step(action)

            if done:
                reward = 0

            length += reward

            agent.observe(state, action, reward)

            if done:
                break
            if len(agent.replay_buffer) < N:
                continue

            agent.train(N)
        writer.add_scalar('live_length', length, global_step=episode)

except KeyboardInterrupt:
    pass
agent.save(name + '.pkl')
