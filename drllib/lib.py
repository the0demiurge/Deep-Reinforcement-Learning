
import torch
from random import randint, gauss


def get_device():
    return torch.device("cuda:{}".format(randint(0, torch.cuda.device_count() - 1)) if torch.cuda.is_available() else "cpu")


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
