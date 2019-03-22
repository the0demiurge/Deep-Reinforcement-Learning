# Introduction
This code is referring to *CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING* with little structure modification.

# Experiment

- Changing all activation from random ReLU to tanh, makes network converge faster. However, after convergence, the network performs worse and worse while training.

# Experience

To reduce play episodes, it is important to make full use of data in the replay buffer. So, it is necessary to make sure that the network takes enough gradient descent steps with as few play episodes as possible. 