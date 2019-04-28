# Initialization

Initialization of DRL agent has a great influence on the training result and convergence speed, because the initialization determines the initial policy of the agent and the data used in the agent training, which as also one of the source of DRL’s training instability. Some initialization would make the agent never come up with a appropriate policy.

# Convergence Speed

Convergence speed is highly influenced by the data the agent received to train and the data use efficiency. To improve the data quality, tips 1 to 4 should be considered; to import data use efficiency, tips 5 to 7 should be followed.

# Tips

1. Multi thread training with different initialization and a shared/semi-shared/not-shared replay buffer would avoid the influence of initialization.
2. The agent may receive a lot of training data in replay buffer with random policy or cross entropy method policy, or just some initial vanilla policy, which is to say, pre-training, or guided policy search.
3. Add some human control data to the replay buffer, that is, imitate learning.
4. Pre-train the agent with human control data by supervised learning.
5. Choose better DRL algorithm.
6. Choose proper data in replay buffer. For example, use stratified sampling by episode average reward/episode length; make sure the replay buffer not be filled by useless and duplicated episodes.
7. Make sure the neural network takes enough gradient descent steps during training.

