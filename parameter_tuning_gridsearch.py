#####################################
# Uniform Reward 
#####################################

k=10
iters = 30000

#Grid search for parameter tuning
grid = np.arange(1,21)/20
regret_grid = np.zeros(20)
episodes = 1000

#Uniform reward setting
def reward_generate(mean):
    return np.random.uniform(-1, 1, 1) + mean

#UCB
for j in range(20):
    ucb = ucb_bandit(k, grid[j], iters, mu='random')
    ucb_rewards = 0
    for i in range(episodes):
        random.seed(i)
        ucb.reset('random')
        ucb.run()
        ucb_rewards = ucb_rewards + (ucb.reward[iters-1] - ucb_rewards) / (i + 1)
    regret_grid[j] = ucb_rewards
print("The optimal tuning parameter for UCB is" + grid[np.argmin(ucb_grid)])

#FTPL-Gaussian
for j in range(20):
    gaussian = RCB_gaussian(k, grid[j], iters, mu='random')
    gaussian_rewards = 0
    for i in range(episodes):
        random.seed(i)
        gaussian.reset('random')
        gaussian.run()
        gaussian_rewards = gaussian_rewards + (gaussian.reward[iters-1] - gaussian_rewards) / (i + 1)
    regret_grid[j] = gaussian_rewards
print("The optimal tuning parameter for FTPL-Gaussian is" + grid[np.argmin(regret_grid)])

#FTPL-Double Exponential
for j in range(20):
    laplace = RCB_DE(k, grid[j], iters, mu='random')
    laplace_rewards = 0
    for i in range(episodes):
        random.seed(i)
        laplace.reset('random')
        laplace.run()
        laplace_rewards = laplace_rewards + (laplace.reward[iters-1] - laplace_rewards) / (i + 1)
    regret_grid[j] = laplace_rewards
print("The optimal tuning parameter for FTPL-DE is" + grid[np.argmin(regret_grid)])

#RCB-Uniform
for j in range(20):
    uniform = RCB_uniform(k, grid[j], iters, mu='random')
    uniform_rewards = 0
    for i in range(episodes):
        random.seed(i)
        uniform.reset('random')
        uniform.run()
        uniform_rewards = uniform_rewards + (uniform.reward[iters-1] - uniform_rewards) / (i + 1)
    regret_grid[j] = uniform_rewards
print("The optimal tuning parameter for RCB-Uniform is" + grid[np.argmin(regret_grid)])

#RCB-Rademacher
for j in range(20):
    rademacher = RCB_rademacher(k, grid[j], iters, mu='random')
    rademacher_rewards = 0
    for i in range(episodes):
        random.seed(i)
        rademacher.reset('random')
        rademacher.run()
        rademacher_rewards = rademacher_rewards + (rademacher.reward[iters-1] - rademacher_rewards) / (i + 1)
    regret_grid[j] = rademacher_rewards
print("The optimal tuning parameter for RCB-Rademacher is" + grid[np.argmin(regret_grid)])
