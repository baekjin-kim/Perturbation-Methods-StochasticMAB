import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

##########################################################################################
#Gaussian reward generation
##########################################################################################
def reward_generate(mean):
    return np.random.normal(mean, 0, 1)

k=10
iters = 30000
# Initialize bandits
ucb = ucb_bandit(k, 0.35, iters, mu='random')
gaussian = RCB_gaussian(k, 0.25, iters, mu=ucb.mu)
rademacher = RCB_rademacher(k, 0.2, iters, mu=ucb.mu)
uniform= RCB_uniform(k, 0.25, iters, mu=ucb.mu)
laplace= RCB_DE(k, 0.05, iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)

episodes = 1000
# Run experiments
for i in range(episodes):
    # Reset counts and rewards
    ucb.reset('random')
    gaussian.reset()
    uniform.reset()
    laplace.reset()
    rademacher.reset()
    
    gaussian.mu = ucb.mu
    uniform.mu = ucb.mu
    laplace.mu = ucb.mu
    rademacher.mu = ucb.mu
    
    # Run experiments
    ucb.run()
    gaussian.run()
    uniform.run()
    laplace.run()
    rademacher.run()
    
    # Update long-term averages
    gaussian_rewards = gaussian_rewards + (
        gaussian.reward - gaussian_rewards) / (i + 1)
    ucb_rewards = ucb_rewards + (
        ucb.reward - ucb_rewards) / (i + 1)
    laplace_rewards = laplace_rewards + (
        laplace.reward - laplace_rewards) / (i + 1)
    uniform_rewards = uniform_rewards + (
        uniform.reward - uniform_rewards) / (i + 1)
    rademacher_rewards = rademacher_rewards + (
        rademacher.reward - rademacher_rewards) / (i + 1)

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.05)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.getcwd()
os.chdir('/Users/paper/NeuIPS2019_version/image')
fig.savefig("Gaussian_reward.png")

##########################################################################################
#Rademacher reward generation
##########################################################################################
def reward_generate(mean):
    return 2* np.random.binomial(1,0.5,k)-1 + mean

k=10
iters = 30000
# Initialize bandit
ucb = ucb_bandit(k, 0.55, iters, mu='random')
gaussian = RCB_gaussian(k, 0.45, iters, mu=ucb.mu)
rademacher = RCB_rademacher(k, 0.45, iters, mu=ucb.mu)
uniform= RCB_uniform(k, 0.5, iters, mu=ucb.mu)
laplace= RCB_DE(k, 0.15, iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)

episodes = 1000
# Run experiments
for i in range(episodes):
    # Reset counts and rewards
    ucb.reset('random')
    gaussian.reset()
    uniform.reset()
    laplace.reset()
    rademacher.reset()
    
    gaussian.mu = ucb.mu
    uniform.mu = ucb.mu
    laplace.mu = ucb.mu
    rademacher.mu = ucb.mu
    
    # Run experiments
    ucb.run()
    gaussian.run()
    uniform.run()
    laplace.run()
    rademacher.run()
    
    # Update long-term averages
    gaussian_rewards = gaussian_rewards + (
        gaussian.reward - gaussian_rewards) / (i + 1)
    ucb_rewards = ucb_rewards + (
        ucb.reward - ucb_rewards) / (i + 1)
    laplace_rewards = laplace_rewards + (
        laplace.reward - laplace_rewards) / (i + 1)
    uniform_rewards = uniform_rewards + (
        uniform.reward - uniform_rewards) / (i + 1)
    rademacher_rewards = rademacher_rewards + (
        rademacher.reward - rademacher_rewards) / (i + 1)

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.05)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.chdir('/Users/paper/NeuIPS2019_version/image')
fig.savefig("Rademacher_reward.png")

##########################################################################################
#Uniform reward generation
##########################################################################################
def reward_generate(mean):
    return np.random.uniform(-1, 1, 1) + mean

k=10
iters = 50000
# Initialize bandit
ucb = ucb_bandit(k, 0.85, iters, mu='random')
gaussian = RCB_gaussian(k, 0.45, iters, mu=ucb.mu)
rademacher = RCB_rademacher(k, 0.6, iters, mu=ucb.mu)
uniform= RCB_uniform(k, 0.7, iters, mu=ucb.mu)
laplace= RCB_DE(k, 0.2, iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)

episodes = 1000
# Run experiments
for i in range(episodes):
    # Reset counts and rewards
    ucb.reset('random')
    gaussian.reset()
    uniform.reset()
    laplace.reset()
    rademacher.reset()
    
    gaussian.mu = ucb.mu
    uniform.mu = ucb.mu
    laplace.mu = ucb.mu
    rademacher.mu = ucb.mu
    
    # Run experiments
    ucb.run()
    gaussian.run()
    uniform.run()
    laplace.run()
    rademacher.run()
    
    # Update long-term averages
    gaussian_rewards = gaussian_rewards + (
        gaussian.reward - gaussian_rewards) / (i + 1)
    ucb_rewards = ucb_rewards + (
        ucb.reward - ucb_rewards) / (i + 1)
    laplace_rewards = laplace_rewards + (
        laplace.reward - laplace_rewards) / (i + 1)
    uniform_rewards = uniform_rewards + (
        uniform.reward - uniform_rewards) / (i + 1)
    rademacher_rewards = rademacher_rewards + (
        rademacher.reward - rademacher_rewards) / (i + 1)

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.1)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.chdir('/Users/paper/NeuIPS2019_version/image')
fig.savefig("Uniform_reward.png")

##########################################################################################
# Guassian Mixture reward generation
##########################################################################################
def reward_generate(mean):
    mu = [-1, 1]
    sigma = [1, 1]
    Z = np.random.choice([0,1])
    return np.random.normal(mu[Z], sigma[Z], 1) + mean

k=10
iters = 100000
# Initialize bandits
ucb = ucb_bandit(k, 1.4, iters, mu='random')
gaussian = RCB_gaussian(k, 1, iters, mu=ucb.mu)
rademacher = RCB_rademacher(k, 1, iters, mu=ucb.mu)
uniform= RCB_uniform(k, 1.4, iters, mu=ucb.mu)
laplace= RCB_DE(k, 0.3, iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)

episodes = 1000
# Run experiments
for i in range(episodes):
    # Reset counts and rewards
    ucb.reset('random')
    gaussian.reset()
    uniform.reset()
    laplace.reset()
    rademacher.reset()
    
    gaussian.mu = ucb.mu
    uniform.mu = ucb.mu
    laplace.mu = ucb.mu
    rademacher.mu = ucb.mu
    
    # Run experiments
    ucb.run()
    gaussian.run()
    uniform.run()
    laplace.run()
    rademacher.run()
    
    # Update long-term averages
    gaussian_rewards = gaussian_rewards + (
        gaussian.reward - gaussian_rewards) / (i + 1)
    ucb_rewards = ucb_rewards + (
        ucb.reward - ucb_rewards) / (i + 1)
    laplace_rewards = laplace_rewards + (
        laplace.reward - laplace_rewards) / (i + 1)
    uniform_rewards = uniform_rewards + (
        uniform.reward - uniform_rewards) / (i + 1)
    rademacher_rewards = rademacher_rewards + (
        rademacher.reward - rademacher_rewards) / (i + 1)

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.1)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.chdir('/Users/paper/NeuIPS2019_version/image')
fig.savefig("Mixture_reward.png")

