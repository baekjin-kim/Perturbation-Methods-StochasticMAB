import numpy as np
import scipy
import matplotlib.pyplot as plt
%matplotlib inline
import os

##########################################################################################
#UCB1 algorithm
##########################################################################################
class ucb_bandit:
    def __init__(self, k, c, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n))    
        reward = reward_generate(self.mu[a])
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = np.max(self.mu)-self.mean_reward
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)
   
##########################################################################################   
#FTPL algorithm via Gaussian perturbation
##########################################################################################
class RCB_gaussian:
    def __init__(self, k, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        #self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        # Select action
        a = np.argmax(self.k_reward + np.random.normal(0, np.sqrt(2), self.k) / np.sqrt(self.k_n))
        reward = reward_generate(self.mu[a])
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = np.max(self.mu)-self.mean_reward
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)

##########################################################################################
#FTPL algorithm via Uniform perturbation   
##########################################################################################
class RCB_uniform:
    def __init__(self, k, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        #self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + np.random.uniform(-1, 1, k) * np.sqrt(2.1*(np.log(self.n)) / self.k_n))
        reward = reward_generate(self.mu[a])
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = np.max(self.mu)-self.mean_reward
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)

##########################################################################################
#FTPL algorithm via Rademacher perturbation     
##########################################################################################
  class RCB_rademacher:
    def __init__(self, k, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        #self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + (2* np.random.binomial(1,0.5,k)-1) * np.sqrt(2.1*(np.log(self.n)) / self.k_n))
        reward = reward_generate(self.mu[a])
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = np.max(self.mu)-self.mean_reward
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)

##########################################################################################
#FTPL algorithm via Double Exponential perturbation  
##########################################################################################
  class RCB_DE:
    def __init__(self, k, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        #self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
        
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + np.random.laplace(0, np.sqrt(2), self.k) / np.sqrt(self.k_n))
        reward = reward_generate(self.mu[a])
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = np.max(self.mu)-self.mean_reward
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)
            
##########################################################################################
#Gaussian reward generation
##########################################################################################
def reward_generate(mean):
    return np.random.normal(mean,0,1)

k=10
iters = 10000
# Initialize bandits
ucb = ucb_bandit(k, 2, iters, mu='random')
gaussian = RCB_gaussian(k,iters, mu=ucb.mu)
rademacher = RCB_rademacher(k,iters, mu=ucb.mu)
uniform= RCB_uniform(k,iters, mu=ucb.mu)
laplace= RCB_DE(k,iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)
opt_ucb = 0
opt_gaussian = 0
opt_uniform = 0
opt_laplace = 0
opt_rademacher = 0

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
    
    # Count optimal actions
    opt_ucb += ucb.k_n[np.argmax(ucb.mu)]
    opt_gaussian += gaussian.k_n[np.argmax(ucb.mu)]
    opt_uniform += uniform.k_n[np.argmax(ucb.mu)]
    opt_laplace += laplace.k_n[np.argmax(ucb.mu)]
    opt_rademacher += rademacher.k_n[np.argmax(ucb.mu)]
    
fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.3)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.getcwd()
os.chdir('/Users/baekjin/Desktop/Box/paper/NeuIPS2019_version/image')
fig.savefig("Gaussian_reward.png")

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.ylim(0, 0.1)
plt.show()
fig.savefig("Gaussian_reward2.png")

##########################################################################################
#Uniform reward generation
##########################################################################################
def reward_generate(mean):
    return np.random.uniform(-1, 1, 1) + mean

k=10
iters = 10000
# Initialize bandit
ucb = ucb_bandit(k, 2, iters, mu='random')
gaussian = RCB_gaussian(k,iters, mu=ucb.mu)
rademacher = RCB_rademacher(k,iters, mu=ucb.mu)
uniform= RCB_uniform(k,iters, mu=ucb.mu)
laplace= RCB_DE(k,iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)
opt_ucb = 0
opt_gaussian = 0
opt_uniform = 0
opt_laplace = 0
opt_rademacher = 0

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
    
    # Count optimal actions
    opt_ucb += ucb.k_n[np.argmax(ucb.mu)]
    opt_gaussian += gaussian.k_n[np.argmax(ucb.mu)]
    opt_uniform += uniform.k_n[np.argmax(ucb.mu)]
    opt_laplace += laplace.k_n[np.argmax(ucb.mu)]
    opt_rademacher += rademacher.k_n[np.argmax(ucb.mu)]
    
fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.3)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.getcwd()
os.chdir('/Users/baekjin/Desktop/Box/paper/NeuIPS2019_version/image')
fig.savefig("Uniform_reward.png")

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.ylim(0, 0.1)
plt.show()
fig.savefig("Uniform_reward2.png")

##########################################################################################
# Guassian Mixture reward generation
##########################################################################################
def reward_generate(mean):
    mu = [-1, 1]
    sigma = [1, np.sqrt(3)]
    Z = np.random.choice([0,1])
    return np.random.normal(mu[Z], sigma[Z], 1) + mean

k=10
iters = 10000
# Initialize bandits
ucb = ucb_bandit(k, 2, iters, mu='random')
gaussian = RCB_gaussian(k,iters, mu=ucb.mu)
rademacher = RCB_rademacher(k,iters, mu=ucb.mu)
uniform= RCB_uniform(k,iters, mu=ucb.mu)
laplace= RCB_DE(k,iters, mu=ucb.mu)

gaussian.mu = ucb.mu
uniform.mu = ucb.mu
laplace.mu = ucb.mu
rademacher.mu = ucb.mu

ucb_rewards = np.zeros(iters)
gaussian_rewards = np.zeros(iters)
uniform_rewards = np.zeros(iters)
laplace_rewards = np.zeros(iters)
rademacher_rewards = np.zeros(iters)
opt_ucb = 0
opt_gaussian = 0
opt_uniform = 0
opt_laplace = 0
opt_rademacher = 0

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
    
    # Count optimal actions
    opt_ucb += ucb.k_n[np.argmax(ucb.mu)]
    opt_gaussian += gaussian.k_n[np.argmax(ucb.mu)]
    opt_uniform += uniform.k_n[np.argmax(ucb.mu)]
    opt_laplace += laplace.k_n[np.argmax(ucb.mu)]
    opt_rademacher += rademacher.k_n[np.argmax(ucb.mu)]
    
fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.ylim(0, 0.3)
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.show()
os.getcwd()
os.chdir('/Users/baekjin/Desktop/Box/paper/NeuIPS2019_version/image')
fig.savefig("Mixture_reward_neq.png")

fig = plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB1")
plt.plot(uniform_rewards, label="RCB-Uniform")
plt.plot(rademacher_rewards, label="RCB-Rademacher")
plt.plot(gaussian_rewards, label="FTPL-Gaussian(Gaussian TS)")
plt.plot(laplace_rewards, label="FTPL-Double Exponential")
plt.legend(loc=1)
plt.xlabel("Iterations")
plt.ylabel("Average Regret")
plt.title("Average Regret for Bandit Algorithms after " 
          + str(episodes) + " Episodes")
plt.ylim(0, 0.1)
plt.show()
fig.savefig("Mixture_reward_neq2.png")
