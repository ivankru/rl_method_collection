import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import numpy as np
from torch import nn
import torch
import torch.functional as F
import torch.optim as t_opt
from torch.distributions import Categorical

Transition = namedtuple('Transition',
                        ('state', 'action', 'log_prob', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.layer1(x)) 
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        x = torch.softmax(x, dim=0)
        return x


def select_action(state, policy_net):
    state = state.squeeze(0)
    #with torch.no_grad():
    action_prob = policy_net(state).cpu()
    random_distribution = Categorical(action_prob)
    action = random_distribution.sample()
    log_prob = random_distribution.log_prob(action)

    return action.item(), log_prob


def trajectory_generation(env, policy_net, params, max_trajectory_length=500):
    device = params["device"]
    trajectory = []
    total_reward = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state = state.unsqueeze(0)

    for i in range(max_trajectory_length):
        action, log_prob = select_action(state, policy_net)
        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)#device=device
        total_reward += reward
        trajectory.append(Transition(state, action, log_prob, next_state, reward))
        done = terminated or truncated
        state = next_state
        if done:
            break
    
    return trajectory, total_reward


def trajectory_optimization(trajectory, optimizer, baseline_reward_value, params):
    gamma = params["gamma"]
    cummulative_reward_list = []
    log_p_list = []
    cummulative_reward = 0

    for current_state in trajectory:
        log_p = current_state.log_prob
        log_p_list.append(log_p)
        r = current_state.reward
        cummulative_reward += r
        cummulative_reward_list.append(r)

    cummulative_discounted_reward = 0
    loss_list = []
    prob_list = []
    for cr, log_p in zip(reversed(cummulative_reward_list), reversed(log_p_list)):
        cummulative_discounted_reward = cr + gamma * cummulative_discounted_reward
        #normalized_reward is important for training stability. It is like advantage in actor-critic
        #normalized reward could have negative values
        normalized_reward = cummulative_discounted_reward - baseline_reward_value
        #after backward it would be grad_theta(Log(pi_theta(at|st)*R(tau)))
        loss_i = log_p.unsqueeze(0) * normalized_reward
        loss_list.append(loss_i)
        prob_list.append(log_p.item())
        #cummulative_discounted_reward_list.append(cummulative_discounted_reward) 
    
    loss =  - torch.cat(loss_list).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return -sum(prob_list) / len(prob_list)


if __name__ == "__main__":
    max_steps = 6000
    params = {"gamma":0.98, "lr":0.00005, "device":"cuda:1"}
    
    policy_net = DQN(4, 2)
    policy_net.load_state_dict(torch.load("weights/cartpole_policy.pth"))
    policy_net.to(params["device"])
    optimizer = t_opt.AdamW(policy_net.parameters(), lr=params["lr"])

    env = gym.make(id='CartPole-v1', render_mode="rgb_array", max_episode_steps=max_steps) #render_mode ansi

    traject_length = 1
    max_length_achieved = 0
    for epoch in range(60):
        loss_list = []
        traject_length_list = []
        #cumulative discounted reward for an average length trajectory
        #so it motivates to create trajectory with length bigger than average 
        baseline_reward_value = (1 - params["gamma"]**traject_length) / (1 - params["gamma"])

        for i in range(200):
            trajectory, total_reward = trajectory_generation(env, policy_net, params, max_trajectory_length=max_steps)
            loss = trajectory_optimization(trajectory, optimizer, baseline_reward_value, params)
            loss_list.append(loss)
            traject_length_list.append(len(trajectory))

        loss = sum(loss_list) / len(loss_list)
        traject_length = sum(traject_length_list) / len(traject_length_list)
        print("{:d} log_p: {:.4f}, val steps: {:.2f}".format(epoch, loss, traject_length))
        if traject_length > max_length_achieved:
                torch.save(policy_net.state_dict(), "weights/cartpole_policy.pth")
        max_length_achieved = max(traject_length, max_length_achieved)


