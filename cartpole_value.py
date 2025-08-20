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
                        ('state', 'action', 'next_state', 'reward'))


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
        return self.layer3(x)


def select_action(state, policy_net, eps):
    rand_x = random.random()
    if rand_x < eps:
        action = np.random.randint(low=0, high=2)
    else:
        state = state.unsqueeze(0)
        with torch.no_grad():
            action_prob = policy_net(state)
        action = torch.argmax(action_prob).item()

    return action


def replay_memory(env, policy_net, memory, params, trajectory_numb = 1000, n_steps=500):
    device = params["device"]
    for trajectory in range(trajectory_numb):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
 
        for steps in range(n_steps):
            state = state.to(device)  
            action = select_action(state, policy_net, eps=0.5)
            state= state.cpu()
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward]) #, device=device
            action = torch.tensor([action])
            done = terminated or truncated
            if terminated:
                #reward = torch.zeros(1) #my code
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)#device=device
            
            memory.push(state, action, next_state, reward)
            state = next_state
            if done:
                break


def optimize_model(memory, policy_net, policy_net_optimizer, target_net, parameters):
    device = params["device"]
    gamma = parameters["gamma"]
    batch_size = parameters["batch_size"]
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool) #device=device, 
    non_final_mask = non_final_mask.to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    non_final_next_states = non_final_next_states.to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values_vect = policy_net(state_batch)#!!!!.gather(1, action_batch)
    action = torch.vstack([action_batch, action_batch]).T
    state_action_values = state_action_values_vect.gather(1, action)[:,1]

    next_state_values = torch.zeros(batch_size, device=device).to(device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    policy_net_optimizer.zero_grad()
    loss.backward()
    policy_net_optimizer.step()

    return loss.item(), state_action_values.mean().item()


def soft_update(local_model, target_model, tau=0.05):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def validation(env, policy_net, params, max_steps=500):
    device = params["device"]
    n_steps_list = []
    for trajectory in range(300):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
 
        for steps in range(max_steps):
            state = state.to(device)  
            action = select_action(state, policy_net, eps=0.0)
            state= state.cpu()
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward]) #, device=device
            action = torch.tensor([action])
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)#device=device
            
            state = next_state
            if done:
                break
        n_steps_list.append(steps)

    avarage_steps = sum(n_steps_list) / len(n_steps_list)
    return avarage_steps


if __name__ == "__main__":
        max_steps = 1000
        env = gym.make(id='CartPole-v1', render_mode="rgb_array", max_episode_steps=max_steps) #render_mode ansi
        params = {"gamma":0.99, "lr":0.0001, "tau":0.05, "eps":0.2,
                   "batch_size":160,  "device":"cuda:1"}

        policy_net = DQN(4, 2)
        #policy_net.load_state_dict(torch.load("weights/cartpole.pth"))
        policy_net.to(params["device"])
        target_net = DQN(4, 2)
        #target_net.load_state_dict(torch.load("weights/cartpole_500R.pth"))
        target_net.to(params["device"])
        # action_net = DQN(4, 2)
        # action_net.to(params["device"])
     
        policy_net_optimizer = t_opt.AdamW(policy_net.parameters(), lr=0.0001)
        memory = ReplayMemory(30000)
        replay_memory(env, policy_net, memory, params, trajectory_numb=10000)
        policy_net.load_state_dict(torch.load("weights/cartpole_500R.pth"))
        replay_memory(env, policy_net, memory, params, trajectory_numb=3000)
        # policy_net.load_state_dict(torch.load("weights/cartpole_600L.pth"))
        # replay_memory(env, policy_net, memory, params, trajectory_numb=3000)
        max_steps_achieved = 0

        for epoch in range(100):
            loss_list = []
            reward_list = []

            steps_in_epoch = 800
            for i in range(steps_in_epoch):
                loss, reward = optimize_model(memory, policy_net, policy_net_optimizer, target_net, params)
                loss_list.append(loss)
                reward_list.append(reward)
            
            loss_average = sum(loss_list) / len(loss_list)
            average_number_of_steps = validation(env, policy_net, params, max_steps=max_steps)
            avarage_reward = sum(reward_list) / len(reward_list)
            print("{:d} loss: {:.4f}, reward: {:.4f}, val steps: {:.2f}".format(epoch, loss_average, avarage_reward, average_number_of_steps))
            if average_number_of_steps > max_steps_achieved:
                torch.save(policy_net.state_dict(), "weights/cartpole.pth")
            
            if epoch % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                #soft_update(policy_net, target_net)
            if epoch > 10 and average_number_of_steps > 500:
                replay_memory(env, policy_net, memory, params, trajectory_numb=20)

        # n_space = env.observation_space
        # n_action = env.action_space
        # aaa = 777
        # state, info = env.reset()
        # rgb_array = env.render()
        # plt.imshow(rgb_array)
        # plt.savefig("cartpole.jpg", bbox_inches="tight")
        # action = 0
        # observation, reward, terminated, truncated, info = env.step(action)
        # aaa = 666