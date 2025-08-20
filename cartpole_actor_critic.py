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
                        ('state', 'action', 'next_state', 'log_p', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear_memory(self):
        del self.memory
        self.memory = deque([], maxlen=self.capacity)

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


def select_action(state, policy_net):
    state = state.squeeze(0)
    action_prob = policy_net(state).cpu()
    action_prob = torch.softmax(action_prob, dim=0)
    random_distribution = Categorical(action_prob)
    action = random_distribution.sample()
    log_prob = random_distribution.log_prob(action)
    
    return action.item(), log_prob


# def replay_memory(env, policy_net, memory, params, trajectory_numb = 1000, n_steps=500):
#     device = params["device"]
#     for trajectory in range(trajectory_numb):
#         state, info = env.reset()
#         state = torch.tensor(state, dtype=torch.float32).to(device)
#         state = state.unsqueeze(0)
 
#         for steps in range(n_steps):
#             state = state.to(device)  
#             action, log_p = select_action(state, policy_net)
#             state= state.cpu()
#             observation, reward, terminated, truncated, _ = env.step(action)
#             reward = torch.tensor([reward]) #, device=device
#             action = torch.tensor([action])
#             done = terminated or truncated
#             if terminated:
#                 #reward = torch.zeros(1) #my code
#                 next_state = None
#             else:
#                 next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)#device=device
            
#             memory.push(state, action, next_state, log_p.unsqueeze(0), reward)
#             state = next_state
#             if done:
#                 break


def critic_step(critic_net, state, next_state, reward, done, params):
    vt = critic_net(state) #Vt - value function for a state
    with torch.no_grad():
        vt_1 = critic_net(next_state) #Vt+1 - value function for the nextstate
        target = reward + done*params["gamma"] * vt_1
    criterion = nn.MSELoss()#nn.SmoothL1Loss()
    loss = criterion(target, vt)

    with torch.no_grad():
        advantage = target - vt

    return advantage, loss.unsqueeze(0)


def optimize_model(actor_net, actor_net_optimizer, critic_net,
                    critic_net_optimizer, env, params, action_critic_mode=None):
    device = params["device"]
    # state, info = env.reset()
    # state = torch.tensor(state, dtype=torch.float32).to(device)
    # state = state.unsqueeze(0)
    actor_loss_list = []
    critic_loss_list = []

    #several trajectories in one batch
    for batch_k in range(params["batch_size"]):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
        for i in range(params["max_trajectory_length"]):
            action, log_p = select_action(state, actor_net)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            d = 0 if done else 1
            advantage, loss_critic = critic_step(critic_net, state,
                                                    next_state, reward, d, params)

            critic_loss_list.append(loss_critic)
            loss_actor = -log_p * advantage
             #loss_actor = optimize_actor(log_p, actor_net_optimizer, advantage, params)
            actor_loss_list.append(loss_actor)
            state = next_state
            if done:
                break

    critic_loss = torch.cat(critic_loss_list).mean()
    critic_net_optimizer.zero_grad()
    critic_loss.backward()
    critic_net_optimizer.step()

    actor_loss = torch.cat(actor_loss_list).sum()
    actor_net_optimizer.zero_grad()
    actor_loss.backward()
    actor_net_optimizer.step()

    average_actor_loss = actor_loss.item()#sum(actor_loss_list) / len(actor_loss_list)
    average_critic_loss = critic_loss.item()#sum(critic_loss_list) / len(critic_loss_list)
    return average_actor_loss, average_critic_loss


def validation(env, policy_net, params, max_steps=500):
    device = params["device"]
    n_steps_list = []
    for trajectory in range(300):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
 
        for steps in range(max_steps):
            state = state.to(device)  
            action, _ = select_action(state, policy_net)
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
        max_steps = 3000
        env = gym.make(id='CartPole-v1', render_mode="rgb_array", max_episode_steps=max_steps) #render_mode ansi
        params = {"gamma":0.99, "lr":0.00005, "max_trajectory_length":500, 
                   "batch_size":2, "device":"cuda:1"}

        actor_net = DQN(4, 2)
        actor_net.load_state_dict(torch.load("weights/cartpole_actor.pth"))
        actor_net.to(params["device"])
        critic_net = DQN(4, 1)
        critic_net.load_state_dict(torch.load("weights/cartpole_critic.pth"))
        critic_net.to(params["device"])
        # target_critic_net = DQN(4, 1)
        # target_critic_net.to(params["device"])
        #target_critic_net.load_state_dict(torch.load("weights/cartpole_600L.pth"))

        actor_net_optimizer = t_opt.AdamW(actor_net.parameters(), lr=0.0001)
        critic_net_optimizer = t_opt.AdamW(critic_net.parameters(), lr=0.0001)
        max_steps_achieved = 0

        for epoch in range(100):
            actor_loss_list = []
            critic_loss_list = []

            steps_in_epoch = 100
            for i in range(steps_in_epoch):
                actor_loss, critic_loss = optimize_model(actor_net, actor_net_optimizer,
                                                          critic_net, critic_net_optimizer, env, params)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
            
            actor_loss_average = sum(actor_loss_list) / len(actor_loss_list)
            average_number_of_steps = validation(env, actor_net, params, max_steps=max_steps)
            criric_loss_average = sum(critic_loss_list) / len(critic_loss_list)
            del actor_loss_list
            del critic_loss_list
            print("{:d} actor loss: {:.5f}, critic loss: {:.4f}, val steps: {:.2f}".format(epoch, actor_loss_average, criric_loss_average, average_number_of_steps))
            #if average_number_of_steps > max_steps_achieved:
            torch.save(actor_net.state_dict(), "weights/cartpole_actor.pth")
            torch.save(critic_net.state_dict(), "weights/cartpole_critic.pth")

            

