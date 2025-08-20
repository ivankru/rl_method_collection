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
from torch.nn.utils import clip_grad_norm_ 

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        del self.memory
        self.memory = deque([], maxlen=self.capacity)

    def clear_random(self, number_to_remain):
        """"
        randomly delete len() - number_to_remain elements
        """
        new_elements = random.sample(self.memory, min(number_to_remain, len(self.memory)))
        self.memory = deque(new_elements, maxlen=self.capacity)



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
    action_prob = policy_net(state)#.cpu()
    action_prob = torch.softmax(action_prob, dim=0)
    random_distribution = Categorical(action_prob)
    action = random_distribution.sample()
    log_prob = random_distribution.log_prob(action)

    if len(state.shape) == 1:
        p = random_distribution.probs[action]
    else:
        action_array = torch.vstack([action, action]).T
        p = random_distribution.probs.gather(1, action_array)[:,1]
    log_1_p = torch.log(1 - p + 0.00000001)
    
    return action, log_prob, log_1_p


def p_action(state, policy_net, action):
    state = state.squeeze(0)
    action_prob = policy_net(state)#.cpu()
    action_prob = torch.softmax(action_prob, dim=0)
    random_distribution = Categorical(action_prob)
    log_prob = random_distribution.log_prob(action)

    # if len(state.shape) == 1:
    #     p = random_distribution.probs[action]
    # else:
    action_array = torch.vstack([action, action]).T
    p = random_distribution.probs.gather(1, action_array)[:,1]
    log_1_p = torch.log(1 - p + 0.00000001)
    
    return log_prob, log_1_p


def replay_memory(env, actor, memory,
                   params, trajectory_numb = 1000, n_steps=500):
    device = params["device"]
    n_steps_list = []

    for trajectory in range(trajectory_numb):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
 
        for steps in range(n_steps):
            state = state.to(device)  
            with torch.no_grad():
                action, _, _ = select_action(state, actor)
            action = action.item()
            state = state.cpu()
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

        n_steps_list.append(steps)

    avarage_steps = sum(n_steps_list) / len(n_steps_list)
    return avarage_steps


def optimize_model(memory, actor_net, actor_net_optimizer,
                    critic_net, critic_net_target, critic_net_optimizer, parameters, optimize_actor=True):
    device = parameters["device"]
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

    advantage, loss_critic = critic_step(critic_net, critic_net_target, state_batch, non_final_next_states,
                                          non_final_mask, reward_batch, params)

    #_, log_p, log_1_p = select_action(state_batch, actor_net)
    log_p, log_1_p = p_action(state_batch, actor_net, action_batch)

    critic_net_optimizer.zero_grad()
    #if not optimize_actor:
    loss_critic.backward()
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=1.0)
    critic_net_optimizer.step()
    
    #final_mask = torch.logical_not(non_final_mask)#DELTE ME!!! for debuggind only
    positive_advantage = -log_p[advantage >= 0] * advantage[advantage >= 0]
    negative_advantage = log_1_p[advantage < 0] * advantage[advantage < 0]
    positive_negative_advantage = torch.hstack([positive_advantage, negative_advantage])
    #loss_actor = -(advantage * log_p).mean()
    loss_actor = positive_negative_advantage.mean()
    #if optimize_actor:
    actor_net_optimizer.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
    actor_net_optimizer.step()

    return loss_actor.item(), loss_critic.item()


def critic_step(critic_net, critic_net_target, state_batch, non_final_next_states,
                non_final_mask, reward_batch, params):
    vt = critic_net(state_batch).squeeze(1) #Vt - value function for a state
    target = torch.zeros_like(vt)
    with torch.no_grad():
        vt_1 = critic_net(non_final_next_states).squeeze(1) #Vt+1 - value function for the nextstate
        target[non_final_mask] = reward_batch[non_final_mask] + params["gamma"] * vt_1
        final_mask = torch.logical_not(non_final_mask)
        target[final_mask] = reward_batch[final_mask]
    criterion = nn.MSELoss()
    loss = criterion(target, vt)

    with torch.no_grad():
        advantage = target - vt

    return advantage, loss.unsqueeze(0)


def validation(env, actor_net, critic_net, params, max_steps=500):
    device = params["device"]
    n_steps_list = []
    for trajectory in range(1):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
 
        for steps in range(max_steps):
            state = state.to(device)
            value = critic_net(state) 
            with torch.no_grad():
                action, log_p, _ = select_action(state, actor_net)
            state= state.cpu()
            observation, reward, terminated, truncated, _ = env.step(action.item())
            # reward = torch.tensor([reward]) #, device=device
            # action = torch.tensor([action])
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)#device=device
                with torch.no_grad():#for debugging only
                    v2 = critic_net(next_state.to(device)) #for debugging only
                advantage = v2 - value #for debugging only

            (steps, value, advantage, log_p)   
            state = next_state
            if done:
                break
        n_steps_list.append(steps)

    avarage_steps = sum(n_steps_list) / len(n_steps_list)
    return avarage_steps


if __name__ == "__main__":
    params = {"gamma":0.99, "lr":0.0002, "max_trajectory_length":3000, 
                "batch_size":128, "device":"cuda:1"}
    
    max_steps = 3000
    env = gym.make(id='CartPole-v1', render_mode="rgb_array", max_episode_steps=max_steps) #render_mode ansi
    
    memory = ReplayMemory(30000)
    actor_net = DQN(4, 2)
    actor_net.to(params["device"])
    critic_net = DQN(4, 1)
    critic_net_target = DQN(4, 1)
    # critic_net.load_state_dict(torch.load("weights/cartpole_critic.pth"))
    # actor_net.load_state_dict(torch.load("weights/cartpole_actor.pth"))
    critic_net.to(params["device"])
    #critic_net_target.load_state_dict(torch.load("weights/cartpole_critic.pth"))
    critic_net_target.to(params["device"])

    actor_net_optimizer = t_opt.AdamW(actor_net.parameters(), lr=0.0001)
    critic_net_optimizer = t_opt.AdamW(critic_net.parameters(), lr=0.0002)
    average_number_of_steps = replay_memory(env, actor_net, memory, params, trajectory_numb=100, n_steps=1000)

    max_average_trajectory_length = 0
    for epoch in range(400):
        actor_loss_list = []
        critic_loss_list = []
        steps_in_epoch = 100
        trajectory_numb =  2 + round((steps_in_epoch*params["batch_size"]) / average_number_of_steps)

        for i in range(steps_in_epoch):
            optimize_actor = True if i % 2 == 1 else False
            actor_loss, critic_loss = optimize_model(memory, actor_net, actor_net_optimizer,
                            critic_net, critic_net_target, critic_net_optimizer, params, optimize_actor)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)

        critic_net_target.load_state_dict(critic_net.state_dict())
        #validation(env, actor_net, critic_net, params, max_steps=500)
        
        # if epoch % 3 == 0:
        #     memory.clear()
        memory.clear_random(steps_in_epoch * params["batch_size"])
        average_number_of_steps = replay_memory(env, actor_net, memory, params,
                                                 trajectory_numb=trajectory_numb, n_steps=params["max_trajectory_length"])
        actor_loss_average = sum(actor_loss_list) / len(actor_loss_list)
        critic_loss_average = sum(critic_loss_list) / len(critic_loss_list)
        print("{:d} actor loss: {:.5f}, critic loss: {:.4f}, val steps: {:.2f}".format(epoch, actor_loss_average, critic_loss_average, average_number_of_steps))
        # if average_number_of_steps > max_average_trajectory_length:
        #     torch.save(actor_net.state_dict(), "weights/cartpole_actor.pth")
        #     torch.save(critic_net.state_dict(), "weights/cartpole_critic.pth")
        max_average_trajectory_length = max(max_average_trajectory_length, average_number_of_steps)
