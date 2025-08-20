import gymnasium as gym
import matplotlib.pyplot as plt
#from collections import namedtuple, deque
# import random
# import numpy as np
# from torch import nn
import torch
# import torch.functional as F
# import torch.optim as t_opt
from cartpole_value import DQN, select_action
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = "cuda:1"
    env = gym.make(id='CartPole-v1', render_mode="rgb_array") #render_mode ansi
    env = gym.wrappers.RecordVideo(env=env, video_folder="images", name_prefix="cartpole-video", episode_trigger=lambda x: x % 2 == 0)

    policy_net = DQN(4,2)
    policy_net.load_state_dict(torch.load("weights/cartpole.pth"))
    policy_net.to(device)

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state = state.unsqueeze(0)

    for i in range(1000):
        state = state.to(device)  
        action = select_action(state, policy_net, eps=0.0)
        #state= state.cpu()
        observation, reward, terminated, truncated, _ = env.step(action)
        env.render()
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)#device=device         
        state = next_state
        if done:
            break

    #env.close_video_recorder()
    env.close()