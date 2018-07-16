import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import math, os
import numpy as np
from Env_2D import EnvUAV
from swarm_uav import Net,v_wrap
os.environ["OMP_NUM_THREADS"] = "1"
torch.cuda.set_device(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
save_name = 'global.pkl'
max_steps = 100
N_S = 21
N_A = 2
num_agents = 1

if __name__ =='__main__':
    net = Net(N_S,N_A)
    net.load_state_dict(torch.load(save_name))
    env = EnvUAV(num_agents)
    s = env.reset()
    for i in range(max_steps):
        a1,a2 = net.choose_action(v_wrap(s[None, :]))
        a1, a2 = a1[np.newaxis, :], a2[np.newaxis, :]
        input = np.array([a1.clip(-2, 2), a2.clip(-1, 1)]).reshape((1, 2))
        s_, r, done, done2 = env.forward(input)
        env.render()
        s = s_