import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import math, os
import numpy as np
from Env_2D_train import EnvUAV_train
os.environ["OMP_NUM_THREADS"] = "1"
torch.cuda.set_device(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200
N_S = 21
N_A = 2

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.1)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba1,ba2, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba1), dtype=np.int64) if ba1[0].dtype == np.int64 else v_wrap(np.vstack(ba1)),
        v_wrap(np.array(ba2), dtype=np.int64) if ba2[0].dtype == np.int64 else v_wrap(np.vstack(ba2)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 100)
        self.mu1 = nn.Linear(100, 1)
        self.sigma1 = nn.Linear(100, 1)
        self.mu2 = nn.Linear(100, 1)
        self.sigma2 = nn.Linear(100, 1)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu1, self.sigma1, self.mu2, self.sigma2, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        mu1 = 2 * F.tanh(self.mu1(a1))
        sigma1 = F.softplus(self.sigma1(a1)) + 0.001      # avoid 0
        mu2 = 10 * F.tanh(self.mu2(a1)) - 5
        sigma2 = F.softplus(self.sigma2(a1)) + 0.001  # avoid 0
        c1 = F.relu(self.c1(x))
        values = self.v(c1)
        return mu1, sigma1, mu2, sigma2, values

    def choose_action(self, s):
        self.training = False
        mu1, sigma1,mu2, sigma2, _ = self.forward(s)
        m1 = self.distribution(mu1.view(1, ).data, sigma1.view(1, ).data)

        m2 = self.distribution(mu2.view(1, ).data, sigma2.view(1, ).data)
        return m1.sample().numpy(),m2.sample().numpy()

    def loss_func(self, s, a1,a2, v_t):
        self.train()
        mu1, sigma1, mu2, sigma2, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m1,m2 = self.distribution(mu1, sigma1),self.distribution(mu2, sigma2)
        log_prob = m1.log_prob(a1)+m2.log_prob(a2)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m1.scale) + torch.log(m2.scale) # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = EnvUAV_train(1)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a1,buffer_a2, buffer_r = [], [], [],[]
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                #if self.name == 'w0':
                #    self.env.render()
                a1,a2 = self.lnet.choose_action(v_wrap(s[None, :]))
                a1,a2 = a1[np.newaxis,:],a2[np.newaxis,:]
                input = np.array([a1.clip(-2, 2),a2.clip(-1, 1)]).reshape((1,2))
                s_, r, done, done2 = self.env.forward(input)
                if t == MAX_EP_STEP - 1:
                    done2 = True
                ep_r += r
                buffer_a1.append(a1)
                buffer_a2.append(a2)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0  or done2:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a1,buffer_a2, buffer_r, GAMMA)
                    buffer_s, buffer_a1,buffer_a2, buffer_r = [], [], [], []

                    if  done2:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(8)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]


    torch.save(gnet.state_dict(),'global.pkl')
    res_array = np.array(res,dtype = np.int64)
    np.save("result.npy",res)

