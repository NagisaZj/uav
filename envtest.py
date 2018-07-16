import numpy as np
import math
import tensorflow as tf
from Env_2D import EnvUAV
class tester():
    def __init__(self,num_agents):
        self.num_agents = num_agents
        self.env = EnvUAV(self.num_agents)
        self.env.reset()
        self.env.render()
        #self.env.end()
    def act_test(self):
        self.action = np.ones([self.num_agentss,2])
        for i in range(200):
            self.env.forward(self.action)
            if i%2 ==0:
                self.env.render()



def main():
    num_agents = 9
    test = tester(num_agents)
    test.act_test()
    test.env.end()
if __name__=='__main__':
    main()

