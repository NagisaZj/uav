# coding: utf-8
''' UAV_environment '''
import numpy as np
import math
import tensorflow as tf





class EnvUAV():
    num_states = 21
    num_actions = 2
    action_space_low = [-1.0,-1.0]
    action_space_high = [1.0,1.0]
    state_space_low = -1.0
    state_space_high = 1.0
    
    def __init__(self,num_agents):
        self.num_agents = num_agents
        self.File = []
        for i in range(self.num_agents):
            s = open('./Result/path_agent' + str(i) + '.txt','w+')
            self.File.append(s)
        
        ##无人机参数
        self.level = 100.0 #定义无人机的飞行高度
        self.scope = 100.0 #定义无人机传感器的视野范围
        self.reward = np.zeros([self.num_agents])
        self.state = np.zeros([self.num_agents,21]) #定义无人机状态向量并随机初始化
        self.position = np.zeros([self.num_agents,2])
        self.target = np.zeros([2])
        self.orient = np.zeros([self.num_agents])
        self.speed = np.zeros([self.num_agents]) #定义无人机的初始速度为0.0
        self.exist_agent = np.zeros([self.num_agents,2])
        

        ##环境参数
        self.num_circle = 10 #crossing环境重复次数
        self.step = 0.1 #定义无人机传感器搜索步长
        self.radius = 60 #障碍物的半径
        self.repeat = 200 #环境重复的周期
        self.gap = 1.0 #无人机之间的最小安全距离
        self.initial_gap = 20.0 #无人机初始间隔
        
    def obtain_state(self, position, target, orient):
        for j in range(self.num_agents):
            for i in range(0,9):
                theta = np.mod((i-4)*np.pi/8 + orient[j],2*np.pi) #计算传感器各个方向的角度,保证不超过360度
                end_cache = position[j]
                end = np.mod(end_cache,self.repeat)
                Count = 0
                while 1:
                    Count = Count + 1
                    position_integer = end_cache/self.repeat
                    if self.mat_exist[np.int32(position_integer[0]) + 8,np.int32(position_integer[1]) + 8]>0:
                        if np.sqrt(np.power(end[0]-self.repeat/2,2) + np.power(end[1]-self.repeat/2,2))-self.radius <= 0:
                            self.state[j,i] = np.linalg.norm(end_cache-position[j])/self.scope
                            break
                    if Count == 1000:
                        self.state[j,i] = 1
                        break
                    end_cache = end_cache + [self.step*np.sin(theta),self.step*np.cos(theta)]
                    end = np.mod(end_cache,self.repeat)
                        
            #################################################################################################            

            #计算当前位置距离目标的距离，记为状态10
            dist  =  np.linalg.norm(target-position[j])
            self.state[j,9] = 2/(np.exp(-0.002*dist) + 1)-1

            #计算目标和当前位置的相对夹角
            theta_target = np.arctan((target[0]-position[j,0])/(target[1]-position[j,1]))
            if (target[0] >= position[j,0] and target[1] >= position[j,1]):
                self.state[j,10] = np.sin(theta_target)
                self.state[j,11] = np.cos(theta_target)
            elif target[1]<position[j,1]:
                self.state[j,10] = np.sin(theta_target + np.pi)
                self.state[j,11] = np.cos(theta_target + np.pi)
            else:
                self.state[j,10] = np.sin(theta_target + 2*np.pi)
                self.state[j,11] = np.cos(theta_target + 2*np.pi)

            #保存目标的绝对方向角
            self.state[j,12] = np.sin(orient[j]) 
            self.state[j,13] = np.cos(orient[j]) 
            #保存速度为另一个状态
            self.state[j,14] = np.tanh(0.1*self.speed[j])

            distance = np.linalg.norm(position[j]-position,axis=1)
            distance_sorted = np.sort(distance)
            counter_left = 0
            counter_right = 0
            for g in range(1,self.num_agents):
                agent_index_temp = np.argmax(1-np.abs(np.sign(distance-distance_sorted[g])))
                theta_agent_temp = np.arctan((position[agent_index_temp,0]-position[j,0])/(0.00000001+(position[agent_index_temp,1]-position[j,1])))
                if (position[agent_index_temp,0] >= position[j,0]) * (position[agent_index_temp,1] >= position[j,1]):
                    theta_agent_temp = theta_agent_temp
                elif position[agent_index_temp,1] < position[j,1]:
                    theta_agent_temp = theta_agent_temp + np.pi
                else:
                    theta_agent_temp = theta_agent_temp + 2*np.pi
                delta_theta = np.mod(theta_agent_temp - orient[j], 2*np.pi)
                #####each UAV considers the relative position of the nearest two neighbors#######
                if (delta_theta >= np.pi) * (counter_left == 0):
                    counter_left = 1
                    #print('distance_left',distance_sorted[g])
                    self.exist_agent[j,0] = 1
                    self.state[j,15] = np.sin(theta_agent_temp)
                    self.state[j,16] = np.cos(theta_agent_temp)
                    self.state[j,17] = 2/(np.exp(-0.02*distance_sorted[g]) + 1)-1
                if (delta_theta < np.pi) * (counter_right == 0):
                    counter_right = 1
                    #print('distance_right',distance_sorted[g])
                    self.exist_agent[j,1] = 1
                    self.state[j,18] = np.sin(theta_agent_temp)
                    self.state[j,19] = np.cos(theta_agent_temp)
                    self.state[j,20] = 2/(np.exp(-0.02*distance_sorted[g]) + 1)-1
                    
                if (counter_left > 0) * (counter_right > 0):
                    break
            if counter_left == 0:
                #print('distance_left inf')
                self.exist_agent[j,0] = 0
                self.state[j,15] = np.sin(3.0/2.0*np.pi)
                self.state[j,16] = np.cos(3.0/2.0*np.pi)
                self.state[j,17] = 2/(np.exp(-0.02*20.0) + 1)-1
            if counter_right == 0:
                #print('distance_right inf')
                self.exist_agent[j,1] = 0
                self.state[j,18] = np.sin(1.0/2.0*np.pi)
                self.state[j,19] = np.sin(1.0/2.0*np.pi)
                self.state[j,20] = 2/(np.exp(-0.02*20.0) + 1)-1  
            
    def reset(self):
        ###########################################随机初始化环境##########################################
        self.mat_height = np.random.randint(1,10,size = (self.num_circle + 16,self.num_circle + 16))*17.0 + 30.0 #定义环境中建筑物的高度
        self.mat_exist = self.mat_height-self.level #确定飞行高度范围内的建筑

        ###########################################首先产生初始位置########################################
        while 1:
            x = np.random.uniform(0,self.repeat)
            y = np.random.uniform(0,self.repeat)
            if np.sqrt(np.power(x-self.repeat/2,2) + np.power(y-self.repeat/2,2))-self.radius>0:
                position = np.array([x,y])
                break
        relative_position = np.floor(np.random.uniform(0, self.num_circle, size = (2,))) 
        self.position[0] = np.array(position  +  relative_position*self.repeat)
        for i in range(1,self.num_agents):
            while(1):
                orient_temp = np.random.uniform(0,1)*2*np.pi
                initial_gap_temp = self.initial_gap + np.random.normal(0.0,3.0)
                position_temp = position + np.array([initial_gap_temp*np.sin(orient_temp),initial_gap_temp*np.cos(orient_temp)])
                position_final = np.mod(position_temp,self.repeat)
                if np.sqrt(np.power(position_final[0]-self.repeat/2,2) + np.power(position_final[1]-self.repeat/2,2)
                           )-self.radius>0 and position_temp[0]>0 and position_temp[1]>0:
                    self.position[i] = np.array(position_temp  +  relative_position*self.repeat)
                    if self.position[i][0]<self.num_circle*self.repeat and self.position[i][1]<self.num_circle*self.repeat:
                        break
        self.orient = np.random.uniform(0,2*np.pi,[self.num_agents])
            
        ###########################################其次产生目标位置########################################
        while 1:
            x = np.random.uniform(0,self.repeat)
            y = np.random.uniform(0,self.repeat)
            if np.sqrt(np.power(x-self.repeat/2,2) + np.power(y-self.repeat/2,2))-self.radius>0:
                target = [x,y]
                break

        relative_target = np.floor(np.random.uniform(0, self.num_circle, size = (2,)))
        self.target  =  np.array(target  +  relative_target*self.repeat)
 
        #################################################################################################
        np.savetxt('./Result/mat_height.txt',self.mat_height,fmt='%d',delimiter=' ',newline='\r\n')
        np.savetxt('./Result/mat_exist.txt',self.mat_exist,fmt='%d',delimiter=' ',newline='\r\n')
        #################################################################################################
        self.obtain_state(np.copy(self.position), np.copy(self.target), np.copy(self.orient))
        observation = np.copy(self.state)
        return observation

    def forward(self, action):
        #################################################################################################
        ##########################################Apply Action###########################################
        #################################################################################################
        #print('speed and orient',self.speed,self.orient)
        position_temp = np.copy(self.position)
        self.orient = np.mod(1.0/4.0*action[:,0]*np.pi + self.orient,2*np.pi)
        self.speed = np.where(action[:,1]>=0,self.speed + action[:,1]*(-np.tanh(0.5*(self.speed-10.0))),self.speed + action[:,1]*np.tanh(0.5*self.speed))
        self.position = self.position + np.append(np.expand_dims(self.speed*np.sin(self.orient),1),np.expand_dims(self.speed*np.cos(self.orient),1),1)
        self.obtain_state(np.copy(self.position),np.copy(self.target),np.copy(self.orient))
        #################################################################################################
        #####################################termination Judgement#######################################
        #################################################################################################
        relative_position = np.mod(self.position,self.repeat)
        position_integer = self.position/self.repeat
        done1 = (np.sqrt(np.power(relative_position[:,0]-self.repeat/2,2) + np.power(relative_position[:,1]-self.repeat/2,2))-self.radius <= 0)\
                * (self.mat_exist[np.int32(position_integer[:,0]) + 8,np.int32(position_integer[:,1]) + 8]>0)
        done2 = (np.linalg.norm((self.position-self.target),axis=1) <= 20)
        mutual_distance_left = np.log(2/(self.state[:,17]+1)-1)/(-0.02)
        mutual_distance_right = np.log(2/(self.state[:,20]+1)-1)/(-0.02)
        done3 = (mutual_distance_left <= self.gap) + (mutual_distance_right <= self.gap)
        done = done1 + done2 + done3
        #print('done1,done2,done3',done1,done2,done3)
        for i in range(self.num_agents):
            if done1[i]:
                print('agent',i,'collides with obstacles!')
            if done2[i]:
                print('agent',i,'arrived at the destination!')
            if done3[i]:
                print('agent',i,'is too close to its nearby agent')

        #################################################################################################
        #####################################Reward Specification########################################
        #################################################################################################       
        reward_sparse = np.where(done2,np.zeros([self.num_agents])+15.0,np.zeros([self.num_agents]))
        reward_distance = np.tanh(0.2*(10.0-self.speed))*(np.linalg.norm(position_temp-self.target,axis=1)-np.linalg.norm(self.position-self.target,axis=1))
        reward_barrier = np.where(np.min(self.state[:,0:9],1)*100<=10.0,-5.0+np.zeros([self.num_agents]),np.zeros([self.num_agents]))
        reward_action = -3.0
        reward_mutual = np.where((mutual_distance_left<=50)*(mutual_distance_right<=50),3.0*np.exp(-np.square(mutual_distance_left-20)*0.05) + 3.0*np.exp(-np.square(mutual_distance_right-20)*0.05),np.zeros([self.num_agents]))
        reward_agent = np.where(mutual_distance_left<=10.0,np.zeros([self.num_agents])-5.0,np.zeros([self.num_agents]))\
                       + np.where(mutual_distance_right<=10.0,np.zeros([self.num_agents])-5.0,np.zeros([self.num_agents]))
        reward_mutual = 0
        reward_distance = reward_distance * 100
        print("sparse:",reward_sparse," distance:",reward_distance," barrier:",reward_barrier,
        " mutual:", reward_mutual," agent:",reward_agent)
        self.reward = reward_sparse + reward_barrier + reward_distance + reward_action + reward_mutual + reward_agent

        #################################################################################################  
        next_observation = np.copy(self.state)
        return next_observation, self.reward, done, done2
    
    def render(self):
        ##输出轨迹参数
        for i in range(self.num_agents):
            self.File[i].write(str(self.target[0]))
            self.File[i].write(' ')
            self.File[i].write(str(self.target[1]))
            self.File[i].write(' ')
            self.File[i].write(str(self.position[i,0]))
            self.File[i].write(' ')
            self.File[i].write(str(self.position[i,1]))
            self.File[i].write(' ')
            for j in range(len(self.state[i])):
                self.File[i].write(str(self.state[i,j]))
                self.File[i].write(' ')
            self.File[i].write(str(self.speed[i]))
            self.File[i].write(' ')
            if self.exist_agent[i,0]:
                self.File[i].write(str(np.log(2/(self.state[i,17]+1)-1)/(-0.02)))
                self.File[i].write(' ')
            else:
                self.File[i].write(str(10000.0))
                self.File[i].write(' ')
                
            if self.exist_agent[i,1]:
                self.File[i].write(str(np.log(2/(self.state[i,20]+1)-1)/(-0.02)))
                self.File[i].write(' ')
            else:
                self.File[i].write(str(10000.0))
                self.File[i].write(' ')
                
            self.File[i].write(str(self.orient[i]))
            self.File[i].write('\n')

    def end(self):
        for i in range(self.num_agents):
            self.File[i].close()
        






