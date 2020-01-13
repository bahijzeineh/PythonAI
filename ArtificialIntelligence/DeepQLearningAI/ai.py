# -*- coding: utf-8 -*-

#import libraries

import random
import os
import torch
import torch.nn as nn
import torch.nn.functional  as F
import torch.optim as optim
from torch.autograd import Variable


# create architecture of NN


class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        hidden_size = self.input_size * 6
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.nb_action)
        
    def forward(self, state):
        X = F.relu(self.fc1(state))
        q_values  = self.fc2(X)
        return q_values
        

#implement Experience Replay
        
    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


#implement Deep Q learning

    
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, replay_capacity = 100, save_file: str = 'last_brain.pth'):
        self.input_size = input_size
        self.nb_action = nb_action
        self.replay_capacity = replay_capacity
        self.save_file = save_file

        self.gamma = gamma
        self.reward_window = []
        
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(replay_capacity)
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.001)
        
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0.0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7) #temp = 7
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimiser.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimiser.step()
        
    def update(self, reward, signal):
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        self.memory.push(
                (self.last_state, 
                 new_state, 
                 torch.LongTensor([int(self.last_action)]), 
                 torch.Tensor([self.last_reward]))
                )
        
        action = self.select_action(new_state)
        
        if len(self.memory.memory) == self.replay_capacity:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(self.replay_capacity)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimiser': self.optimiser.state_dict},
                    self.save_file)

    def load(self):
        if os.path.isFile(self.save_file):
            print('loading last brain')
            checkpoint = torch.load(self.save_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser'])
        else:
            print('no save file found')