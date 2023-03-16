from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from snake_game import SnakeGameAI, Direction, Point
from collections import deque
from plotting import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        file_name = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, current_state, action, reward, new_state, done):
    
        current_state = torch.tensor(current_state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #it has to be (n,x) shape - n is number of batches

        if len(current_state.shape) == 1:
            #changing to this shape (1,x) 
            current_state = torch.unsqueeze(current_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(current_state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(new_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
           
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
    
        #loss.backward(retain_graph=True)
        self.optimizer.step()







class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #removes from left if greater than max memory
        self.model = QNet(11, 256, 3) #11 is input vector and output has 3 possible actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            #Current moving direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food loc
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y, #food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory
        
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        action = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1
        
        return action
            


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    if os.path.exists('./model/model.pth'):
        agent.model.load_state_dict(torch.load('./model/model.pth'))

    while True:
        current_state = agent.get_state(game)
        action = agent.get_action(current_state)

        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)


        #agent.memory.append([current_state, action, new_state, reward])
        agent.train_short_memory(current_state, action, reward, new_state, done)

        agent.remember(current_state, action, reward, new_state, done)

        if done:
            #train long memory and plot results
            game.reset()
            agent.n_games += 1
            #agent.model.load_state_dict(torch.load('./model/model.pth'))
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Reward', reward, 'Score', score, 'Record:',record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()