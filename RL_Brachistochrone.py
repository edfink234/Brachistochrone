from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gym import Env
from gym.spaces import Box
from pysr import PySRRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import warnings
warnings.filterwarnings("ignore")

def scale_between(unscaledNum, minAllowed, maxAllowed, Min, Max):
     return (maxAllowed - minAllowed) * (unscaledNum - Min) / (Max - Min) + minAllowed

class BrachistohroneEnv(Env):
    def __init__(self, x_start = 0, x_end = 1, y_start = 10, y_end = 0, iterations = 10000, outputs = 24, interactive = False, num_x_points = 50, y_min_plot = -30, y_max_plot = 15):
        
        self.y_i = y_start
        self.y_f = y_end
        self.state = self.y_i
        self.g = 9.80665
        self.prev_state = self.y_i
        self.v = 0 #initial velocity
        y_coord = self.y_i
        self.y_min = self.y_f - (self.v**2) / (2*self.g)
        self.y_max = (self.v**2) / (2*self.g) + y_coord
        self.action_space = Box(low = self.y_min, high = self.y_max)
        self.observation_space = Box(low = self.y_min, high = self.y_max)
        self.y_coords = [self.y_i]
        self.count = 0
        self.iterations = iterations
        self.completed_iterations = 0
        self.done = False
        self.t = 0 #total time
        self.best_y_coords = []
        self.best_t = np.inf
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(y_min_plot, y_max_plot)
        self.ax.set_xlim(10**x_start,10**x_end)
        self.ln, = self.ax.plot([], [], animated = True)
        plt.show(block=False)
        plt.pause(0.1)
        self.x_coords = np.logspace(x_start, x_end, num_x_points)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ln.set_data(self.x_coords,[0]*len(self.x_coords))
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)
        self.outputs = outputs
        self.interactive = interactive
        self.times = []
        
    def step(self, action):
        self.count += 1
        self.y_min = self.y_f - (self.v**2) / (2*self.g)
        self.y_max = (self.v**2) / (2*self.g) + self.prev_state
        
        action = scale_between(action, self.y_min, self.y_max, 0, self.outputs-1)
        
        if self.count == len(self.x_coords)-1:
            self.state = self.y_f
            assert(self.y_min-0.1 <= self.state <= self.y_max+0.1)
            
        else:
            self.state = action
            self.action_space = Box(low = self.y_min, high = self.y_max)
            self.observation_space = Box(low = self.y_min, high = self.y_max)
            assert(self.y_min-0.1 <= self.state <= self.y_max+0.1)
            
        opp = abs(self.state - self.prev_state)
        adj = abs(self.x_coords[self.count]-self.x_coords[self.count-1])
        theta = np.arctan(opp/adj)
        
        d = np.sqrt(opp**2 + adj**2)
#        print(opp/adj, np.sin(opp/adj))
        if self.state > self.prev_state:
            v_f = np.sqrt(np.abs(self.v**2 - 2*self.g*d*np.sin(theta)))
            self.t += abs(v_f - self.v)/(self.g*np.sin(theta))
            self.v = v_f
        else:
            v_f = np.sqrt(np.abs(self.v**2 + 2*self.g*d*np.sin(theta)))
            self.t += abs(v_f - self.v)/(self.g*np.sin(theta))
            self.v = v_f
        
        self.prev_state = self.state
        self.y_coords.append(self.prev_state)
        
        reward = 0
        
        if (len(self.y_coords) == len(self.x_coords)):
            self.completed_iterations += 1
            if np.isnan(self.t):
                reward = -1
            elif not self.times:
                self.best_y_coords = self.y_coords
                self.times.append(self.t)
                reward = 1
                self.best_t = self.t
                print(self.best_t)
                print("\nNew Best Time = ",self.best_t)
                if not self.interactive:
                    self.render()
            elif self.t < min(self.times):
                self.best_y_coords = self.y_coords
                self.times.append(self.t)
                reward = 1
                self.best_t = self.t
                print("\nNew Best Time = ",self.best_t)
                if not self.interactive:
                    self.render()
            elif self.t >= min(self.times):
                reward = -1
#                self.render()
            if self.interactive:
                self.render(1)
            self.reset()
        if self.completed_iterations == self.iterations:
            self.done = True
        
        return self.state, reward, self.done, {}

    def render(self, pause_time = 0.1):
        if len(self.x_coords) == len(self.best_y_coords):
            self.fig.canvas.restore_region(self.bg)
            self.ln.set_ydata(self.best_y_coords)
            self.ax.draw_artist(self.ln)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()
            plt.pause(pause_time)
        
    def reset(self):
        self.state = self.y_i
        self.prev_state = self.y_i
        self.v = 0 #initial velocity
        y_coord = self.y_i
        self.y_min = self.y_f - (self.v**2) / (2*self.g)
        self.y_max = (self.v**2) / (2*self.g) + y_coord
        self.action_space = Box(low = self.y_min, high = self.y_max)
        self.observation_space = Box(low = self.y_min, high = self.y_max)
        self.y_coords = [self.y_i]
        self.count = 0
        self.t = 0
        self.completed_iterations = 0
        return self.state
        
def build_model(states, actions, outputs = 48):
    model = Sequential()
    model.add(Dense(24,activation='relu', input_shape=states))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(outputs,activation='relu', input_shape=actions))
    return model

def build_agent(model, outputs = 48):
    memory = SequentialMemory(limit = 50000, window_length = 1)
    dqn = DQNAgent(model, nb_actions = outputs, memory = memory, nb_steps_warmup=1000)
    return dqn

if __name__ == '__main__':
    test = BrachistohroneEnv(x_end = 0, x_start = -10, outputs = 1000, num_x_points = 100, y_start = 0, y_end = -1, y_max_plot = 0, y_min_plot = -5)
#    print(test.action_space.sample())
#    print(test.observation_space.sample())
    states = test.observation_space.shape
    actions = test.action_space.shape
#    print(states,actions)
    
    model = build_model(states, actions, outputs = test.outputs)
    model.summary()
    dqn = build_agent(model, outputs = test.outputs)
    dqn.compile(Adam(learning_rate=1e-5), metrics=['mae'])

    dqn.fit(test, nb_steps = 50000, visualize = False)
    print("Best time = ",test.best_t)
    
    model = PySRRegressor(
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    )
    X = test.x_coords
    y = test.best_y_coords
    X = X.reshape(-1,1)
    model.fit(X, y)
    print(model)
        
        
