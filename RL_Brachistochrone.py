#Using reinforcement learning and symbolic regression to discover the 
#non-parametric form of the Brachistochrone

#TODO: Add autoscale parameter so that it will show the whole path if it trails off the page (6.)
#https://proofwiki.org/wiki/Time_of_Travel_down_Brachistochrone
#TODO: Make step function compute all N y-values in one step (5.)
#TODO: Integrate the tautochrone constraint to the reward function (7.)
#TODO: Post on math stack exchange the question about how to calculate at which x-range the tautochrone condition would hold for the brachistochrone (7.)
#TODO: Use recurrent network? (7.)
#https://www.tensorflow.org/guide/keras/rnn

#Modified Files
#==============
#/Users/edwardfinkelstein/MachineLearning/myvenv/lib/python3.9/site-packages/keras/engine/training_utils_v1.py -> Commented out line 711-721
#/Users/edwardfinkelstein/MachineLearning/myvenv/lib/python3.9/site-packages/rl/agents/ddpg.py: comment out everything line that has uses_learning_phase (except lines 80-82), i.e., lines 153-154, 310-311
#if hasattr(actor.output, '__len__') and len(actor.output) > 1: -> if hasattr(actor.output, '__shape__') and len(actor.output) > 2:
#if hasattr(critic.output, '__len__') and len(critic.output) > 1: -> if hasattr(critic.output, '__shape__') and len(critic.output) > 2:

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from gym import Env
from gym.spaces import Box
from pysr import PySRRegressor

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras import Model, initializers
from keras.layers import Concatenate

from tensorflow_addons.optimizers import LazyAdam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Input, LSTM, SimpleRNN, Embedding, GRU

from rl.agents import DQNAgent, DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from sympy import symbols
from sympy.utilities.lambdify import lambdify

import warnings
warnings.filterwarnings("ignore")

class AdjustModel(tf.keras.callbacks.Callback):
    def __init__(self, update_every = 0, lr_factor = 1, update_lr = False, reset_weights = False):
        super().__init__()
        self.curr_step = 0
        self.update_every = abs(update_every)
        self.lr_factor = lr_factor
        self.update_lr = update_lr
        self.reset_weights = reset_weights
    def on_batch_end(self, batch, logs=[]):
    
        if self.update_every:
            if K.eval(self.model.step) == self.curr_step + self.update_every:
                if self.update_lr:
                    self.model.actor.optimizer.learning_rate.assign(self.model.actor.optimizer.learning_rate*self.lr_factor)
                if self.reset_weights:
                    for l in self.model.layers:
                        if hasattr(l,"kernel_initializer"):
                            kernel_initializer=initializers.glorot_uniform(seed=np.random.randint(0,1000))
                            l.kernel.assign(kernel_initializer(tf.shape(l.kernel).eval(session=tf.compat.v1.Session())))
                        if hasattr(l,"bias_initializer"):
                            bias_initializer = initializers.RandomNormal(stddev=0.01, seed=np.random.randint(0,1000))
                            l.bias.assign(bias_initializer(tf.shape(l.bias).eval(session=tf.compat.v1.Session())))
                        if hasattr(l,"recurrent_initializer"):
                            recurrent_initializer = initializers.RandomNormal(stddev=0.01, seed=np.random.randint(0,1000))
                            l.recurrent_kernel.assign(recurrent_initializer(tf.shape(l.recurrent_kernel).eval(session=tf.compat.v1.Session())))
                self.curr_step = self.model.step

def sigmoid(x, c = 1):
    return 1/(1+np.exp(c*x))

def scale_between(unscaledNum, minAllowed, maxAllowed, Min, Max):
     return (maxAllowed - minAllowed) * (unscaledNum - Min) / (Max - Min) + minAllowed
     
def BrachistohronePoints(start_point, end_point, g = 9.80665):
    x_start, y_start = start_point
    x_end, y_end = end_point
    x_diff, y_diff = 0 - x_start, 0 - y_start
    x_start, x_end = x_start + x_diff, x_end + x_diff
    y_start, y_end = y_start + y_diff, y_end + y_diff
    
    def func(t, C):
        return (t-np.sin(t))/(1-np.cos(t)) + C
    
    root = fsolve(func, [1], args = (x_end/y_end,))
    t = root[0]
    a = x_end/(t-np.sin(t))
    theta = np.linspace(0, np.pi, 1000)
    x_points = a*(theta - np.sin(theta)) - x_diff
    y_points = -a*(1-np.cos(theta)) - y_diff
    
    cut = len(x_points[x_points <= end_point[0]])
    
    optimal_time = t * np.sqrt(a / g)
    
    return x_points[:cut], y_points[:cut], optimal_time

class BrachistohroneEnv(Env):
    def __init__(self, x_start = 0, x_end = 1, y_start = 10, y_end = 0, iterations = 10000, interactive = False, num_x_points = 50, g = 9.80665, point_dist="log", autoscale = False):
        
        if y_end >= y_start:
            y_start, y_end = y_end, y_start
        if x_end <= x_start:
            x_start, x_end = x_end, x_start
            
        self.y_i = y_start
        self.y_f = y_end
        self.state = self.y_i
        self.g = g
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
        self.ln, = self.ax.plot([], [], animated = True)
        if point_dist == "linear":
            x_points, y_points, self.optimal_time = BrachistohronePoints((x_start, y_start), (x_end, y_end))
            plt.scatter((x_start, x_end), (y_start, y_end))
            
        else:
            x_points, y_points, self.optimal_time = BrachistohronePoints((10**x_start, y_start), (10**x_end, y_end))
            plt.scatter((10**x_start, 10**x_end), (y_start, y_end))
            
        
        plt.plot(x_points, y_points, label = f"Best Time = {self.optimal_time:0.3f}")
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        if point_dist == "linear":
            self.x_coords = np.linspace(x_start, x_end, num_x_points)
        else:
            self.x_coords = np.logspace(x_start, x_end, num_x_points)
        
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ln.set_data(self.x_coords,[0]*len(self.x_coords))
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)
        self.interactive = interactive
        self.min_action = 0
        self.max_action = 0
        
    def step(self, action):
        action = action[0]
        
#        Trying to figure out in what range the actions like...
#        ======================================================
#        if action < self.min_action:
#            self.min_action = action
#            print("\n")
#            print(self.min_action,self.max_action)
#            print("\n")
#        elif action > self.max_action:
#            self.max_action = action
#            print("\n")
#            print(self.min_action,self.max_action)
#            print("\n")
            
        self.count += 1
        self.y_min = self.y_f - (self.v**2) / (2*self.g)
        self.y_max = (self.v**2) / (2*self.g) + self.prev_state
        
#        Currently, I don't know what the maximum and minimum possible
#        values for action are, so I just picked -100 to 100
#        arbitrarily, and if for some reason this range is exceeded,
#        then we adjust the high/low values accordingly.
#        It's obviously not ideal though...
        
        high = 1
        low = 0
        action = sigmoid(action)
#        high = 100 if action < 100 else action
#        low = -100 if action > -100 else action
        
        action = scale_between(action, self.y_min, self.y_max, low, high)
        
        if self.count == len(self.x_coords)-1:
            self.state = self.y_f
            #For debugging purposes, if an error happens here, please lmk and send me the output!
            try:
                assert(self.y_min-0.1 <= self.state <= self.y_max+0.1)
            except:
                raise(AssertionError(f"Assertion error at path end, y_min-0.1 <= state <= self.y_max+0.1 not satisfied.\nBelow are the values\ny_min = {self.y_min}\ny_max = {self.y_max}\nstate = {self.state}"))
            
        else:
            self.state = action if not np.isnan(action) else np.median((self.y_min, self.y_max))
            
            self.action_space = Box(low = self.y_min, high = self.y_max)
            self.observation_space = Box(low = self.y_min, high = self.y_max)
            #For debugging purposes, if an error happens here, please lmk and send me the output!
            try:
                assert(self.y_min-0.1 <= self.state <= self.y_max+0.1)
            except:
                raise(AssertionError(f"Assertion error before path end, y_min-0.1 <= state <= self.y_max+0.1 not satisfied.\nBelow are the values\ny_min = {self.y_min}\ny_max = {self.y_max}\nstate = {self.state}"))
            
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
            elif self.best_t == np.inf:
                self.best_y_coords = self.y_coords
                reward = 1
                reward = 1/self.t
                self.best_t = self.t
                print(self.best_t)
                print(f"\nNew Best Time = {self.best_t}, \t Optimal Time = {self.optimal_time}")
                if not self.interactive:
                    self.render()
            elif self.t < self.best_t:
                self.best_y_coords = self.y_coords
                reward = 1
                reward = 1/self.t
                self.best_t = self.t
                print(f"\nNew Best Time = {self.best_t}, \t Optimal Time = {self.optimal_time}")
                if not self.interactive:
                    self.render()
            elif self.t >= self.best_t:
                reward = -1
                reward = 1/self.t
            
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

#https://programtalk.com/python-examples/rl.agents.DDPGAgent/
#https://github.com/tensorneko/keras-rl2/pull/18
#https://keras-rl.readthedocs.io/en/latest/agents/ddpg/
def build_actor(env):
    
    nb_actions = env.action_space.shape[0]
    actor = Sequential()

    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    
#    actor.add(Embedding(input_dim=100, output_dim=10))
##     The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
#    actor.add(LSTM(10, return_sequences=True))
#
##     The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
#    actor.add(SimpleRNN(10))
#
#    actor.add(Dense(nb_actions))
#    actor.add(Activation('linear'))
    
    return actor

def build_critic(env):
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input

def build_agent(env, actor, critic, action_input):
    nb_actions = env.action_space.shape[0]
    memory = SequentialMemory(limit = 50000, window_length = 1)
    ddpg = DDPGAgent(nb_actions, actor, critic, action_input, memory = memory, batch_size = 32)
    return ddpg

if __name__ == '__main__':
#    test = BrachistohroneEnv(x_end = 1, x_start = -1, num_x_points = 10, y_start = 10, y_end = 0)
    test = BrachistohroneEnv(x_end = 10, x_start = 0.1, num_x_points = 100, y_start = 10, y_end = 0, point_dist = "linear")
    print(test.action_space.sample())
    print(test.observation_space.sample())
    states = test.observation_space.shape
    actions =  test.action_space.shape
    print(states,actions)
    print(test.action_space.high,test.action_space.low)
    actor = build_actor(test)
    actor.summary()
#    for layer in actor.layers:
#        print(actor.input_shape)
    critic, action_input = build_critic(test)
    critic.summary()
    
    ddpg = build_agent(test, actor, critic, action_input)
    ddpg.compile([LazyAdam(1),LazyAdam(1)])
    
    adjust_model = AdjustModel(update_every = 0, lr_factor = 1, update_lr = False, reset_weights = False)
    ddpg.fit(test, nb_steps = 5e4, visualize = False, callbacks=[adjust_model])
    print("Best time = ",test.best_t)
    
    X = test.x_coords
    y = np.array(test.best_y_coords)
    plt.plot(X,y, label = f"Time Taken = {test.best_t:.3f} seconds")
    plt.scatter((X[0], X[-1]), (y[0], y[-1]))
    
    np.savetxt("RL_Brachistochrone.txt",np.concatenate((np.expand_dims(X,axis=1),np.expand_dims(y,axis=1)), axis=1))
    best_vals = np.loadtxt("RL_Brachistochrone.txt")
    
    X = X.reshape(-1,1)
    
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
    model.fit(X, y)
    print(model.latex())
    print(model.sympy())
    model_selection = lambdify(symbols('x0'), model.sympy())
    
    x_points = np.linspace(0.1, 10, 1000)
    plt.plot(x_points, model_selection(x_points), label = rf"f(x) = ${model.latex()}$")
    plt.legend()
    plt.savefig("RL_Brachistochrone.png",dpi=5*96)
        
