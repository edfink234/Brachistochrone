'''
****************************************************************************
*     Reinforcement-Learning: Discovering the Brachistochrone Equation     *
****************************************************************************

Using reinforcement learning and symbolic regression to discover the
non-parametric form of the Brachistochrone
'''

# Modified Files
# ==============
# /Users/edwardfinkelstein/MachineLearning/myvenv/lib/python3.9/site-packages/keras/engine/training_utils_v1.py -> Commented out line 711-721
# /Users/edwardfinkelstein/MachineLearning/myvenv/lib/python3.9/site-packages/rl/agents/ddpg.py: comment out everything line that has uses_learning_phase (except lines 80-82), i.e., lines 153-154, 310-311
# if hasattr(actor.output, '__len__') and len(actor.output) > 1: -> if hasattr(actor.output, '__shape__') and len(actor.output) > 2:
# if hasattr(critic.output, '__len__') and len(critic.output) > 1: -> if
# hasattr(critic.output, '__shape__') and len(critic.output) > 2:

import sys
sys.setrecursionlimit(10000)
import warnings
import sympy as sp
from sympy import sin, cos
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents import DDPGAgent
from tensorflow.keras.layers import Dense, Flatten, Activation, Input
from tensorflow.keras.models import Sequential
from tensorflow_addons.optimizers import LazyAdam
from keras.layers import Concatenate
from tensorflow.keras import Model, initializers
import tensorflow.keras.backend as K
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import ridder, fsolve
from gym import Env
from gym.spaces import Box
import operator
from operator import add, sub, mul
from deap import base, creator, tools, gp, algorithms
from math import isclose
import random
import argparse
import re
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


warnings.filterwarnings("ignore")

'''
+++++++++++++++++++++++
+     AdjustModel     +
+++++++++++++++++++++++

Class to update the learning rate and/or reset the weights of the agent.

Parameters
==========

 - update_every: Number of iterations before updating the learning_rate and/or reset the weights and biases

 - lr_factor: Factor to multiply the learning rate by every 'update_every' iterations

 - update_lr: Whether or not to update the learning rate every 'update_every' iterations. Only supports updating the actor's learning rate

 - reset_weights: Whether or not to reset the weights and biases every 'update_every' iterations
'''


class AdjustModel(tf.keras.callbacks.Callback):
    def __init__(
            self,
            update_every=0,
            lr_factor=1,
            update_lr=False,
            reset_weights=False):
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
                    # Only the actor's learning rate is updated
                    self.model.actor.optimizer.learning_rate.assign(
                        self.model.actor.optimizer.learning_rate * self.lr_factor)

#                https://stackoverflow.com/a/70450963/18255427
                if self.reset_weights:
                    for l in self.model.layers:
                        if hasattr(l, "kernel_initializer"):
                            kernel_initializer = initializers.glorot_uniform(
                                seed=np.random.randint(0, 1000))
                            l.kernel.assign(
                                kernel_initializer(
                                    tf.shape(
                                        l.kernel).eval(
                                        session=tf.compat.v1.Session())))
                        if hasattr(l, "bias_initializer"):
                            bias_initializer = initializers.RandomNormal(
                                stddev=0.01, seed=np.random.randint(0, 1000))
                            l.bias.assign(
                                bias_initializer(
                                    tf.shape(
                                        l.bias).eval(
                                        session=tf.compat.v1.Session())))
                        if hasattr(l, "recurrent_initializer"):
                            recurrent_initializer = initializers.RandomNormal(
                                stddev=0.01, seed=np.random.randint(0, 1000))
                            l.recurrent_kernel.assign(
                                recurrent_initializer(
                                    tf.shape(
                                        l.recurrent_kernel).eval(
                                        session=tf.compat.v1.Session())))
                self.curr_step = self.model.step


'''
sigmoid
=======
The sigmoid activation function; takes in a value from -infinity to +infinity and returns a value b/t 0 and 1
'''


def sigmoid(x, c=1):
    return 1 / (1 + np.exp(c * x))


'''
scale_between
=============
Scales a number unscaledNum that originally lies between Min and Max to lie between minAllowed
and maxAllowed
'''


def scale_between(unscaledNum, minAllowed, maxAllowed, Min, Max):
    return (maxAllowed - minAllowed) * \
        (unscaledNum - Min) / (Max - Min) + minAllowed


'''
BrachistohronePoints
====================
Function to generate the coordinate of the Brachistochrone curve from
start_point to end_point with gravitational acceleration g

Parameters
----------
 - start_point: list of starting x coordinate and y coordinate

 - end_point: list of end x coordinate and y coordinate

 - g: gravitational acceleration
'''


def BrachistohronePoints(start_point, end_point, g=9.80665):
    x_start, y_start = start_point
    x_end, y_end = end_point
    # Normalizing coordinates so starting point is (0,0)
    x_diff, y_diff = 0 - x_start, 0 - y_start
    x_start, x_end = x_start + x_diff, x_end + x_diff
    y_start, y_end = y_start + y_diff, y_end + y_diff
    '''
    The Brachistochrone equations are (for x(t) = x_end and y(t) = y_end)

    x(t) = R(t-sin(t))
    y(t) = R(cos(t)-1)

    simplifying

    x(t) = (t-sin(t))
    y(t) = (cos(t)-1)

    -> x/y = (t-sin(t))/(cos(t)-1)

       ***********************************
    -> * 0 = (t-sin(t))/(cos(t)-1) - x/y *
       ***********************************

    Which is coded in 'func', where C = x/y
    '''
    def func(t, C):
        return (t - np.sin(t)) / (1 - np.cos(t)) + C

#    Has to be solved numerically: to consistently get the right points.
#    we needed to combine scipy.ridder and scipy.fsolve

    t = ridder(func, disp=0, args=(x_end / y_end,), a=0, b=2 * np.pi)
    a = x_end / (t - np.sin(t))
#    If ridder fails, then fsolve should work, and vice versa.
#    The following example code shows that using both methods
#    results in one correct solution:
#    https://www.sololearn.com/compiler-playground/cdsL5oJBXS7v
    if not isclose(func(t, x_end / y_end), 0, abs_tol=0.1):
        root = fsolve(func, [1], args=(x_end / y_end,))
        t = root[0]
        a = x_end / (t - np.sin(t))

    theta = np.linspace(0, t, 1000)
    # Shifting back coordinates to what they were originally
    x_points = a * (theta - np.sin(theta)) - x_diff
    y_points = -a * (1 - np.cos(theta)) - y_diff
    optimal_time = t * np.sqrt(a / g)  # Brachistochrone optimal time

    return x_points, y_points, optimal_time


'''
+++++++++++++++++++++++++++++
+     BrachistohroneEnv     +
+++++++++++++++++++++++++++++

Reinforcement-Learning Envrionment to discover the optimal path
to get from (x_start, y_start) to (x_end, y_end) via only the
acceleration of gravity g. Inherits from gym.Env. Coordinates are swapped
automatically if one enters x_start > x_end or y_end > y_start. An error will
be raised if one enters x_start = x_end and/or y_start = y_end

Parameters
==========

 - x_start: if point_dist = "linear", then it's the starting x-coordinate
            if point_dist = "log", then it's the starting x-coordinate raised to the power of 10
            -> e.g. x_start = 0 means starting x coordinate is 10^0 = 1

 - x_end: if point_dist = "linear", then it's the ending x-coordinate
          if point_dist = "log", then it's the ending x-coordinate raised to the power of 10
          -> e.g. x_end = 1 means ending x coordinate is 10^1 = 10

 - y_start: starting y-coordinate

 - y_end: ending y-coordinate

 - iterations: Number of iterations to train the agent. NOTE: The number of iterations to construct a full path is equal to num_x_points

 - interactive: Determines whether to display the graph where the mouse doesn't buffer, 'interactive = True', or to display the graph but have the mouse buffer, 'interactive = False'.
                * interactive = True: Will result in longer training times, but will allow the user to hover over the real-time display of the plot and see the coordinates display
                * interactive = False: Will result in shorter training times, but will the user will not be able to see the coordinates when hovering over the live plot with the mouse

 - num_x_points: Number of points the agent will use to construct a path from (x_start, y_start) to (x_end, y_end) via only the acceleration of gravity g.

 - g: acceleration of gravity in meters per second squared (m/s^2)

 - point_dist: whether to use a linear space "linear" (np.linspace), or a logarithmic space "log" (np.logspace) for the x-coordinates.
               * point_dist = "linear": The starting and ending x-coordinates will be x_start and x_end
               * point_dist = "log": The starting and ending x-coordinates will be 10^x_start and 10^x_end

 - autoscale: whether to scale the graph if the plot trails off the page True, or to keep the x and y limits as originally determined False
              * auto_scale = True: Every time the plot updates, the entire canvas will be redrawn. In principle slower than auto_scale = False, but the effect in this example is not noticeable (at least to me)
              * auto_scale = False: Only the points that change are redrawn on every update; the axes do not get updated after the initial configuration. This is done using blitting, see: https://matplotlib.org/stable/tutorials/advanced/blitting.html

 - activation: Choice of activation function for the continuous action of the agent, either sigmoid "sigmoid" or tanh "tanh"
               * activation = "tanh": Will use the tanh activation function to condense the action from -infinity to infinity -> -1 to 1. The output is then internally fed into scale_between which scales the action between the minimum and maximum y-value allowed by classical mechanics

                * activation = "sigmoid": Will use the sigmoid activation function to condense the action from -infinity to infinity -> 0 to 1. The output is then internally fed into scale_between which scales the action between the minimum and maximum y-value allowed by classical mechanics.

- activation_factor: Argument for activation function, where the activation function, A(x), is calculated as A((x_end-x_start)*activation_factor*x) for point_dist = "linear" and A((10^{x_end}-10^{x_start})*activation_factor*x) for point_dist = "log"
'''


class BrachistohroneEnv(Env):
    def __init__(
            self,
            x_start=0,
            x_end=1,
            y_start=10,
            y_end=0,
            iterations=10000,
            interactive=False,
            num_x_points=50,
            g=9.80665,
            point_dist="log",
            autoscale=False,
            activation="tanh",
            activation_factor=0.6):

        if y_end > y_start:
            y_start, y_end = y_end, y_start
        if x_end < x_start:
            x_start, x_end = x_end, x_start
        if x_start == x_end:
            raise (AssertionError(f"x_start and x_end cannot be equal!"))
        if y_start == y_end:
            raise (AssertionError(f"y_start and y_end cannot be equal!"))

        self.y_i = y_start
        self.y_f = y_end
        self.state = self.y_i
        self.g = g
        self.prev_state = self.y_i
        self.v = 0  # initial velocity
        y_coord = self.y_i
        # The minimum height such that the ball would be able to just reach the
        # end point with velocity 0.
        self.y_min = self.y_f - (self.v**2) / (2 * self.g)
        # The maximum height that can be reached by a ball with velocity self.v
        # rolling up a frictionless incline
        self.y_max = (self.v**2) / (2 * self.g) + y_coord
        self.action_space = Box(low=self.y_min, high=self.y_max)
        self.observation_space = Box(low=self.y_min, high=self.y_max)
        self.y_coords = [self.y_i]
        self.count = 0
        self.iterations = iterations
        self.completed_iterations = 0
        self.done = False
        self.t = 0  # total time
        self.best_y_coords = []
        # best time; since a path hasn't been constructed yet, it's undefined,
        # minus well be infinity
        self.best_t = np.inf
        self.autoscale = autoscale
        self.fig, self.ax = plt.subplots()
        if not self.autoscale:
            self.ln, = self.ax.plot([], [], animated=True)
        else:
            self.ln, = plt.plot([], [])
        if point_dist == "linear":
            x_points, y_points, self.optimal_time = BrachistohronePoints(
                (x_start, y_start), (x_end, y_end))
            self.disp = x_end - x_start
            plt.scatter((x_start, x_end), (y_start, y_end))
        else:
            x_points, y_points, self.optimal_time = BrachistohronePoints(
                (10**x_start, y_start), (10**x_end, y_end))
            self.disp = 10**x_end - 10**x_start
            plt.scatter((10**x_start, 10**x_end), (y_start, y_end))
        self.activation = activation
        self.activation_factor = activation_factor
        self.iterations_since_new_best = 0
        plt.plot(x_points, y_points,
                 label=f"Best Time = {self.optimal_time:0.3f} seconds")
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        if point_dist == "linear":
            self.x_coords = np.linspace(x_start, x_end, num_x_points)
        else:
            self.x_coords = np.logspace(x_start, x_end, num_x_points)

        if not self.autoscale:
            self.ln.set_data(self.x_coords, [0] * len(self.x_coords))
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.ax.draw_artist(self.ln)
            self.fig.canvas.blit(self.fig.bbox)
        else:
            self.ln.set_data(self.x_coords, [0] * len(self.x_coords))
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()

        self.interactive = interactive

    def step(self, action):
        action = action[0]
        self.count += 1
        self.y_min = self.y_f - (self.v**2) / (2 * self.g)
        self.y_max = (self.v**2) / (2 * self.g) + self.prev_state

        if self.activation == "sigmoid":
            high = 1
            low = 0
            action = sigmoid(action * (self.disp * self.activation_factor))
        else:  # tanh
            high = 1
            low = -1
            action = np.tanh(action * (self.disp * self.activation_factor))

        action = scale_between(action, self.y_min, self.y_max, low, high)

        if self.count == len(self.x_coords) - 1:
            self.state = self.y_f
            # For debugging purposes, if an error happens here, please lmk and
            # send me the output!
            try:
                assert (self.y_min - 0.1 <= self.state <= self.y_max + 0.1)
            except BaseException:
                raise (AssertionError(
                    f"Assertion error at path end, y_min-0.1 <= state <= self.y_max+0.1 not satisfied.\nBelow are the values\ny_min = {self.y_min}\ny_max = {self.y_max}\nstate = {self.state}"))

        else:
            self.state = action if not np.isnan(
                action) else np.median((self.y_min, self.y_max))

            self.action_space = Box(low=self.y_min, high=self.y_max)
            self.observation_space = Box(low=self.y_min, high=self.y_max)
            # For debugging purposes, if an error happens here, please lmk and
            # send me the output!
            try:
                assert (self.y_min - 0.1 <= self.state <= self.y_max + 0.1)
            except BaseException:
                raise (AssertionError(
                    f"Assertion error before path end, y_min-0.1 <= state <= self.y_max+0.1 not satisfied.\nBelow are the values\ny_min = {self.y_min}\ny_max = {self.y_max}\nstate = {self.state}"))

        '''
        (self.x_coords[self.count-1], self.prev_state)
        .
        |\
        | \
        |  \
        |   \
        |    \
   opp  |     \\  d
        |      \
        |       \
        |        \
        |      θ  \
        |__________\
            adj     |=> (self.x_coords[self.count], self.state)

        '''
        # Computing the quantities in the above picture:
        opp = abs(self.state - self.prev_state)
        adj = abs(self.x_coords[self.count] - self.x_coords[self.count - 1])
        theta = np.arctan(opp / adj)  # angle θ
        d = np.sqrt(opp**2 + adj**2)

        # Calculating the current velocity v_f:
        if self.state > self.prev_state:  # if the current y-coordinate is greater than the previous one
            v_f = np.sqrt(np.abs(self.v**2 - 2 * self.g * d * np.sin(theta)))
        else:  # as is the case in the picture above
            v_f = np.sqrt(np.abs(self.v**2 + 2 * self.g * d * np.sin(theta)))

        # Calculating the time to travel from the previous point to the current
        # one
        step_time = abs(v_f - self.v) / (self.g * np.sin(theta)
                                         ) if not isclose(theta, 0) else adj / v_f
        self.t += step_time  # Add step time to total time of path
        self.v = v_f  # set current velocity

        self.prev_state = self.state  # Set current y-coordinate
        self.y_coords.append(self.prev_state)  # Append y-coordinate to path

        '''
        Reward Function
        ===============
        * After each step, reward = 1/step_time if step_time is a finite number, else reward = 0
        * After each iteration, i.e., when one full path has been constructed, multiply the reward of the agent by one of the following factors:
          - (number of iterations since new best)*-10: if the time of travel for the current iteration is greater than or equal to the shortest (best) time achieved by the agent, or if the time of travel for the current iteration is not a number
          - (number of iterations since new best)*10: if time of travel for the current iteration is less than the shortest (best) time achieved by the agent
          - 10: for the first path that is successfully created
        '''

        reward = 1

        if (len(self.y_coords) == len(self.x_coords)):
            self.completed_iterations += 1
            if np.isnan(self.t) or self.t == np.inf:
                self.iterations_since_new_best += 1
                reward *= -10 * self.iterations_since_new_best
            elif self.best_t == np.inf:
                self.best_y_coords = self.y_coords
                reward *= 10
                self.best_t = self.t
                print(self.best_t)
                print(
                    f"\nNew Best Time = {self.best_t}, \t Optimal Time = {self.optimal_time}")
                if not self.interactive:
                    self.render()
            elif self.t < self.best_t:
                self.best_y_coords = self.y_coords
                reward *= 10 * (self.iterations_since_new_best)
                self.iterations_since_new_best = 0
                self.best_t = self.t
                print(
                    f"\nNew Best Time = {self.best_t}, \t Optimal Time = {self.optimal_time}")
                if not self.interactive:
                    self.render()
            elif self.t >= self.best_t:
                self.iterations_since_new_best += 1
                reward *= -10 * self.iterations_since_new_best

            if self.interactive:
                self.render(1)
            self.reset()
        if self.completed_iterations == self.iterations:
            self.done = True
        if reward == np.inf:
            reward = 0
        return self.state, reward, self.done, {}

    '''
    renders the matplotlib window that shows the optimal path and the best one achieved my the agent
    '''

    def render(self, pause_time=0.1):
        if len(self.x_coords) == len(self.best_y_coords):
            self.ln.set_ydata(self.best_y_coords)
            if not self.autoscale:
                self.fig.canvas.restore_region(self.bg)
                self.ax.draw_artist(self.ln)
                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()
            else:
                self.ax.relim()
                self.ax.autoscale_view()
                plt.draw()
            plt.pause(pause_time)

    '''
    Called every time a new path is finished being created, resets attributes
    '''

    def reset(self):
        self.state = self.y_i
        self.prev_state = self.y_i
        self.v = 0  # initial velocity
        y_coord = self.y_i
        self.y_min = self.y_f - (self.v**2) / (2 * self.g)
        self.y_max = (self.v**2) / (2 * self.g) + y_coord
        self.action_space = Box(low=self.y_min, high=self.y_max)
        self.observation_space = Box(low=self.y_min, high=self.y_max)
        self.y_coords = [self.y_i]
        self.count = 0
        self.t = 0
        self.completed_iterations = 0
        return self.state


'''
build_actor
===========
Builds Actor deep-neural network for DDPG agent using the
Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
'''


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
    return actor


'''
build_critic
============
Builds critic deep-neural network for DDPG agent
Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
'''


def build_critic(env):
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input


'''
build_agent
============
Builds DDPG agent using actor and critic deep-neural networks
Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
'''


def build_agent(env, actor, critic, action_input):
    nb_actions = env.action_space.shape[0]
    memory = SequentialMemory(limit=int(1e5), window_length=1)
    ddpg = DDPGAgent(nb_actions, actor, critic, action_input,
                     memory=memory, batch_size=32)
    return ddpg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_only_SR", action="store_true", help="Run only SR")
    args = parser.parse_args()
    run_only_Sr = args.run_only_SR

    if not run_only_Sr:
        x_start, x_end, = 0, np.pi  # starting and ending x coordinates
        y_start, y_end = 0, -10  # starting and ending y coordinates
        num_x_points = 12  # number of points we wish to sample

        # create environment
        test = BrachistohroneEnv(
            x_end=x_end,
            x_start=x_start,
            num_x_points=num_x_points,
            y_start=y_start,
            y_end=y_end,
            point_dist="linear",
            autoscale=True,
            activation="tanh",
            activation_factor=0.6)

        actor = build_actor(test)  # actor neural network
        actor.summary()  # summary of actor neural network architecture
        critic, action_input = build_critic(test)  # critic neural network
        critic.summary()  # summary of critic neural network architecture

        ddpg = build_agent(test, actor, critic, action_input)
        # Giving actor and critic neural networks Adam optimizers with learning
        # rate 1e-5 and 1e-4 respectively. Generally a good idea to make the
        # actor a slower learner than the critic. See the brief explanation
        # here:
        # https://www.reddit.com/r/reinforcementlearning/comments/lsza9m/why_different_learning_rates_for_actor_and_critic/
        ddpg.compile([LazyAdam(1e-5), LazyAdam(1e-4)])

        # Defining a callback that gets called every 10000 steps
        adjust_model = AdjustModel(
            update_every=10000,
            lr_factor=0.95,
            update_lr=True,
            reset_weights=False)
        try:
            # Start the training
            ddpg.fit(test, nb_steps=1e10, visualize=False,
                     callbacks=[adjust_model])
        except Exception as exception:
            # Sometimes an unexpected error occurs. In that case, let's save
            # the model and stop training.
            print(
                f"\n\nAn exception of type {exception.__class__.__name__} was raised and caught\n")

        print("Best time = ", test.best_t)
        try:
            input("\nPress Enter for Symbolic Regression, or ctr-c to exit: ")
        except KeyboardInterrupt:
            sys.exit()

        X = test.x_coords
        y = np.array(test.best_y_coords)
        plt.title(f"Number of Sampled Points = {num_x_points}")
        plt.plot(X, y, label=f"Time Taken = {test.best_t:.3f} seconds")
        plt.scatter((X[0], X[-1]), (y[0], y[-1]))

        np.savetxt("RL_Brachistochrone_deap.txt", np.concatenate(
            (np.expand_dims(X, axis=1), np.expand_dims(y, axis=1)), axis=1))
        np.savetxt("RL_Brachistochrone_deap_best_time.txt", np.array([test.best_t]))

    else:  # If symbolic regression didn't go well and you want to redo it
        best_vals = np.loadtxt("RL_Brachistochrone_deap.txt")
#        best_vals = np.loadtxt("train_data_25p.csv", delimiter=',')
        X = best_vals[:, 0]
        y = best_vals[:, 1]
#        X = best_vals[0]
#        y = best_vals[1]
        x_points, y_points, optimal_time = BrachistohronePoints(
            (X[0], y[0]), (X[-1], y[-1]))
        plt.plot(x_points, y_points,
                 label=f"Best Time = {optimal_time:0.3f} seconds")
        best_t = np.loadtxt("RL_Brachistochrone_deap_best_time.txt")
        plt.plot(X, y, label=f"Time Taken = {best_t:.3f} seconds")
        plt.scatter((X[0], X[-1]), (y[0], y[-1]))
    
    # Symbolic Regression
    # ===================
    X = X.reshape(-1, 1)
    # Define custom operators and unary functions
    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    
    pset.addEphemeralConstant("const", lambda: random.uniform(-10, 10))
    def inv(x):
        x = np.array(x)
        return 1/x if np.all(x) else x
    pset.addPrimitive(inv, 1)
    
    # Define custom loss function
    def custom_loss(y_pred, y):
        if np.allclose(y_pred, y_pred[0]):
            return np.inf
        loss = np.sum((y_pred - y) ** 2)
        return loss
        
    # Define fitness function (minimize the loss)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    def evaluate_individual(individual, X, y):
        func = gp.compile(individual, pset)
        y_pred = np.array([func(x) for x in X])
        loss = custom_loss(y_pred, y)
        return loss,
        
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define other DEAP components and settings
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("mutate", gp.mixedMutate, expr=toolbox.expr, pset=pset, prob = [1, 0, 0])
#    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True)
    toolbox.register("select", tools.selStochasticUniversalSampling)
    toolbox.register("evaluate", evaluate_individual, X=X, y=y)
    
    def feasible(individual):
        """Feasibility function for the individual. Returns True if feasible False
        otherwise."""
        tree = gp.PrimitiveTree(individual)
        if len(tree) <= 9 and tree.height <= 2:
            return True
        return False
    
    def distance(individual):
        """A distance function to the feasibility region."""
        
        tree = gp.PrimitiveTree(individual)
        complexity = len(tree)
        height = tree.height
        return 0 if complexity <= 9 and height <= 2 else (complexity - 9)**2 + (height - 2)**2
    
#    https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, np.inf, distance))
    
    pop = toolbox.population(n=10)
    best_loss = np.inf
    best_individual = None
    best_func = None
        
    # Run the evolutionary algorithm
    try:
        while True:
            # Run one generation of the algorithm
            algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.5, ngen=1, stats=None, verbose = False)
            
            # Calculate the loss value of the best individual
            individual = tools.selBest(pop, k=1)[0]
    
            func = gp.compile(individual, pset)
            try:
                y_pred = func(X)
                y_pred = y_pred.flatten()
            except:
                if isinstance(y_pred, float):
                    y_pred = np.full_like(X, fill_value = y_pred)
                else:
                    print(func(X), type(func(X)), X.shape)
                    print("Exiting...")
                    exit()
            loss = np.sum((y_pred-y)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_individual = individual
                best_func = func
                print(f"New Best Loss = {best_loss}")
                print("Best individual:", best_individual)
                print("Complexity =",len(gp.PrimitiveTree(individual)))
                print("Height =",gp.PrimitiveTree(individual).height)
                
    except KeyboardInterrupt:
        print(f"Best Loss = {best_loss}")
        x_start = X[0]
        x_end = X[-1]
        x_points = np.linspace(x_start, x_end, 1000)
        
        ARG0 = sp.symbols('ARG0')
        expr = sp.sympify(str(best_individual))
        expr = sp.latex(expr)
        
        floats = [float(i) for i in re.findall(r"\d+\.\d+",expr)]
        non_floats = re.split(r"\d+\.\d+",expr)
        
        new_expr = ""
        length = min(len(floats), len(non_floats))
        
        float_start_idx = expr.index(str(floats[0]))
        non_float_start_idx = expr.index(str(non_floats[0]))

        if (non_float_start_idx < float_start_idx): #Then it starts with non float
            for i in range(length):
                new_expr += non_floats[i] + f"{floats[i]:0.3f}"
            new_expr += non_floats[-1]
        else: #Then it starts with float
            for i in range(length):
                new_expr += f"{floats[i]:0.3f}" + non_floats[i]
            new_expr += f"{floats[-1]:0.3f}"
        
        plt.plot(x_points, best_func(x_points),
                 label=rf"f(x) = ${new_expr}$")
        plt.legend()
        # Save the figure and show your friends! :)
        if best_loss < 11.412352072449064:
            print("New best")
            plt.savefig("RL_Brachistochrone_deap.png", dpi=5 * 96)
