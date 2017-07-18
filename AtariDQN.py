#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:21:59 2017

@author: tobykreiman
"""

import gym
import numpy as np
import scipy
import tensorflow as tf
import os
import csv
import sys

#env = gym.make("Breakout-v0")

possible_actions = 4
state_shape = [84,84,4]

last_4_frames = []
state_4_frames = []


checkpoint_base_dir = 'checkpoints_DQN/'

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reward_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None

def update_paths(env_name):
    """
    Update the path-names for the checkpoint-dir and log-files.
    
    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.
    
    :param env_name:
        Name of the game-environment you will use in OpenAI Gym.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path
    

    # Add the environment-name to the checkpoint-dir.
    checkpoint_dir = os.path.join(checkpoint_base_dir, env_name)

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")


def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


class Log:
    """
    Base-class for logging data to a text-file during training.
    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and other aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        """Set the path for the log-file. Nothing is saved or loaded yet."""

        # Path for the log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(count_episodes, count_states, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.
        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            #self.count_episodes, self.count_states, *data = zip(*reader)
            self.count_episodes = 0
            self.count_states = 0

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param reward_episode:
            Reward for one episode.
        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """

        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.
        It sets self.count_episodes, self.count_states, self.episode and self.mean
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the episode reward.
        self.episode = self.data[0]

        # Get the mean reward.
        self.mean = self.data[1]


class LogQValues(Log):
    """Log the Q-Values during training."""

    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        """
        Write basic statistics for the Q-values to file.
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """

        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values),
                                                          np.mean(q_values),
                                                          np.max(q_values),
                                                          np.std(q_values))

        self._write(count_episodes=count_episodes,
                    count_states=count_states,
                    msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.
        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]

class LinearControlSignal:
    """
    A control signal that changes linearly over time.
    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    
    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.
        :param start_value:
            Start-value for the control signal.
        :param end_value:
            End-value for the control signal.
        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.
        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value
    
    
class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.
    
    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.
    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.
    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """
        
        :param num_actions:
            Number of possible actions in the game-environment.
        :param epsilon_testing:
            Epsilon-value when testing.
        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.
            
        :param start_value:
            Starting value for linearly decreasing epsilon.
        :param end_value:
            Ending value for linearly decreasing epsilon.
        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.
        
        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.
         
        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.
        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.
        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon
    
    
    

class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural Network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, size, num_actions, discount_factor=0.97):
        """
        
        :param size:
            Capacity of the replay-memory. This is the number of states.
        :param num_actions:
            Number of possible actions in the game-environment. 
        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Whether the life had ended in each state of the game-environment.
        self.end_life = np.zeros(shape=size, dtype=np.bool)

        # Whether the episode had ended (aka. game over) in each state.
        self.end_episode = np.zeros(shape=size, dtype=np.bool)
        
        self.next_states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Estimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

    def is_full(self):
        #Return boolean whether the replay-memory is full
        return self.num_used == self.size

    def used_fraction(self):
        #Return the fraction of the replay-memory that is used
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode, next_state):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc. 
        
        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.
        :param q_values: 
            The estimated Q-values for the state.
        :param action: 
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        :param end_life:
            Boolean whether the agent has lost a life in this state.
         
        :param end_episode: 
            Boolean whether the agent has lost all lives aka. game over
            aka. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elements in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode
            self.next_states[k] = next_state

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.
        
        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used-1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]
            end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life or end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

        self.print_statistics()

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.
        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.
        
        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err<self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.
        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]
        next_states_batch = self.next_states[idx]
        rewards_batch = self.rewards[idx]
        end_episode_batch = self.end_episode[idx]

        return states_batch, q_values_batch, next_states_batch, rewards_batch, end_episode_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.
        
        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.
        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.
        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            #print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))


# Pass in [210, 160, 3] rgb image and returns [210, 160] gray scale image
def rgb2_grayScale(rgb):
    gray = np.mean(rgb, -1)
    return gray

#Convert Image (given as [210, 160, 3]) into [84,84]
def process_image(image):
    #First Make Image Grayscale: [210, 160]
    gray = rgb2_grayScale(image)
    # Reshape image
    img = scipy.misc.imresize(gray, size=[84, 84], interp='bicubic')
    
    return img


def convert_shape_last_4_frames(last_4_frames):
    converted = np.zeros([84,84,4])
    for i in range(4):
        for j in range(84):
            for g in range(84):
               converted[j][g][i] = last_4_frames[i][j][g]

    return converted

class NeuralNetwork:
    
    def __init__(self, replay_memory, num_actions):
        self.replay_memory = replay_memory
        self.num_actions = num_actions
        
        self.x = tf.placeholder(tf.float32, shape=[None,84,84,4])
        self.target_q_values = tf.placeholder(tf.float32, shape=[None, possible_actions])

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            
        #Create Deep Neural Network
        # 3 convolutional layers
        # 2 fully connected layers
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        padding = 'SAME'
        
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')
        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')
        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states, self.count_states + 1)
        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes, self.count_episodes + 1)

        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # Activation function for all convolutional and fully-connected
        # layers, except the last.
        activation = tf.nn.relu

        net = self.x

        #Values of the Neural Network are modified from teh one used by Deep Mind
        # First convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                       filters=32, kernel_size=8, strides=4,
                       padding=padding,
                       kernel_initializer=init, activation=activation)

        # Second convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                               filters=64, kernel_size=4, strides=2,
                               padding=padding,
                               kernel_initializer=init, activation=activation)

        # Third convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                               filters=64, kernel_size=3, strides=1,
                               padding=padding,
                               kernel_initializer=init, activation=activation)
        
        # Flatten output of the last convolutional layer so it can
        # be input to a fully-connected (aka. dense) layer.

        net = tf.contrib.layers.flatten(net)

        # First fully-connected (aka. dense) layer.
        net = tf.layers.dense(inputs=net, name='layer_fc1', units=512,
                              kernel_initializer=init, activation=activation)
        
        # Second fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_output', units=possible_actions,
                              kernel_initializer=init, activation=None)
        # The output of the Neural Network is the estimated Q-values
        # for each possible action in the game-environment.
        self.predicted_q_values = net

        #Original Neural Net

        #1st convolutional layer: 32 filters 8x8, stride 4
        #2nd convolutional layer: 64 filters 4x4, stride 2
        #3rd convolutional layer: 64 filters 3x3, stride 1
        #4th fully connected layer with 512 units
        #5th output fully connected layer with output for each valid of action
        
        
        # L2 loss
        squared_error = tf.square(self.predicted_q_values - self.target_q_values)
        sum_squared_error = tf.reduce_sum(squared_error, reduction_indices=[1])
        self.loss = tf.reduce_mean(sum_squared_error)       

        # Testing different optimizers
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)



        self.load_checkpoint()
        self.clone_network()

#replay_memory = ReplayMemory(200000, possible_actions)
    #Takes last 4 frames as input
    def get_q_values(self, states):
        values = self.sess.run(self.predicted_q_values, feed_dict={self.x: states})
    
        return values
    
    def get_cloned_q_values(self, states):
        values = self.sess.run(self.clone_model, feed_dict={self.x: states})
        return values

    def clone_network(self):
        conv1_weights = self.get_weights_variable("layer_conv1")
        conv2_weights = self.get_weights_variable("layer_conv2")
        conv3_weights = self.get_weights_variable("layer_conv3")
        fc1_weights = self.get_weights_variable("layer_fc1")
        output_weights = self.get_weights_variable("layer_output")

        layer_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, conv1_weights, strides=[1, 4, 4, 1], padding='SAME'))

        layer_conv2 = tf.nn.relu(tf.nn.conv2d(layer_conv1, conv2_weights, strides=[1, 2, 2, 1], padding='SAME'))

        layer_conv3 = tf.nn.relu(tf.nn.conv2d(layer_conv2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME'))

        flattened = tf.contrib.layers.flatten(layer_conv3)
        layer_fc1 = tf.nn.relu(tf.matmul(flattened, fc1_weights))
        layer_output = tf.nn.relu(tf.matmul(layer_fc1, output_weights))

        
        self.clone_model = layer_output

    
    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.
        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """
        # The tf.layers API uses this name for the weights in a conv-layer.
        variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    
    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """
        self.sess.run(tf.global_variables_initializer())
        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            # Try and load the data in the checkpoint.
            self.saver.restore(self.sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            self.sess.run(tf.global_variables_initializer())

    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.sess,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)
        
        print("Saved checkpoint.")
    
    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.sess.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.sess.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.sess.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.sess.run(self.count_episodes_increase)
        
    def train(self, min_epochs=1.0, max_epochs=10,
                batch_size=128, loss_limit=0.015,
                learning_rate=1e-3):
       """
       Optimize the Neural Network by sampling states and Q-values
       from the replay-memory.
       The original DeepMind paper performed one optimization iteration
       after processing each new state of the game-environment. This is
       an un-natural way of doing optimization of Neural Networks.
       So instead we perform a full optimization run every time the
       Replay Memory is full (or it is filled to the desired fraction).
       This also gives more efficient use of a GPU for the optimization.
       The problem is that this may over-fit the Neural Network to whatever
       is in the replay-memory. So we use several tricks to try and adapt
       the number of optimization iterations.
       :param min_epochs:
           Minimum number of optimization epochs. One epoch corresponds
           to the replay-memory being used once. However, as the batches
           are sampled randomly and biased somewhat, we may not use the
           whole replay-memory. This number is just a convenient measure.
       :param max_epochs:
           Maximum number of optimization epochs.
       :param batch_size:
           Size of each random batch sampled from the replay-memory.
       :param loss_limit:
           Optimization continues until the average loss-value of the
           last 100 batches is below this value (or max_epochs is reached).
       :param learning_rate:
           Learning-rate to use for the optimizer.
       """
       print("Optimizing Neural Network to better estimate Q-values ...")
       print("\tLearning-rate: {0:.1e}".format(learning_rate))
       print("\tLoss-limit: {0:.3f}".format(loss_limit))
       print("\tMax epochs: {0:.1f}".format(max_epochs))
       # Prepare the probability distribution for sampling the replay-memory.
       self.replay_memory.prepare_sampling_prob(batch_size=batch_size)
       # Number of optimization iterations corresponding to one epoch.
       iterations_per_epoch = self.replay_memory.num_used / batch_size
       # Minimum number of iterations to perform.
       min_iterations = int(iterations_per_epoch * min_epochs)
       # Maximum number of iterations to perform.
       max_iterations = int(iterations_per_epoch * max_epochs)
       # Buffer for storing the loss-values of the most recent batches.
       loss_history = np.zeros(100, dtype=float)
       clone_model_update_frequency = int(max_iterations / 10)
       for i in range(max_iterations):

           if i % clone_model_update_frequency == 0:
               self.clone_network()

           # Randomly sample a batch of states and target Q-values
           # from the replay-memory. These are the Q-values that we
           # want the Neural Network to be able to estimate.
           state_batch, q_values_batch, next_states_batch, rewards_batch, end_episode_batch = self.replay_memory.random_batch()
           # Create a feed-dict for inputting the data to the TensorFlow graph.
           # Note that the learning-rate is also in this feed-dict.

           target_q = self.get_cloned_q_values(next_states_batch)
           #target_q_max = np.argmax(target_q, axis=1)
           #target_q = np.array(rewards_batch) + ((1 - np.array(end_episode_batch)) * (self.replay_memory.discount_factor * np.array(target_q)))
           #target_q = rewards_batch +  (self.replay_memory.discount_factor * target_q)

           for i in range(len(end_episode_batch)):
                if end_episode_batch[i]:
                   shape = np.shape(target_q[i])
                   target_q[i] = np.full(shape, rewards_batch[i])
                else:
                    target_q[i] = target_q[i] * self.replay_memory.discount_factor
                    target_q[i] = target_q[i] + rewards_batch[i]


           feed_dict = {self.x: state_batch,
                        self.target_q_values: target_q,
                        self.learning_rate: learning_rate}
           # Perform one optimization step and get the loss-value.
           loss_val, _ = self.sess.run([self.loss, self.train_step],
                                          feed_dict=feed_dict)
           # Shift the loss-history and assign the new value.
           # This causes the loss-history to only hold the most recent values.
           loss_history = np.roll(loss_history, 1)
           loss_history[0] = loss_val
           # Calculate the average loss for the previous batches.
           loss_mean = np.mean(loss_history)
           # Print status.
           pct_epoch = i / iterations_per_epoch
           msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}"
           msg = msg.format(i, pct_epoch, loss_val, loss_mean)
           print_progress(msg)

           # Stop the optimization if we have performed the required number
           # of iterations and the loss-value is sufficiently low.
           if i > min_iterations and loss_mean < loss_limit:
               break
       # Print newline.
       print()
       
       
   

class Agent:
    
    
    def __init__(self, env_name, training, target_network_update_frequency, render=False):
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.target_network_update_frequency = target_network_update_frequency
        self.training = training
        self.render = render
        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e6,
                                            num_actions=self.num_actions,
                                            epsilon_testing=0.01)
        self.log_q_values = LogQValues()
        self.log_reward = LogReward()
        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=1e-3,
                                                             end_value=1e-5,
                                                             num_iterations=5e6)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=5e6)

            # The maximum number of epochs to perform during optimization.
            # This is increased from 5 to 10 epochs, because it was found for
            # the Breakout-game that too many epochs could be harmful early
            # in the training, as it might cause over-fitting.
            # Later in the training we would occasionally get rare events
            # and would therefore have to optimize for more iterations
            # because the learning-rate had been decreased.
            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=5e6)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=5e6)
        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM. The image-frames from the
            # game-environment are resized to 105 x 80 pixels gray-scale,
            # and each state has 2 channels (one for the recent image-frame
            # of the game-environment, and one for the motion-trace).
            # Each pixel is 1 byte, so this replay-memory needs more than
            # 3 GB RAM (105 x 80 x 2 x 200000 bytes).

            self.replay_memory = ReplayMemory(size=200000, num_actions=self.num_actions)
        else:
            self.replay_memory = None

        
        self.model = NeuralNetwork(self.replay_memory, self.num_actions)
        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []
    
    def reset_episode_rewards(self):
        #Reset the log of episode-rewards.
        self.episode_rewards = []

    def get_action_name(self, action):
        #Return the name of an action.
        return self.action_names[action]

    def get_lives(self):
        #Get the number of lives the agent has in the game-environment.
        return self.env.unwrapped.ale.lives()
    
    def run(self, num_episodes=None):
        global last_4_frames
        global state_4_frames
        #To reset for first run
        end_episode = True
        count_episodes = self.model.get_count_episodes()
        count_states = self.model.get_count_states()
        reward_episode = 0
        
        num_episodes += count_episodes
        
        while count_episodes <= num_episodes:
            
            if count_states % self.target_network_update_frequency == 0:
                self.model.clone_network()
            
            if end_episode:
                last_4_frames = []
                state_4_frames = []
                img = self.env.reset()
                img = process_image(img)
                last_4_frames.append(img)
                reward_episode = 0
                
                count_episodes = self.model.increase_count_episodes()
                num_lives = self.get_lives()
                
                # Populate state with first 4 frames
                for i in range(3):
                    img, r, d, _ = self.env.step(self.env.action_space.sample())
                    img = process_image(img)
                    last_4_frames.append(img)
                
                last_4_frames = list(reversed(last_4_frames))
            
            # state = convert_shape_last_4_frames(last_4_frames)
            state_4_frames = convert_shape_last_4_frames(last_4_frames)
            #print("Last 4 frames: " + str(last_4_frames[0][50][50]) + " State 4 Frames: " + str(state_4_frames[50][50][0]))

            q_values = self.model.get_q_values([state_4_frames])
            
            action, epsilon = self.epsilon_greedy.get_action(q_values=q_values, iteration=count_states, training=self.training)
            
            img, reward, end_episode, info = self.env.step(action)
            img = process_image(img)
            
            # Replace oldest frame with new image
            new_state = last_4_frames[len(last_4_frames) - 1:] + last_4_frames[0:len(last_4_frames) - 1]
            new_state[0] = img
            new_state_processed = convert_shape_last_4_frames(new_state)
            
            reward_episode += reward
            
            # Was a life lost?
            num_lives_new = self.get_lives()
            end_life = (num_lives_new < num_lives)
            num_lives = num_lives_new
            
            count_states = self.model.increase_count_states()
            
            if not self.training and self.render:
                self.env.render()
            
            if self.training:
                self.replay_memory.add(state_4_frames, q_values=q_values, action=action,
                                       reward=reward,end_life=end_life,end_episode=end_episode, 
                                       next_state=new_state_processed)
                
                use_fraction = self.replay_fraction.get_value(iteration=count_states)
                
                if self.replay_memory.is_full() or self.replay_memory.used_fraction() > use_fraction:
                    # Train neural network on memory
                    self.replay_memory.update_all_q_values()
                    self.log_q_values.write(count_episodes=count_episodes, count_states=count_states, q_values=self.replay_memory.q_values)
                    learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                    loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                    max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                    # Perform an optimization run on the Neural Network so as to
                    # improve the estimates for the Q-values.
                    # This will sample random batches from the replay-memory.
                    self.model.train(learning_rate=learning_rate, loss_limit=loss_limit, max_epochs=max_epochs)
                    self.model.save_checkpoint(count_states)
                    self.replay_memory.reset()

            if end_episode:
                # Add the episode's reward to a list for calculating statistics.
                self.episode_rewards.append(reward_episode)

            if len(self.episode_rewards) == 0:
                # The list of rewards is empty.
                reward_mean = 0.0
            else:
                reward_mean = np.mean(self.episode_rewards[-30:])
        
            if self.training and end_episode:
                self.log_reward.write(count_episodes=count_episodes,count_states=count_states,reward_episode=reward_episode,reward_mean=reward_mean)
                # Print reward to screen.
                msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
                print(msg.format(count_episodes, count_states, epsilon,
                                         reward_episode, reward_mean))
            elif not self.training and (reward != 0.0 or end_life or end_episode):
                # Print Q-values and reward to screen.
                msg = "{0:4}:{1}\tQ-min: {2:5.3f}\tQ-max: {3:5.3f}\tLives: {4}\tReward: {5:.1f}\tEpisode Mean: {6:.1f}"
                print(msg.format(count_episodes, count_states, np.min(q_values),
                np.max(q_values), num_lives, reward_episode, reward_mean))
            last_4_frames = new_state


if __name__ == '__main__':
    
    env_name = "Breakout-v0"
    update_paths(env_name=env_name)
    agent = Agent(env_name, training=True, render=False, target_network_update_frequency=2000)
    
    agent.run(300)
    
    print("--------------------")
    print("--------------------")
    
    agent.reset_episode_rewards()
    
    agent.training = False
    agent.render = True
    
    agent.run(1)
    
    rewards = agent.episode_rewards
    print("Rewards for {0} episodes:".format(len(rewards)))
    print("- Min:   ", np.min(rewards))
    print("- Mean:  ", np.mean(rewards))
    print("- Max:   ", np.max(rewards))
    print("- Stdev: ", np.std(rewards))

    