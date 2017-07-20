import gym
import tensorflow as tf
import numpy as np
import scipy.misc
from collections import deque
import random
import os
import argparse

# Helper functions to process image

# Pass in [210, 160, 3] rgb image and returns [210, 160] gray scale image
def rgb2_grayScale(rgb):
    gray = np.mean(rgb, -1)
    return gray


# Convert Image (given as [210, 160, 3]) into [84,84]
def process_image(image):
    # First Make Image Grayscale: [210, 160]
    gray = rgb2_grayScale(image)
    # Reshape image
    img = scipy.misc.imresize(gray, size=[105, 80], interp='bicubic')

    return img
# --------------------------------------------------------------------------

# A class to help manage the current state of the game
# The state consists of the last 4 frames in the game. You can't use only the last frame
# Since you wouldn't be able to tell the direction of the bal for example.
class StateController:

    def __init__(self, state_width, state_height, history):
        self.dimension = (state_width, state_height, history)
        self.state = np.zeros(self.dimension, dtype=np.float32)

    def add(self, state):
        self.state[:, :, :-1] = self.state[:, :, 1:]
        self.state[:, :, -1] = state

    def get_input(self):
        x = np.reshape(self.state, (1,) + self.dimension)
        return x

    def get_state(self):
        return self.state

    def reset(self):
        self.state.fill(0)


class ConvNet:

    def __init__(self, input_width, input_height, history_length, num_actions, trainable):

        # Values for the conv net
        self.shape = [None, input_width, input_height, history_length]
        self.x = tf.placeholder(tf.float32, shape=self.shape)
        self.out_dims = num_actions
        self.filters = [32, 64, 64]
        self.num_conv_layers = 3
        self.filter_size = [8, 4, 4]
        self.filter_stride = [4, 2, 1]
        self.fc_size = [512]
        self.fc_layers = 1
        self.trainable = trainable

        # Keep track of weights of the network so it can be cloned
        self.weights = {}

        # Prediction
        self.y = self.infer(self.x)

    # Helper functions to make the Neural Network
    def create_weight(self, shape):
        init = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init, name="weight")

    def create_bias(self, shape):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init, name="bias")

    def create_conv(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def infer(self, input):
        self.layers = [input]

        # Create The Deep Neural Network

        for layer in range(self.num_conv_layers):
            with tf.variable_scope("conv" + str(layer)) as scope:
                if layer == 0:
                    in_channels = self.shape[-1]
                    out_channels = self.filters[layer]
                else:
                    in_channels = self.filters[layer - 1]
                    out_channels = self.filters[layer]

                shape = [self.filter_size[layer], self.filter_size[layer], in_channels, out_channels]

                # Create Conv Layer
                W = self.create_weight(shape)
                conv = self.create_conv(self.layers[-1], W, self.filter_stride[layer])
                b = self.create_bias([out_channels])

                # Remeber weights
                self.weights[W.name] = W
                self.weights[b.name] = b

                bias = tf.nn.bias_add(conv, b)
                # Apply Relu
                conv = tf.nn.relu(bias, name=scope.name)
                self.layers.append(conv)


        # Flatten the conv layer
        last_conv = self.layers[-1]
        dim = 1
        for d in last_conv.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(last_conv, [-1, dim], name='flat')
        self.layers.append(reshape)

        # Create Fully Connected Layers

        for layer in range(self.fc_layers):
            with tf.variable_scope("hidden" + str(layer)) as scope:
                if layer == 0:
                    in_size = dim
                else:
                    in_size = self.fc_size[layer - 1]

                out_size = self.fc_size[layer]
                shape = [in_size, out_size]
                W = self.create_weight(shape)
                b = self.create_bias([out_size])

                self.weights[W.name] = W
                self.weights[b.name] = b

                dense = tf.nn.relu_layer(self.layers[-1], W, b, name=scope.name)
                self.layers.append(dense)


        # Create output layer
        with tf.variable_scope("output") as scope:
            in_size = self.fc_size[self.fc_layers - 1]
            out_size = self.out_dims
            shape = [in_size, out_size]

            W = self.create_weight(shape)
            b = self.create_bias([out_size])
            self.weights[W.name] = W
            self.weights[b.name] = b

            output = tf.nn.bias_add(tf.matmul(self.layers[-1], W), b)
            self.layers.append(output)


        return self.layers[-1]



class Environment:

    def __init__(self, env_name, render, height, width):
        self.env = gym.make(env_name)
        self.obs = None
        self.render = render
        self.terminal = False
        self.dimension = (height, width)

    def actions(self):
        return self.env.action_space.n

    def act(self, action):
        # Display the game if needed
        if self.render:
            self.env.render()

        # Execute action and observe result
        self.obs, reward, self.terminal, info = self.env.step(action)

        # If lost, reset the game
        if self.terminal:
            self.env.reset()

        return reward

    def restart(self):
        self.obs = self.env.reset()
        self.terminal = False

    def get_screen(self):
        # Process the image to make it grayscale and 84x84
        img = process_image(self.obs)
        return img

    def is_terminal(self):
        return self.terminal



class Agent:

    def __init__(self, env, episodes, steps, train_steps, save_weights, history_length, discount, init_eps, final_eps, final_eps_frame, replay_start_size, random_starts, batch_size, ckpt_dir, game, lr, lr_anneal, width, height, memory_size, decay_rate, update_freq):
        # Reference to environment class
        self.env = env
        # Number of actions in game
        self.num_actions = self.env.actions()
        # Number of games to play after training
        self.episodes = episodes
        # Copy the main network to the target network after this many steps
        # This corresponds to the 'C' parameter in the Deep Mind paper
        self.steps = steps
        # Number of training steps
        self.train_steps = train_steps
        # Save the weights after this many steps
        self.save_weights = save_weights
        # The number of previous frames that are part of the current state
        self.history_length = history_length
        # Discount factor used for training
        self.discount = discount
        # Number of actions selected between training runs
        self.update_freq = update_freq
        # Epsilon or exploration values
        self.eps = init_eps
        self.eps_delta = (final_eps - init_eps) / final_eps_frame
        self.final_eps = final_eps
        self.eps_endt = final_eps_frame
        # Amount of memory to initially populate the replay memory
        self.replay_start_size = replay_start_size
        # Number of random games to run at first
        self.random_starts = random_starts
        # Batch size for training
        self.batch_size = batch_size
        # Checkpoint path
        self.ckpt_file = ckpt_dir + "/" + game + "/checkpoint"
        self.ckpt_restore = ckpt_dir + "/" + game
        # Epsilon for testing
        self.epsilon_test = 0.01

        if not os.path.exists(ckpt_dir + "/" + game):
            os.makedirs(ckpt_dir + "/" + game)

        self.count_states = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="count_states")
        self.increase_count_states = tf.assign(self.count_states, self.count_states + 1)

        self.global_step = tf.Variable(0, trainable=False)

        # Learning rate
        if lr_anneal:
            self.lr = tf.train.exponential_decay(lr, self.global_step, lr_anneal, 0.96, staircase=True)
        else:
            self.lr = lr

        # Set up replay memory and state controller to get state of game
        self.state_controller = StateController(width, height, self.history_length)
        self.memory = ReplayMemory(memory_size, self.batch_size)

        # Build conv network and its clone to predict the target
        with tf.variable_scope("train") as self.train_scope:
            self.train_net = ConvNet(width, height, self.history_length, self.num_actions, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = ConvNet(width, height, self.history_length, self.num_actions, trainable=False)

        # Optimizer to train the network
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, decay_rate, 0.0, self.eps)

        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        self.q_target = tf.placeholder(tf.float32, shape=[None])
        self.q_train = tf.reduce_max(tf.multiply(self.train_net.y, self.actions), reduction_indices=1)

        # Get loss
        self.diff = tf.subtract(self.q_target, self.q_train)
        self.diff_square = tf.multiply(tf.constant(0.5), tf.square(self.diff))
        self.loss = tf.reduce_mean(self.diff_square)

        # Trainer
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def random_restart(self):
        self.env.restart()

        for _ in range(self.random_starts):
            action = random.randrange(self.num_actions)
            # Take random action and get result
            reward = self.env.act(action)
            state = self.env.get_screen()
            terminal = self.env.is_terminal()
            self.state_controller.add(state)

            if terminal:
                self.env.restart()

    def train_eps(self, train_step):
        amount = train_step / self.eps_endt
        if amount > 1:
            return self.final_eps
        else:
            diff = self.eps - self.final_eps
            return self.eps - (diff * amount)

    def observe(self, exploration_rate):
        # Decide best action based on epsilon greedy
        if random.random() < exploration_rate:
            a = random.randrange(self.num_actions)
        else:
            x = self.state_controller.get_input()
            action_values = self.train_net.y.eval(feed_dict={self.train_net.x: x})
            a = np.argmax(action_values)

        # Do the action and get the result
        state = self.state_controller.get_state()
        action = np.zeros(self.num_actions)
        action[a] = 1.0
        reward = self.env.act(a)
        screen = self.env.get_screen()
        self.state_controller.add(screen)
        next_state = self.state_controller.get_state()
        terminal = self.env.is_terminal()
        reward = np.clip(reward, -1.0, 1.0)

        # Add this to the replay memory so it can be used for training
        self.memory.add(state, action, reward, next_state, terminal)

        return state, action, reward, next_state, terminal


    def do_minibatch(self, sess, successes, failures):
        # Get batch from replay memory
        batch = self.memory.get_sample()

        # Process batch
        state = np.array([batch[i][0] for i in range(self.batch_size)]).astype(np.float32)
        actions = np.array([batch[i][1] for i in range(self.batch_size)]).astype(np.float32)
        rewards = np.array([batch[i][2] for i in range(self.batch_size)]).astype(np.float32)
        successes += np.sum(rewards == 1)
        next_state = np.array([batch[i][3] for i in range(self.batch_size)]).astype(np.float32)
        terminals = np.array([batch[i][4] for i in range(self.batch_size)]).astype(np.float32)
        failures += np.sum(terminals==1)

        # Target netowrk (cloned network) used to predict q value for next state
        q_target = self.target_net.y.eval(feed_dict={self.target_net.x: next_state})
        q_target_max = np.argmax(q_target, axis=1)
        # Up date target based on reward and discount factor
        q_target = rewards + ((1 - terminals) * (self.discount * q_target_max))

        # Run the train step and the loss
        result, loss = sess.run([self.train_step, self.loss], feed_dict={self.q_target: q_target, self.train_net.x: state, self.actions: actions})

        return successes, failures, loss

    def play(self):
        self.random_restart()
        self.env.restart()
        # Play certain number of episodes
        rewards = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            agent.restore(self.saver, sess)
            for i in range(self.episodes):
                terminal = False
                total_reward = 0
                while not terminal:
                    state, action, reward, obs, terminal = self.observe(self.epsilon_test)
                    total_reward += reward
                rewards.append(total_reward)

        print()
        print("--------------------------------------------------")
        print("Average reward for " + str(self.episodes) + " games:")
        print(np.mean(rewards))

    def copy_weights(self, sess):
        # Clone the prediction network to form the target network (copy the weights)
        for key in self.train_net.weights.keys():
            t_key = "target/" + key.split("/", 1)[1]
            sess.run(self.target_net.weights[t_key].assign(self.train_net.weights[key]))

    def save(self, saver, sess, step):
        saver.save(sess, self.ckpt_file, global_step=step)


    def restore(self, saver, sess):

        try:
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.ckpt_restore)
            saver.restore(sess, last_chk_path)
            print("Restored checkpoint")
        except:
            print("Failed to restore checkpoint")


class Trainer:

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.saver = tf.train.Saver()

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            agent.restore(self.saver, sess)
            agent.random_restart()
            successes = 0
            failures = 0
            total_loss = 0

            print(str(self.agent.replay_start_size) + " random moves to initially populate the replay memory")

            for i in range(self.agent.replay_start_size):
                # Give it a 100% chance of doing a random action
                state, action, reward, next_state, terminal = self.agent.observe(1)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1

                if (i+1) % 10000 == 0:
                    print ("\nmemory size: %d" % len(self.agent.memory), "\nSuccesses: ", successes, "\nFailures: ", failures)


            sample_success = 0
            sample_failure = 0
            print("\n Starting Training")

            reward_episode = 0
            reward_history = []
            count_states = int(sess.run(self.agent.count_states))
            self.agent.train_steps += count_states

            for i in range(count_states, self.agent.train_steps):
                eps = self.agent.train_eps(i)

                state, action, reward, next_state, terminal = self.agent.observe(eps)
                reward_episode += reward

                if terminal:
                    reward_history.append(reward_episode)
                    reward_episode = 0

                if len(self.agent.memory) > self.agent.batch_size and (i + 1) % self.agent.update_freq == 0:
                    sample_success, sample_failure, loss = self.agent.do_minibatch(sess, sample_success, sample_failure)
                    total_loss += loss

                if (i + 1) % self.agent.steps == 0:
                    self.agent.copy_weights(sess)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1

                if (i + 1) % self.agent.save_weights == 0:
                    self.agent.save(self.saver, sess, i + 1)

                if (i + 1) % (self.agent.batch_size * 3) == 0:
                    avg_loss = total_loss / (self.agent.batch_size * 3)
                    if len(reward_history) == 0:
                        average_reward = 0.0
                    else:
                        average_reward = np.mean(reward_history[-30:])
                    print("\nTraining step: ", i + 1,
                    "\nmemory size: ", len(self.agent.memory),
                    "\nLearning rate: ", sess.run(self.agent.lr),
                    "\nEpsilon:", eps,
                    #"\nSuccesses: ", successes,
                    #"\nFailures: ", failures,
                    "\nSample successes: ", sample_success,
                    "\nSample failures: ", sample_failure,
                    "\nAverage batch loss: ", avg_loss,
                    "\nAverage Reward:", average_reward)

                    total_loss = 0

                sess.run(self.agent.increase_count_states)






class ReplayMemory:

    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, terminal):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()

        self.memory.append([state, action, reward, next_state, terminal])

    def get_sample(self):
        return random.sample(list(self.memory), self.batch_size)
        #return np.random.choice(self.memory, self.batch_size, replace=False)
        #sample = []
        #for i in range(self.batch_size):
         #   idx = random.randrange(0, len(self.memory))


    def reset(self):
        self.memory.clear()


if __name__ == "__main__":

    ckpt_dir = "ckpt_DQN"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    parser = argparse.ArgumentParser(description="Reinforcement learning with Tensorflow")

    parser.add_argument("--env", required=False, default='Breakout-v0',
                        help="name of the game-environment in OpenAI Gym")

    parser.add_argument("--render", required=False, default=False, help="render the test games")

    parser.add_argument("--train_steps", required=False, default=750000, type=int, help="number of train steps to do")

    args = parser.parse_args()

    env_name = args.env
    env = Environment(env_name, False, 105, 80)
    # Create the agent              vv  This number is the number of training states to go over
    agent = Agent(env, 100, 10000, args.train_steps, 10000, 4, 0.99, 1, 0.1, 1000000, 40000, 30, 32, ckpt_dir, env_name, 0.00025,
                  20000, 105, 80, 100000, 0.95, 4)

    Trainer(agent).run()
    render = True
    if args.render == "False":
        render = False
    env.render = render

    print()
    print("Done training. Playing " + str(agent.episodes) + " games to test...")

    agent.play()





