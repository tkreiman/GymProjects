import gym
import scipy.misc


def _rgb_to_grayscale(image):
    """
        Convert an RGB-image into gray-scale using a formula from Wikipedia:
        https://en.wikipedia.org/wiki/Grayscale
        """
    
    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    
    return img_gray


def _pre_process_image(image, size):
    """Pre-process a raw image from the game-environment."""
    
    # Convert image to gray-scale.
    img = _rgb_to_grayscale(image)
    
    # Resize to the desired size using SciPy for convenience.
    img = scipy.misc.imresize(img, size=size, interp='bicubic')
    
    return img


class Environment:

    def __init__(self, params):
        self.gym = gym.make(params.game)
        self.observation = None
        self.display = params.display
        self.terminal = False
        self.dims = (params.height, params.width)
        self.training = True

    def actions(self):
        return self.gym.action_space.n

    def restart(self):
        self.observation = self.gym.reset()
        self.terminal = False

    def act(self, action):
        if self.display:
            if self.training == False:
                self.gym.render()
        self.observation, reward, self.terminal, info = self.gym.step(action)
        if self.terminal:
            #if self.display:
            #    print "No more lives, restarting"
            self.gym.reset()
        return reward

    def getScreen(self):
        return _pre_process_image(self.observation, self.dims)

    def isTerminal(self):
        return self.terminal

    def get_lives(self):
        """Get the number of lives the agent has in the game-environment."""
        return self.gym.unwrapped.ale.lives()
