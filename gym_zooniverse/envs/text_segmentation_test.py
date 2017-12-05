import gym
from gym import spaces
from gym.utils import seeding

import os
import random
import numpy as np
from scipy import misc
from skimage.filters import threshold_otsu
from skimage.transform import resize

class TextSegmentationTestEnv(gym.Env):
    """Text Segmentation Test
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    def __init__(self, assets_path='/Users/dwright/dev/gym-zooniverse/gym_zooniverse/envs'):

        self.height = 100
        self.width  = 500
        self.scale = 10
        self.fps=10 # added for rl-teacher
        
        self.action_space = spaces.Discrete(2)

        self.guess_count = 0
        self.guess_max = 2000
        self._max_episode_steps = 2000 # added for rl-teacher
        
        self.gamma = 0.1 # reward scaling factor
        
        self.viewer = None
        
        self.assets = []
        for root, dirs, files in os.walk(assets_path):
          for file in files:
            if file.endswith('.png'):
              self.assets.append(os.path.join(root, file))

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action == 0: # move right
          if self.state[0] + 1*self.scale == self.width:
            pass
          else:
            self.state = (self.state[0]+1*self.scale, self.state[1]+1*self.scale)
        elif action == 1: # move left
          if self.state[0] - 1*self.scale < 0:
            self.state = (0, 0)
          else:
            self.state = (self.state[0]-1*self.scale, self.state[1]-1*self.scale)
        
        (x0, x1) = self.state
        y0, y1 = 1, self.height
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        zi = self.image[y.astype(np.int)-1, x.astype(np.int)]
        pixel_sum = np.sum(zi)
        if pixel_sum == 0:
          try:
            reward = 1 / (self.gamma*self.guess_count)
          except ZeroDivisionError:
            reward = 1
        else:
          reward = 0
        
        self.observation = self.image.astype('float').copy()
        self.observation[:, x0] = .5
        
        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"guesses": self.guess_count}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
            
        line = np.zeros(self.observation.shape)
        line[:,self.state[0]] += 255
        img = np.concatenate((self.observation[:,:,np.newaxis]*255, \
                              line[:,:,np.newaxis],\
                              np.zeros(self.observation.shape)[:,:,np.newaxis]), axis=2)
        if mode == 'rgb_array':
            #img = self.observation
            return img.astype('uint8')
            #return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img.astype('uint8'))
    
    def _reset(self):
        try:
          self.viewer.close()
        except AttributeError:
          pass
        self.viewer = None
        self.guess_count = 0

        # initialise the state of the environment
        image = misc.imread(random.choice(self.assets))
        print(image.shape)
        image = resize(image, (self.height, self.width))
        thresh = threshold_otsu(image)
        self.image = image > thresh
        
        #self.height, self.width = self.image.shape
        
        if self.width < self.height:
          self.reset()
        
        self.observation_space = spaces.Box(low=0, high=1, \
          shape=(self.height, self.width, 3))

        state = np.random.randint(0,self.width/self.scale)*self.scale
        print(state)
        self.state = (state,state)

        self.observation = self.image.astype('float').copy()
        self.observation[:, self.state[0]] = .5
        return self.observation


def main():
  import matplotlib.pyplot as plt
  
  env = TextSegmentationTestEnv()
  print(hasattr(env,'guess_max'))
  print(env.guess_max)
  plt.imshow(env.observation, cmap='gray')
  plt.show()

  env.step(1)
  env.render()
  input()
  env.render(close=True)

if __name__ == '__main__':
  main()
