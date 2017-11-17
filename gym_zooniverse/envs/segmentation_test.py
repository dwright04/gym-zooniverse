import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.ndimage
from scipy.misc import toimage
import cv2

class SegmentationTestEnv(gym.Env):
    """Segementation Test
    The goal of this environment is to test ideas for text segmentation.
    
    As a step towards that end this environment will be an image with a blob
    at either end representing characters.  The agents goal is to identify gap 
    between the blobs as the point at which to segment.
    
    After each step the agent recieves an observation of:
    0 - No guess yet submitted (only after reset)
    1 - Guess is better than previous one
    2 - Optimal segmentation has been achieved
    3 - Guess is worse than previous one

    At each step the agent acts by choosing two pixels,  The first pixel is 
    along the base of the image and the second along the top of the image.  A
    line drawn between these pixels acts as the line along which to segment.
    
    The reward will be the sum of the pixel counts lying along the line between
    the two chosen pixels.  This will be minimised when the blobs are corretly
    segmented.
    
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    
    def __init__(self):
        self.range = 200 # number of pixels in the x-direction
        self.height = 10 # number of pixels on the y-direction

        #self.action_space = spaces.MultiDiscrete([ [0,self.range-1], [0,self.range-1] ])
        self.action_space = spaces.Discrete([0,self.range-1])
        self.observation_space = spaces.Box(low=0, high=255, \
          shape=(self.height, self.range, 3))

        self.guess_count = 0
        self.guess_max = 2000
        
        self.viewer = None
        
        self.image = np.ones((self.height,self.range))
        self.image[:,25:26] *= 0
        
        self.state = (2,2)
        
        obs = self.image.copy()[:,:,np.newaxis]
        obs[self.state[0],:] += 1
        self.observation = np.concatenate((obs, obs, obs),axis=2)
        
        self.optimal_solution = 0.
        
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        x0, x1 = action, action
        self.state = (x0, x1)
        y0, y1 = 1, self.height
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        zi = self.image[y.astype(np.int)-1, x.astype(np.int)]
        pixel_sum = np.sum(zi)
        reward = np.log(1/(pixel_sum+1e-9))
        
        obs = self.image.copy()[:,:,np.newaxis]
        obs[y.astype(np.int)-1, x.astype(np.int)] += 1
        self.observation = np.concatenate((obs, obs, obs),axis=2)
        
        self.guess_count += 1
        done = self.guess_count >= self.guess_max or pixel_sum == self.optimal_solution

        return self.observation, reward, done, \
            {"optimal_solution": self.optimal_solution, "guesses": self.guess_count}
    
    def _reset(self):
        self.guess_count = 0
        
        self.image = np.ones((self.height,self.range))
        self.image[:,25:26] *= 0
        
        self.state = (2,2)
        
        obs = self.image.copy()[:,:,np.newaxis]
        
        obs[self.state[0],:] += 1
        
        self.observation = np.concatenate((obs, obs, obs),axis=2)
        return self.observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
    
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.range, self.height)
            l, r, t, b = 0, 25, 10, 0
            left_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            left_blob.set_color(0,0,0)
            self.viewer.add_geom(left_blob)
            
            l, r, t, b = 25, 26, 10, 0
            gap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            gap.set_color(255,255,255)
            self.viewer.add_geom(gap)
            
            l, r, t, b = 26, 50, 10, 0
            right_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            right_blob.set_color(0,0,0)
            self.viewer.add_geom(right_blob)
            
            #segmentation = rendering.Line((0, 0), (0, 100))
            segmentation = rendering.Line((0, 0), (0, 50))
            segmentation.set_color(255,0,0)
            segmentation.add_attr(rendering.LineWidth(10))
            self.segtrans = rendering.Transform(translation=(0,0), \
                                                rotation=(0))
            segmentation.add_attr(self.segtrans)
            self.viewer.add_geom(segmentation)
            
        if self.state is None: return None

        translationx = (self.state[1] + (self.state[0]-self.state[1]))
        #print(translationx)
        translationy = 0
        self.segtrans.set_translation(translationx, translationy)
        rotation = - np.tan(((self.state[1]-self.state[0]))/(self.height-1))
        #print(np.degrees(rotation))
        self.segtrans.set_rotation(rotation)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def main():
  import matplotlib.pyplot as plt
  
  test = SegmentationTestEnv()
  observation, reward, done, meta = test.step(np.array([25,25]))
  print(observation.shape)
  plt.imshow(observation)
  plt.show()
  print(test.state, reward)
  test.render()
  input()
"""
  observation, reward, done, meta = test.step(np.array([1,49]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([25,49]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([25,29]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([29,25]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([2,2]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([250,250]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([10,490]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([250,490]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([250,290]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([290,250]))
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([20,20]))
  print(test.state, reward)
  test.render()
  input()
  test.render(close=True)
"""


if __name__ == '__main__':
  main()

