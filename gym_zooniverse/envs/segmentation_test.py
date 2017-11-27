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
        'video.frames_per_second' : 10
    }
    
    def __init__(self):
        self.range = 50 # number of pixels in the x-direction
        self.height = 10 # number of pixels on the y-direction
        self.scale = 10 # scale for rendering.
        
        self.action_space = spaces.Discrete(2)
        #self.action_space.n = 2
        
        self.observation_space = spaces.Box(low=0, high=1, \
          shape=(self.height, self.range, 3))

        self.guess_count = 0
        self.guess_max = 2000
        
        self.gamma = 0.1 # reward scaling factor
        
        self.viewer = None

        self.optimal_solution = 0.
        
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action == 0: # move right
          if self.state[0]+1 == self.range:
            self.state = (self.range-1, self.range-1)
          else:
            self.state = (self.state[0]+1, self.state[1]+1)
        elif action == 1: # move left
          if self.state[0]-1 < 0:
            self.state = (0, 0)
          else:
            self.state = (self.state[0]-1, self.state[1]-1)
        
        (x0, x1) = self.state
        y0, y1 = 1, self.height
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        zi = self.image[y.astype(np.int)-1, x.astype(np.int)]
        pixel_sum = np.sum(zi)
        if pixel_sum == 0:
          reward = 1 / (self.gamma*self.guess_count)
        else:
          #reward = 1/pixel_sum
          reward = 0
        
        obs = np.zeros((self.height,self.range))[:,:,np.newaxis]
        obs[:,x0] += 1
        self.observation = np.concatenate((self.image[:,:,np.newaxis], self.image[:,:,np.newaxis], obs),axis=2)
        
        self.guess_count += 1
        done = self.guess_count >= self.guess_max or pixel_sum == self.optimal_solution

        return self.observation, reward, done, \
            {"optimal_solution": self.optimal_solution, "guesses": self.guess_count}
    
    def _reset(self):
        try:
          self.viewer.close()
        except AttributeError:
          pass
        self.viewer = None
        self.guess_count = 0
        
        # initialise the state of the environment
        self.image = np.ones((self.height,self.range))
        self.gap = np.random.randint(0,self.range)
        self.image[:,self.gap:self.gap+1] *= 0
        
        state = np.random.randint(0,self.range)
        self.state = (state,state)
        
        obs = np.zeros((self.height,self.range))[:,:,np.newaxis]
        obs[:,self.state[0]] = .5
        self.observation = np.concatenate((self.image[:,:,np.newaxis], self.image[:,:,np.newaxis], obs),axis=2)
        #self.viewer = None
        return self.observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
    
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.scale*self.range, self.scale*self.height)
            l, r, t, b = 0, self.scale*self.gap, self.scale*self.height, 0
            left_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            left_blob.set_color(0,0,0)
            self.viewer.add_geom(left_blob)
            
            l, r, t, b = self.scale*self.gap, self.scale*(self.gap+1), self.scale*self.height, 0
            gap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            gap.set_color(255,255,255)
            self.viewer.add_geom(gap)
            
            l, r, t, b = self.scale*(self.gap+1), self.scale*self.range, self.scale*self.height, 0
            right_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            right_blob.set_color(0,0,0)
            self.viewer.add_geom(right_blob)
            
            #segmentation = rendering.Line((0, 0), (0, 100))
            segmentation = rendering.Line((0, 0), (0, self.scale*self.range))
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
        self.segtrans.set_translation(self.scale*translationx, self.scale*translationy)
        rotation = - np.tan(((self.state[1]-self.state[0]))/(self.height-1))
        #print(np.degrees(rotation))
        self.segtrans.set_rotation(rotation)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def main():
  import matplotlib.pyplot as plt
  
  test = SegmentationTestEnv()
  observation, reward, done, meta = test.step(0)
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(0)
  print(test.state, reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(1)
  print(test.state, reward)
  test.render()
  input()
  test.render(close=True)

if __name__ == '__main__':
  main()

