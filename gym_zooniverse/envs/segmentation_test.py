import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.ndimage

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
        'video.frames_per_second' : 50
    }
    
    def __init__(self):
        self.range = 50 # number of pixels in the x-direction
        self.height = 10 # number of pixels on the y-direction
        self.bounds = 50 # action bounds space
        self.action_space = spaces.Box(low=np.array([0, 0]), \
                                       high=np.array([self.range, self.range]))
        self.observation_space = spaces.Discrete(4)

        self.optimal_solution = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self._seed()
        self._reset()
        
        self.image = np.ones((self.height,self.range))
        self.image[:,24:26] *= 0

        self.viewer = None
        

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.action = action
        x0, x1 = action[0], action[1]
        y0, y1 = 0, self.range
        x, y = np.linspace(x0, x1, 50), np.linspace(y0, y1, 50)
        zi = scipy.ndimage.map_coordinates(self.image, np.vstack((x,y)))
        reward = np.sqrt(self.height*self.height \
               + self.range*self.range) \
               - np.sum(zi)
        
        if reward < self.optimal_solution:
            self.observation = 1

        elif reward == self.optimal_solution:
            self.observation = 2

        elif reward > self.optimal_solution:
            self.observation = 3

        self.guess_count += 1
        done = self.guess_count >= self.guess_max or reward == self.optimal_solution

        return self.observation, reward, done, \
            {"optimal_solution": self.optimal_solution, "guesses": self.guess_count}
    
    def _reset(self):
        self.guess_count = 0
        self.observation = 0
        return self.observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
    
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(10*self.range, 10*self.height)
            l, r, t, b = 0, 240, 100, 0
            left_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            left_blob.set_color(0,0,0)
            self.viewer.add_geom(left_blob)
            
            l, r, t, b = 240, 260, 100, 0
            gap = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            gap.set_color(255,255,255)
            self.viewer.add_geom(gap)
            
            l, r, t, b = 260, 500, 100, 0
            right_blob = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            right_blob.set_color(0,0,0)
            self.viewer.add_geom(right_blob)
            
            segmentation = rendering.Line((10*self.action[0], 0), (10*self.action[1], 100))
            segmentation.set_color(255,0,0)
            segmentation.add_attr(rendering.LineWidth(10))
            self.viewer.add_geom(segmentation)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def main():
  import matplotlib.pyplot as plt
  
  test = SegmentationTest()
  observation, reward, done, meta = test.step(np.array([25,25]))
  print(reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([0,49]))
  print(reward)
  test.render()
  input()
  observation, reward, done, meta = test.step(np.array([2,2]))
  print(reward)
  test.render()
  test.render(close=True)

if __name__ == '__main__':
  main()
