from gym.envs.registration import register

register(
    id='SegmentationTestEnv-v0',
    entry_point='gym_zooniverse.envs:SegmentationTestEnv',
)

register(
    id='TextSegmentationTestEnv-v0',
    entry_point='gym_zooniverse.envs:TextSegmentationTestEnv',
)
