from gym.envs.registration import register

register(
    id='PathFollower-v0',
    entry_point='gym_path.envs:PathEnv',
)
register(
    id='PathFollower-DifferentPaths-v0',
    entry_point='gym_path.envs:PathEnvDifferentPaths',
)
register(
    id='PathFollower-FeedbackLinearized-v0',
    entry_point='gym_path.envs:PathFeedbackLinearized',
)
