from gym.envs.registration import register

register(
    id='PathFollower-v0',
    entry_point='gym_path.envs:PathEnv',
)
register(
    id='PathFollower-extension-v0',
    entry_point='gym_path.envs:PathExtensionEnv',
)
register(
    id='PathFollower-DifferentPaths-v0',
    entry_point='gym_path.envs:PathEnvDifferentPaths',
)
