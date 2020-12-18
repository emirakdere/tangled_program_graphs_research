from gym.envs.registration import register

register(
    id='delayGo-v0',
    entry_point='gym_task.envs:DelayGo'
    # entry_point='gym_task.envs:TaskEnv',
)

register(
    id='simplePerceptualDM-v0',
    entry_point='gym_task.envs:SimplePerceptualDM'
)

register(
    id='contextDependentDM-v0',
    entry_point='gym_task.envs:ContextDependentDM'
)

register(
    id='multisensoryDM-v0',
    entry_point='gym_task.envs:MultisensoryDM'
)

register(
    id='parametricWorkingMem-v0',
    entry_point='gym_task.envs:ParametricWorkingMem'
)

register(
    id='inhibitoryControl-v0',
    entry_point='gym_task.envs:InhibitoryControl'
)

register(
    id='delayedMatchToSample-v0',
    entry_point='gym_task.envs:DelayedMatchToSample'
)

register(
    id='delayedMatchToCategory-v0',
    entry_point='gym_task.envs:DelayedMatchToCategory'
)


# register(
#     id='SoccerEmptyGoal-v0',
#     entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
#     timestep_limit=1000,
#     reward_threshold=10.0,
#     nondeterministic = True,
# )
