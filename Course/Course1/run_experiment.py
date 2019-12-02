import numpy as np
from rl_glue import RLGlue
import matplotlib.pyplot as plt

def run_experiment(env, agent, agent_info, env_info, num_experiments=1, num_steps=None, seeds=None):
    all_scores = []
    for _ in range(num_experiments):
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()

        scores = [0]
        averages = []

        for _ in range(num_steps):
            reward, state, action, is_terminal = rl_glue.rl_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))

        all_scores.append(averages)

    return all_scores