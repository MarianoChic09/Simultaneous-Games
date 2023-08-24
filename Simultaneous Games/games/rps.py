import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
from pettingzoo.utils.env import ActionDict, ObsDict
from base.game import SimultaneousGame

class RPS(SimultaneousGame):

    def __init__(self):
        self._R = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions
        self._moves = ['R', 'P', 'S']
        self._num_actions = 3
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        # rewards
        (a0, a1) = tuple(map(lambda agent: actions[agent], self.agents))
        r = self._R[a0][a1]
        self.rewards[self.agents[0]] = r
        self.rewards[self.agents[1]] = -r

        # observations
        self.observations = dict(map(lambda agent: (agent, actions), self.agents))

        # etcetera
        self.terminations = dict(map(lambda agent: (agent, True), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsDict, dict[str, dict]]:
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, None

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._moves[self.agent_name_mapping[agent]], self.rewards[agent])
