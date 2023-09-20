import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
#from pettingzoo.utils.env import ActionDict, ObsDict
from base.game import SimultaneousGame
ActionDict = dict
ObsDict = dict
AgentID = dict

class MP(SimultaneousGame):

    def __init__(self):
        self._R = np.array([[1., -1.], [-1., 1.]])

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions
        self._moves = ['H', 'T']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # self.action_to_index = {
        #     'H': 0,
        #     'T': 1
        # }

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        # rewards
        # (a0, a1) = tuple(map(lambda agent: self.action_to_index[actions[agent]], self.agents))

        print("Actions:", actions)
        print("self._R:", self._R)

        (a0, a1) = tuple(map(lambda agent: actions[agent], self.agents))

        print("a0:", a0)
        print("a1:", a1)

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
