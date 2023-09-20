from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        self.count: dict[AgentID, ndarray] = {}

        if initial:
            self.count = initial
        else:
            for agent in game.agents:
                self.count[agent] = np.random.randint(game.num_actions(agent))

        self.learned_policy: dict[AgentID, ndarray] = {}
        for agent in game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])


    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        for joint_action in product(*agents_actions):
            actions = dict(map(lambda agent: (agent, joint_action[g.agent_name_mapping[agent]]), g.agents))
            g.step(actions)
            rewards[joint_action] = g.reward(self.agent)
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))

        # for action in range(self.game.num_actions(self.agent)):
        for joint_action in rewards.keys(): #product(*[range(self.game.num_actions(a)) for a in self.game.agents]):
            # if joint_action[self.agent] == action:
            # actions = dict(map(lambda agent: (agent, act[g.agent_name_mapping[agent]]), g.agents))
            action = joint_action[self.game.agent_name_mapping[self.agent]]
            prob = np.prod([self.learned_policy[other_agent][joint_action[self.game.agent_name_mapping[other_agent]]] for other_agent in self.game.agents if other_agent != self.agent])
            utility[action] += rewards[joint_action] * prob
        return utility
    
    def bestresponse(self):
        utility = self.get_utility()
        a = np.argmax(utility) 

        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        
        if actions is None:
            return
        # action_to_index = {'H': 0, 'T': 1}
        # actions = {agent: action_to_index[action] for agent, action in self.game.observe(self.agent).items()}
        # print("Converted actions:", actions)

        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    