import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent)) # si acciones = 3 entonces [1/3, 1/3, 1/3]
        else:
          self.curr_policy = initial.copy() # sino copio la politica inicial
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent)) # inicializo los regrets en 0
        self.sum_policy = self.curr_policy.copy() # inicializo la suma de las politicas con la politica inicial
        self.learned_policy = self.curr_policy.copy() # inicializo la politica aprendida con la politica inicial
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy() # copio las acciones jugadas
        a = actions[self.agent] # a es la accion tomada por el agente
        g = self.game.clone() # clono el juego
        u = np.zeros(g.num_actions(self.agent), dtype=float) # inicializo el vector de utilidades
        for a_prime in range(g.num_actions(self.agent)): # para cada accion posible del agente
            actions[self.agent] = a_prime # la accion del agente es a_prime
            # u[a_prime] = g.utility(self.agent, actions)
            _, rewards, _, _, _ = g.step(actions) # obtengo las recompensas simulando la accion tomada por el agente
            u[a_prime] = rewards[self.agent] # guardo la utilidad del agente como la recompensa obtenida
        r = u - u[a] # calculo los regrets como la diferencia entre la utilidad de la accion y la utilidad de la accion tomada
        return r
    
    def regret_matching(self):
        positive_regrets = np.maximum(self.cum_regrets, 0) # me quedo con los regrets positivos
        sum_positive_regrets = np.sum(positive_regrets) # sumo los regrets positivos
        if sum_positive_regrets > 0: # si la suma de los regrets positivos es mayor a 0
            self.curr_policy = positive_regrets / sum_positive_regrets # la politica actual es el vector de regrets positivos dividido la suma de los regrets positivos
        else:
            self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent)) # sino por ejemplo si acciones = 3 entonces [1/3, 1/3, 1/3]
        self.sum_policy += self.curr_policy # acumula la politica actual a la suma de las politicas

    def update(self) -> None:
        actions = self.game.observe(self.agent) # obtengo las acciones jugadas
        if actions is None: 
           return
        regrets = self.regrets(actions) # calculo los regrets
        self.cum_regrets += regrets # acumulo los regrets
        self.regret_matching() # actualizo la politica actual
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter # actualizo la politica aprendida

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1)) # elijo una accion aleatoria de acuerdo a la politica actual
    
    def policy(self):
        return self.learned_policy
