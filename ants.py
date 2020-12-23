# Philippe, Lavoie, 20144246
# Imad, Mechmachi, 20163388

import numpy as np
import random as rand

class Colony:
    class Ant:
        def __init__(self, colony):
            self.colony = colony
            self.pos = rand.randrange(self.colony.n)

            self.mem = np.zeros(self.colony.n)
            self.mem[self.pos] = 1

            self.path = [self.pos]
            self.cost = 0

        def reset(self, colony):
            self.__init__(colony)

        def __str__(self):
            #TO DO
            return str(self.path) + ', cost : ' + str(self.cost)

        def __lt__(self, other):
            #TO DO
            return self.cost < other.cost

        # Returns city to be travelled to from current position
        def policy(self):
            if rand.random() < self.colony.q_0:
                # Deterministic decision
                # TODO
                u = self.pos
                '''
                Version python vanilla peu performante 
                
                idx = 0
                while self.mem[idx]:  # recherche le premier non visite
                    idx += 1
                for v in range(self.colony.n): # iteration pour trouver le max
                    if not self.mem[v]:
                        if self.colony.tau[u][idx]*self.colony.eta(u,idx)**self.colony.beta < self.colony.tau[u][v]*self.colony.eta(u,v)**self.colony.beta:
                            idx = v

                return idx
                '''
                adjMat_modif = self.colony.adjMat[u]
                adjMat_modif[u] = 9223372036854775807 # On modifie la case [u,u] pour evite la division par 0
                # on calcul ici v = arg max (1-mem[v])*tau[v]*eta(u,v)**beta
                return np.argmax((1-self.mem) * self.colony.tau[u]/adjMat_modif[u]**self.colony.beta)
            else:
                # Stochastic decision
                # TODO
            
                r = self.pos    
                # 1. Calculer les numerateurs, la somme et les probabilitees
                numerateurs = np.zeros(self.colony.n)
                for s in range(self.colony.n):
                    if not self.mem[s]:
                        numerateurs[s] = self.colony.tau[r][s]*self.colony.eta(r,s)**self.colony.beta
                prob = numerateurs/np.sum(numerateurs)

                # 2. Generer z ~ U(0,1) uniforme
                z = rand.random()

                #3. Generer S : par la methode de l'inverse de la distribution cummuler
                       
                cummul = 0
                S = -1
                while cummul < z:
                    S += 1
                    if not self.mem[S]:
                        cummul += prob[S]
                return S

        # Updates the local pheromones and position of ant
        # while keeping track of total cost and path
        def move(self):
            destination = self.policy()

            # local updating
            # TODO
            self.colony.tau[self.pos][destination] = (1-self.colony.alpha)*self.colony.tau[self.pos][destination] + self.colony.alpha*self.colony.tau_0
            self.colony.tau[destination][self.pos] = self.colony.tau[self.pos][destination]

            # Change position
            # TODO
            self.cost += self.colony.adjMat[self.pos][destination]
            self.pos = destination
            self.mem[destination] = 1
            self.path.append(destination)

            # Test si il faut revenir au noeud de depart
            if len(self.path) == self.colony.n :
                # Update : du l'arrete de retour au noeud de depart 
                depart = self.path[0]          
                self.cost += self.colony.adjMat[depart][self.pos]
                self.colony.tau[self.pos][depart] = (1-self.colony.alpha)*self.colony.tau[self.pos][depart] + self.colony.alpha*self.colony.tau_0
                self.colony.tau[depart][self.pos] = self.colony.tau[self.pos][depart]

        # Updates the pheromone levels of ALL edges that form 
        # the minimum cost loop at each iteration
        def globalUpdate(self):
            # TODO
            depart = self.path[0]
            # mise a jour global : mis a jours des arrete du chemin inlcuant le chemin de retour
            for i in range(len(self.path)):
                origine = self.path[i]
                if i == len(self.path)-1:
                    destination = depart # le retour au noeud de depart
                else:
                    destination = self.path[i+1]
                # mise a jour : update global sur l'arrete (origine, destination)
                self.colony.tau[origine][destination]=(1-self.colony.alpha)*self.colony.tau[origine][destination] + self.colony.alpha/self.cost
                self.colony.tau[destination][origine] = self.colony.tau[origine][destination] # Mise a jour symetrique t(u,r)=t(r,u)
            
            print(self)

    def __init__(self, adjMat, m=10, beta=2, alpha=0.1, q_0=0.9):
        # Parameters: 
        # m => Number of ants
        # beta => Importance of heuristic function vs pheromone trail
        # alpha => Updating propensity
        # q_0 => Probability of making a non-stochastic decision
        # tau_0 => Initial pheromone level

        self.adjMat = adjMat
        self.n = len(adjMat)

        self.tau_0 = 1 / (self.n * self.nearestNearbourHeuristic())
        self.tau = [[self.tau_0 for _ in range(self.n)] for _ in range(self.n)]
        self.ants = [self.Ant(self) for _ in range(m)]

        self.beta = beta
        self.alpha = alpha
        self.q_0 = q_0

    def __str__(self):
        # TODO
        nn_val = 1/(self.n*self.tau_0)
        txt = 'Colony \n'
        txt += 'Nombre de fourmis: ' + str(len(self.ants)) + '\n'
        txt += 'Params : alpha='+ str(self.alpha) + " beta=" + str(self.beta) + ' q0=' + str(self.q_0) + '\n'
        txt += 'Nearest Nearbour Heuristic Cost :  ' + str(nn_val)
        return txt

    # Returns the cost of the solution produced by 
    # the nearest neighbour heuristix
    def nearestNearbourHeuristic(self):
        costs = np.zeros(self.n)

        # TODO
        for depart in range(self.n): # On itere pour tout les noeuds de depart
            noeud = depart
            visite = np.zeros(self.n) # 1 si deja visite, 0 sinon
            for _ in range(self.n-1): # itere pour tracer le chemin
                visite[noeud] = 1
                idx_min = 0 # index
                while visite[idx_min] != 0: # On trouve le premier candidat
                    idx_min += 1
                for noeud_next in range(self.n): # on itere pour trouver l'arete minimum
                    if not visite[noeud_next]:
                        if self.adjMat[noeud][noeud_next] < adjMat[noeud][idx_min]:
                            idx_min = noeud_next
                
                # on met a jour le cout de la solution
                costs[depart] += self.adjMat[noeud][idx_min]
                noeud = idx_min
            # on ajoute le cout de retour
            costs[depart] += self.adjMat[noeud][depart]
    
        return min(costs)

    # Heuristic function
    # Returns inverse of smallest distance between r and u
    def eta(self, r, u):
        # TODO
        return 1/self.adjMat[r][u]

    def optimize(self, num_iter):
        for _ in range(num_iter):
            for _ in range(self.n-1):
                for ant in self.ants:
                    ant.move()

            min(self.ants).globalUpdate()


            for ant in self.ants:
                ant.reset(self)

if __name__ == "__main__":
    rand.seed(421)
    np.random.seed(70)

    #file = open('d198.csv')
    file = open('dantzig.csv')

    adjMat = np.loadtxt(file, delimiter=",")
    ant_colony = Colony(adjMat)
    print(ant_colony)
    ant_colony.optimize(1000)