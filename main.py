from GeneticAlgorithm import GA
from scipy.integrate import solve_ivp
from fuzzy_create import fuzzyCreate
from myodefun import u, odefun
import numpy as np


class Problem:
    def __init__(self, lb, ub, crossovercase):
        self.lb = lb
        self.ub = ub
        self.crossovercase = crossovercase

    def obj(self, n):
        l = int(len(n)/3)
        x0 = [-1, -1, 0, 0, 0, 0, 0, 0]
        univ = [-1, 1, 0.001]
        sys = [1, 0.5, 1, 9.81, 4.4*10**-3] # M, m, l, g, J
        fis1 = fuzzyCreate(univ, 7, n[0:l])
        fis2 = fuzzyCreate(univ, 7, n[l:2*l])
        fis3 = fuzzyCreate(univ, 7, n[2*l:3*l])
        sol = solve_ivp(odefun, [0, 10], x0, t_eval=np.linspace(0, 10, 300), args=(fis1, fis2, fis3, sys))

        return -1 * (np.dot(sol.t, np.absolute(sol.y[1, :])) + np.dot(sol.t, np.absolute(sol.y[1, :])))


if __name__ == '__main__':
    Pop_size = 30
    nVar = 49*3
    MaxGen = 200
    Pc = 0.95
    Pm = 0.001
    Pi = 0.001
    Er = 0.2
    prob = Problem(-3, 3, 1)

    bestpop, cgcurve = GA(Pop_size, nVar, MaxGen, Pc, Pm, Pi, Er, prob)
    print(bestpop.Chromosome)
    print(cgcurve)
