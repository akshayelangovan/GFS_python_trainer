import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class fuzzyCreate:
    def __init__(self, universe, no_of_mf, rule_params):
        # Designing the FIS
        self.input1 = ctrl.Antecedent(np.arange(universe[0], universe[1], universe[2]), 'input1')
        self.input2 = ctrl.Antecedent(np.arange(universe[0], universe[1], universe[2]), 'input2')
        self.output = ctrl.Consequent(np.arange(universe[0], universe[1], universe[2]), 'output')

        if no_of_mf == 3:
            names = ['-1', '0', '1']
        elif no_of_mf == 5:
            names = ['-2', '-1', '0', '1', '2']
        elif no_of_mf == 7:
            names = ['-3', '-2', '-1', '0', '1', '2', '3']
        else:
            print("Error!: no_of_mf not defined for this value, check fuzzy_create.py")

        self.input1.automf(names=names)
        self.input2.automf(names=names)
        self.output.automf(names=names)

        # Setting up the rule base of the FIS
        # rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
        k = 0
        rulelist = []
        for j in names:
            for i in names:
                rulelist.append(ctrl.Rule(self.input1[j] & self.input2[i], self.output[str(rule_params[k])]))
                k = k+1

        self.system = ctrl.ControlSystem(rules = rulelist)
        self.sim = ctrl.ControlSystemSimulation(self.system)


if __name__ == '__main__':
    # Basic code to test class
    univ = [-3, 3, 0.001]
    fis1 = fuzzyCreate(univ, 3, [-1, -1, -1, 0, 0, 0, 0, 1, 1])
    fis1.sim.input['input1'] = 0.5
    fis1.sim.input['input2'] = 0
    fis1.sim.compute()
    print(fis1.sim.output['output'])
    fis1.output.view(sim=fis1.sim)
    plt.show()
