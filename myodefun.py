import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from fuzzy_create import fuzzyCreate

def u(x, z, theta, xdot, zdot, thetadot, fis1, fis2, fis3):
    fis1.sim.input['input1'] = x
    fis1.sim.input['input2'] = xdot
    fis1.sim.compute()
    thetades  = fis1.sim.output['output']
    fis2.sim.input['input1'] = theta - thetades
    fis2.sim.input['input2'] = thetadot
    fis2.sim.compute()
    tau = fis2.sim.output['output']
    fis3.sim.input['input1'] = z
    fis3.sim.input['input2'] = zdot
    fis3.sim.compute()
    F = fis3.sim.output['output']
    return [F, tau]

def odefun(t, y, fis1, fis2, fis3, sys):
    #ode goes here along with parameters
    x = y[0]
    z = y[1]
    alpha = y[2]
    theta = y[3]
    xdot = y[4]
    zdot = y[5]
    alphadot = y[6]
    thetadot = y[7]
    # Need to pass M,m,l,g,J
    M = sys[0]
    m = sys[1]
    l = sys[2]
    g = sys[3]
    J = sys[4]

    #do a function call for control input u
    [F, tau] = u(x, z, theta, xdot, zdot, thetadot, fis1, fis2, fis3)
    #calculate error and pass to controller to get [F, tau]
    fx = F * np.sin(theta) * (M+m)*g
    fz = F * np.cos(theta) * (M+m)*g

    #Equations of motion
    xddot = ((M+m*np.cos(alpha)*np.cos(alpha))/(M*(M+m)))*fx + ((m*np.sin(alpha)*np.cos(alpha))/(M*(M+m)))*fz + (m*l*(alphadot**2)*np.sin(alpha))/(M+m)
    zddot = ((m * np.sin(alpha) * np.cos(alpha)) / (M * (M + m))) * fx + ((M + m * np.sin(alpha) * np.sin(alpha)) / (M * (M + m))) * fz - (m * l * alphadot ** 2 * np.cos(alpha)) / (M + m) - g
    alphaddot = -np.cos(alpha) * fx / (M * l) - np.sin(alpha) * fz / (M * l)
    thetaddot = tau / J
    return [xdot, zdot, alphadot, thetadot, xddot, zddot, alphaddot, thetaddot]


if __name__=="__main__":
    x0 = [0, 0, 0, 0, 0, 0, 0, 0]
    univ = [-3, 3, 0.001]
    sys = [1, 0.5, 1, 9.81, 4.4 * 10 ** -3]  # M, m, l, g, J
    fis1 = fuzzyCreate(univ, 3, [-1, -1, -1, 0, 0, 0, 0, 1, 1])
    fis2 = fuzzyCreate(univ, 3, [-1, -1, -1, 0, 0, 0, 0, 1, 1])
    fis3 = fuzzyCreate(univ, 3, [-1, -1, -1, 0, 0, 0, 0, 1, 1])

    sol = solve_ivp(odefun, [0, 10], x0, t_eval=np.linspace(0, 10, 300), args=(fis1, fis2, fis3, sys))
    plt.plot(sol.t, sol.y[1, :])
    #plt.show()
    print(np.dot(sol.t, np.absolute(sol.y[1, :])))
