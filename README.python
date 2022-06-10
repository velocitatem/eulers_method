import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Function:
    def compute(x, y):
        return None

class InitialData:
    def __init__(self, derivative_function, point):
        self.derivative_function = derivative_function
        self.point = point

def approximate(init, new):
    y = init.point[1] + \
        (new - init.point[0]) * \
        init.derivative_function.compute(
            init.point[0],
            init.point[1]
        )
    return (new,y)

def approximate_with_steps(init, start, end, steps):
    p_approx = init.point
    delta = (end - start) / steps
    r = np.arange(start, end+delta, delta)
    for i in r:
        d = InitialData(init.derivative_function, p_approx)
        p_approx = apporoximate(d, i)
    return p_approx

class function(Function):
    def compute(x,y):
        return np.sin(x*y)

ic_s = (0,1) # initial condition (x,y)
mx = 30 # max evaluation x -- x: [0,mx]

def solution(x): # solution to the differential equation
    return 5*x - 7

res_range = [0.01, 2]
approximations = 5

ic = ic_s
data = InitialData(function, ic)
error_plot_cache = []

resolutions = np.round_(np.arange(res_range[0], res_range[1], (res_range[1]-res_range[0])/approximations), 3)

for res in resolutions:

    ys = []
    ysc = []

    rng = np.arange(ic_s[0], mx+res, res)
    for i in rng:
        ic = approximate(
            InitialData(function, ic),
            i)
        ys.append(ic[1])
        ysc.append(solution(i))

    plt.subplot(1,2,1)
    plt.plot(rng, ys, label = "Step Size: " + str(res) )

    ys = np.array(ys)
    ysc = np.array(ysc)

    error = np.abs(ysc-ys)

    plt.subplot(1,2,2)
    plt.plot(rng,error , label = "ε for step size " + str(res))

    ic = ic_s

rng_og = np.arange(ic_s[0], mx, 0.1)
plt.subplot(1,2,1)
#plt.plot(rng_og, solution(rng_og), label = "Solution") # Uncomment to show real solution

plt.legend()
plt.subplot(1,2,2)
plt.legend()
plt.show()
