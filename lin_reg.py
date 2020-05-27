
"""
This program animates an algorithm that finds and plots the line of best fit
for a data set
"""

from pylab import meshgrid, title
from matplotlib import cm

# reads in the files
import pandas as pd

# manages the data
import numpy as np

# plots the graph
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class Regression:
    """
    Creates a regression of the form y = theta0 + theta1x
    """
    def __init__(self, x, y, alpha, precision):
        self.x = np.array(x)
        self.y = np.array(y)
        # learning rate
        self.alpha = alpha
        self.precision = precision
        # setting  values to begin
        midpoint = (self.y.max() + self.y.min())/2
        self.theta0 = midpoint
        self.theta1 = 0
        self.iteration = 0
        if x.size == y.size:
            self.m = x.size
        else:
            print('Invalid Data')

        # setting up the graphs
        self.fig = plt.figure(figsize=(14,6))
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_title('Linear Regression')
        # latex style
        self.ax2.set_xlabel('$theta_0$')
        self.ax2.set_ylabel('$theta_1$')
        self.ax2.set_title('Cost($theta_0$, $theta_1$)')

        # regression scatter plot
        x, y = self.line_eq()
        self.line, = self.ax1.plot(x, y)
        self.ax1.plot(self.x, self.y, 'go')

        # cost function plot
        def cost(theta0, theta1):
            dist_squared = 0
            for i in range(self.m):
                dist_squared += (theta0 + theta1*self.x[i] - self.y[i])**2
            return dist_squared/(2*self.m)
        x = np.arange(midpoint - 10.0, midpoint + 10.0,0.2)
        y = np.arange(-10.0,10.0,0.2)
        X, Y = meshgrid(x, y)  # grid of points
        Z = cost(X, Y)  # evaluation of the function on the grid

        # contours
        levels = np.arange(-10.0,10.0,0.3)
        CS = self.ax2.contour(X, Y, Z, levels=levels, cmap=cm.Set2)
        # adding the Contour lines with labels
        self.ax2.clabel(CS, inline=True, fmt='%1.1f', fontsize=10)

        # setting up scatter points animation
        self.graph, = plt.plot([], [], 'o', markersize=3)
        self.theta0s = []
        self.theta1s = []


    def line_eq(self):
        x = np.linspace(self.x.min(), self.x.max(),100)
        y = self.theta0 + self.theta1*x
        return x, y

    def h_theta(self, x):
        return self.theta0 + self.theta1*x

    def gradient_descent(self):
        """Returns whether gradient has reached desired precision"""
        # partial derivatives for cost function
        dtheta0_sum = 0
        dtheta1_sum = 0
        for j in range(self.m):
            diff_squared = self.h_theta(self.x[j]) - self.y[j]
            dtheta0_sum += diff_squared
            dtheta1_sum += diff_squared*self.x[j]
        avg_dtheta0 = dtheta0_sum/self.m
        avg_dtheta1 = dtheta1_sum/self.m
        theta0_adj = self.alpha*avg_dtheta0
        theta1_adj = self.alpha*avg_dtheta1
        if abs(theta0_adj) < self.precision and abs(theta1_adj) < self.precision:
            return True
        self.theta0 -= theta0_adj
        self.theta1 -= theta1_adj
        return False

    def animate(self):
        def update(i):
            # plot
            x, y = self.line_eq()
            self.line.set_xdata(x)  # update the data
            self.line.set_ydata(y)  # update the data

            finished = self.gradient_descent()

            cost = self.cost()

            self.theta0s.append(self.theta0)
            self.theta1s.append(self.theta1)

            if not finished:
                # update variables
                self.iteration += 1
                print(f'Iteration: {self.iteration} | Cost: {cost} '
                      f'| y = {round(self.theta0,2)} + {round(self.theta1,2)}x')

            # insert point on cost function
            self.graph.set_data(self.theta0s, self.theta1s)

            return self.line, self.graph,

        # Init only required for blitting to give a clean slate.
        def init():
            self.line.set_ydata(self.y)
            return self.line,

        # you can change the speed of the animation by modifying "interval"
        ani = anim.FuncAnimation(self.fig, update, frames=np.arange(1, 10), init_func=init, interval=100, blit=True)
        plt.show()

    def cost(self):
        dist_squared = 0
        for i in range(self.m):
            dist_squared += (self.h_theta(self.x[i]) - self.y[i])**2
        return dist_squared/(2*self.m)



if __name__ == '__main__':
    df = pd.read_csv("2019.csv", delimiter=',')
    print(df.columns)
    alpha = 1  # learning rate
    precision = 0.01
    happy_reg = Regression(df['Freedom to make life choices'], df['Score'], alpha, precision)
    happy_reg.animate()

