"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from matplotlib import pyplot as plt
from numpy import linspace, meshgrid, hypot, exp

from colormap import colormap

a = -5
b = 1
r = .6
depth = 6
x0 = -.3
y0 = -.1
power = 3
r2 = .2
depth2 = 2
x02 = -.2
y02 = .4

x = y = linspace(-1, 1, 51)
X, Y = meshgrid(x, y, indexing="ij")
image = (a*X + b*Y -
         depth*exp(-(hypot(X - x0, Y - y0)/r)**power) -
         depth2*exp(-(hypot(X - x02, Y - y02)/r2)**power))

plt.imshow(image.T, cmap=colormap)
plt.contour(image.T, colors="w")
plt.show()
