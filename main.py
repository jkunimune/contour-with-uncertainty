"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from matplotlib import pyplot as plt
from numpy import linspace, meshgrid, hypot, exp, random, newaxis, ravel

from colormap import colormap

rng = random.default_rng(0)

N = 10_000

a = rng.normal(-5, .1, (N, 1, 1))
b = rng.normal(1, .1, (N, 1, 1))
r = rng.normal(.6, .01, (N, 1, 1))
depth = rng.normal(6, .1, (N, 1, 1))
x0 = rng.normal(-.3, .02, (N, 1, 1))
y0 = rng.normal(-.1, .02, (N, 1, 1))
power = 3
r2 = rng.normal(.2, .01, (N, 1, 1))
depth2 = rng.uniform(0, 2**(1/2), (N, 1, 1))**2
x02 = rng.normal(-.2, .02, (N, 1, 1))
y02 = rng.normal(.4, .02, (N, 1, 1))

x = y = linspace(-1, 1, 51)
x = x[newaxis, :, newaxis]
y = y[newaxis, newaxis, :]

image = (a*x + b*y -
         depth*exp(-(hypot(x - x0, y - y0)/r)**power) -
         depth2*exp(-(hypot(x - x02, y - y02)/r2)**power))

fig, axs = plt.subplots(
	3, 3, figsize=(5, 5), facecolor="none",
	gridspec_kw=dict(
		wspace=0, hspace=0,
	)
)
axs = ravel(axs)
for i in range(9):
	axs[i].imshow(image[i].T, cmap=colormap)
	axs[i].contour(image[i].T, colors="w")
	axs[i].get_xaxis().set_visible(False)
	axs[i].get_yaxis().set_visible(False)
plt.show()
