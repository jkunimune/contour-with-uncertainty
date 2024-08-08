"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import linspace, hypot, exp, random, newaxis, ravel, mean, quantile, pi, minimum, floor, cos, sin, sqrt, \
	zeros, arange, meshgrid, stack
from skimage import measure

from colormap import colormap

rng = random.default_rng(0)


def main():
	N = 10_000
	M = 101

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
	y02 = rng.normal(.5, .02, (N, 1, 1))

	base_noise = 0.3*perlin_noise((1, M, M), [
		(1, 2**-0), (2, 2**-0), (4, 2**-2), (8, 2**-4), (16, 2**-6), (32, 2**-8),
	])
	variant_noise = 0.6*perlin_noise((1, M, M), [
		(1, 2**-0), (2, 2**-0), (4, 2**-1), (8, 2**-2), (16, 2**-3), (32, 2**-4),
	])

	x = y = linspace(-1, 1, M)
	x = x[newaxis, :, newaxis]
	y = y[newaxis, newaxis, :]

	image = (a*x + b*y -
	         depth*exp(-(hypot(x - x0, y - y0)/r)**power) -
	         depth2*exp(-(hypot(x - x02, y - y02)/r2)**power) +
	         base_noise + variant_noise)

	# plot a few samples from the distribution
	fig, axs = plt.subplots(
		3, 3, figsize=(5, 5), facecolor="none",
		gridspec_kw=dict(
			wspace=0, hspace=0,
		)
	)
	axs = ravel(axs)
	for i in range(9):
		plot_image(axs[i], image[i], colormap=colormap, vmin=-5.5, vmax=5)
		axs[i].contour(image[i].T, colors="w", levels=[-2.5])
	fig.tight_layout()

	# plot a contour with uncertainty
	fig, ax = plt.subplots(facecolor="none")
	plot_image(ax, mean(image, axis=0), colormap=colormap, vmin=-5.5, vmax=5)
	plot_contour(ax, image, color="w", level=-2.6)
	fig.tight_layout()

	plt.show()


def plot_image(ax, image, *, colormap, vmin, vmax):
	ax.imshow(image.T, cmap=colormap, origin="lower", vmin=vmin, vmax=vmax)
	ax.set_xlim(0, image.shape[0] - 1)
	ax.set_ylim(0, image.shape[1] - 1)
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)


def plot_contour(ax, images, level, color, credibility=.90, opacity=1):
	# calculate the bounds of the contour band
	outer_bound = measure.find_contours(quantile(images, 1/2 - credibility/2, axis=0), level)
	inner_bound = measure.find_contours(quantile(images, 1/2 + credibility/2, axis=0), level)
	# convert the bands into Path objects
	path_sections = outer_bound + [loop[::-1, :] for loop in inner_bound]
	paths = []
	for section in path_sections:
		path = []
		for i in range(len(section)):
			path.append((Path.MOVETO if i == 0 else Path.LINETO, section[i, :]))
		paths.append(path)
	# connect any that are open into a single mass
	fused_paths = [[]]
	for path in paths:
		closed = np.array_equal(path[0][1], path[-1][1])
		if closed:
			fused_paths.append(path)
		else:
			if len(fused_paths[0]) > 0:
				path[0] = (Path.LINETO, path[0][1])
			fused_paths[0] += path
	path = [segment for path in fused_paths for segment in path]
	if len(path) > 0:
		commands, points = zip(*path)
		ax.add_patch(PathPatch(Path(points, commands),
		                       facecolor=color,
		                       alpha=opacity,
		                       edgecolor="none"))


def perlin_noise(shape, layers):
	"""
	this function is copied and modified from Pierre Vigier (licensed under MIT license):
	https://github.com/pvigier/perlin-numpy/tree/master
	"""
	assert shape[-2] == shape[-1]
	total = zeros(shape)
	for frequency, amplitude in layers:
		period = (shape[-1] - 1)/frequency
		x = y = (arange(shape[-1])/period)  # a cartesian system in units of period
		X, Y = meshgrid(x, y, indexing="ij")
		i = minimum(floor(X), frequency - 1).astype(int)
		j = minimum(floor(Y), frequency - 1).astype(int)
		dX, dY = X - i, Y - j  # relative coordinates for each cell
		# Gradients
		angles = rng.uniform(0, 2*pi, shape[:-2] + (frequency + 1, frequency + 1))
		gradients = stack((cos(angles), sin(angles)), axis=-1)
		g00 = gradients[..., i, j, :]
		g10 = gradients[..., i + 1, j, :]
		g01 = gradients[..., i, j + 1, :]
		g11 = gradients[..., i + 1, j + 1, :]
		# Ramps
		n00 = dX*g00[..., 0] + dY*g00[..., 1]
		n10 = (dX - 1)*g10[..., 0] + dY*g10[..., 1]
		n01 = dX*g01[..., 0] + (dY - 1)*g01[..., 1]
		n11 = (dX - 1)*g11[..., 0] + (dY - 1)*g11[..., 1]
		# Interpolation
		cx, cy = interpolant(dX), interpolant(dY)
		n0 = n00*(1 - cx) + cx*n10
		n1 = n01*(1 - cx) + cx*n11
		total += amplitude*sqrt(2)*((1 - cy)*n0 + cy*n1)

	return total


def interpolant(t):
	return t*t*t*(t*(t*6 - 15) + 10)


if __name__ == "__main__":
	main()
