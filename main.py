"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import linspace, meshgrid, hypot, exp, random, newaxis, ravel, mean, quantile
from skimage import measure

from colormap import colormap

def main():
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

	# plot a few samples from the distribution
	fig, axs = plt.subplots(
		3, 3, figsize=(5, 5), facecolor="none",
		gridspec_kw=dict(
			wspace=0, hspace=0,
		)
	)
	axs = ravel(axs)
	for i in range(9):
		plot_image(axs[i], image[i], colormap=colormap)
		axs[i].contour(image[i].T, colors="w")
	fig.tight_layout()

	# plot a contour with uncertainty
	fig, ax = plt.subplots(facecolor="none")
	plot_image(ax, mean(image, axis=0), colormap=colormap)
	plot_contour(ax, image, color="w", level=0)
	fig.tight_layout()

	plt.show()

def plot_image(ax, image, colormap):
	ax.imshow(image.T, cmap=colormap, origin="lower")
	ax.set_xlim(0, image.shape[0] - 1)
	ax.set_ylim(0, image.shape[1] - 1)
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)

def plot_contour(ax, images, level, color, credibility=.90, opacity=1):
	outer_bound = measure.find_contours(quantile(images, 1/2 - credibility/2, axis=0), level)
	inner_bound = measure.find_contours(quantile(images, 1/2 + credibility/2, axis=0), level)
	path_sections = outer_bound + [loop[::-1, :] for loop in inner_bound]
	path_points = []
	path_commands = []
	for loop in path_sections:
		loop_x = loop[:, 0]
		loop_y = loop[:, 1]
		path_points += list(zip(loop_x, loop_y))
		path_commands += [Path.LINETO]*len(loop)
	path_commands[0] = Path.MOVETO
	if len(path_points) > 0:
		ax.add_patch(PathPatch(Path(path_points, path_commands),
		                       facecolor=color,
		                       alpha=opacity,
		                       edgecolor="none"))

if __name__ == "__main__":
	main()
