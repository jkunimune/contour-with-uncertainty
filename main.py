"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import linspace, hypot, exp, random, newaxis, ravel, mean, quantile, pi, minimum, floor, cos, sin, sqrt, \
	zeros, arange, meshgrid, stack, histogram, full, concatenate, interp, nonzero, diff, argsort
from skimage import measure

from height_colormap import height_colormap
from probability_colormap import probability_colormap

rng = random.default_rng(0)

rainbow_colormap = plt.get_cmap("turbo")

plt.rcParams["font.size"] = 12


def main():
	N = 10_000
	M = 101

	a = rng.normal(-5, .1, (N, 1, 1))
	b = rng.normal(1, .1, (N, 1, 1))
	r = rng.normal(.6, .02, (N, 1, 1))
	depth = rng.normal(6, .2, (N, 1, 1))
	x0 = rng.normal(-.3, .02, (N, 1, 1))
	y0 = rng.normal(-.1, .02, (N, 1, 1))
	power = 3
	r2 = rng.normal(.3, .01, (N, 1, 1))
	depth2 = rng.uniform(0, 2**(1/2), (N, 1, 1))**2
	x02 = rng.normal(-.2, .02, (N, 1, 1))
	y02 = rng.normal(.5, .02, (N, 1, 1))
	r3 = rng.normal(.2, .01, (N, 1, 1))
	depth3 = rng.uniform(0, 1**(1/2), (N, 1, 1))**2
	x03 = rng.normal(.2, .02, (N, 1, 1))
	y03 = rng.normal(-.4, .02, (N, 1, 1))

	base_noise = 0.3*perlin_noise((1, M, M), [
		(1, 2**-0), (2, 2**-0), (4, 2**-2), (8, 2**-4), (16, 2**-6), (32, 2**-8),
	])
	variant_noise = 0.6*perlin_noise((1, M, M), [
		(1, 2**-0), (2, 2**-0), (4, 2**-1), (8, 2**-2), (16, 2**-3), (32, 2**-4),
	])

	x = y = linspace(-1, 1, M)
	X = x[newaxis, :, newaxis]
	Y = y[newaxis, newaxis, :]
	z_edges = linspace(-5.2, 7.2, 201)
	z_centers = (z_edges[0:-1] + z_edges[1:])/2

	image = (a*X + b*Y -
	         depth*exp(-(hypot(X - x0, Y - y0)/r)**power) -
	         depth2*exp(-(hypot(X - x02, Y - y02)/r2)**power) +
	         depth3*exp(-(hypot(X - x03, Y - y03)/r3)**power) +
	         base_noise + variant_noise + 1)
	lineout = image[:, :, M//2]

	os.makedirs("figures", exist_ok=True)

	# plot a bunch of overlapping lineouts
	fig, ax = plt.subplots(facecolor="none", figsize=(6, 4))
	ax.plot(x, lineout[:9, :].T, zorder=1)
	ax.set_xlim(x[0], x[-1])
	ax.set_ylim(-5, 5)
	ax.grid()
	save_plot(fig, [ax], "figures/lineouts.png")

	# plot a histogram for every point on the lineout (not every single point)
	z05, z95 = quantile(lineout, [.05, .95], axis=0)
	fig_empty, ax_empty = plt.subplots(facecolor="none", figsize=(6, 4))
	fig_full, ax_full = plt.subplots(facecolor="none", figsize=(6, 4))
	for i in range(0, M, 5):
		density, _ = histogram(lineout[:, i], bins=z_edges)
		density = density/density.mean()*0.004
		ax_empty.plot(x[i] + density, z_centers,
		              linewidth=1.0, zorder=1.01 - i*.1)
		ax_empty.fill_betweenx(z_centers, x[i], x[i] + density,
		                       zorder=1.00 - i*.1, color="white")
		ax_full.plot(x[i] + density, z_centers,
		             linewidth=1.0, zorder=1.02 - i*.1, color=f"C{i//5}")
		z_interval = concatenate([[z05[i]], z_centers[(z_centers > z05[i]) & (z_centers < z95[i])], [z95[i]]])
		ax_full.fill_betweenx(z_interval, x[i], x[i] + interp(z_interval, z_centers, density),
		                      zorder=1.01 - i*.1, color=f"#aaa")
	ax_full.plot(x, z05, linewidth=1.0, color="k", zorder=-100)
	ax_full.plot(x, z95, linewidth=1.0, color="k", zorder=-100)
	for ax in [ax_empty, ax_full]:
		ax.set_xlim(x[0] - 0.05, x[-1] + 0.2)
		ax.set_ylim(-5, 5)
	ax_empty.grid(axis="y")
	save_plot(fig_empty, [ax_empty], "figures/histograms 1d.png")
	save_plot(fig_full, [ax_full], "figures/histograms 1d with intervals.png")

	# plot the lineout as a heatmap of cumulative probability
	fig, ax = plt.subplots(facecolor="none", figsize=(7, 4))
	cdf = mean(z_centers[newaxis, newaxis, :] > lineout[:, :, newaxis], axis=0)*100
	picture = plot_image(ax, cdf, vmin=0, vmax=100, colormap=probability_colormap,
	                     extent=(1.5*x[0] - 0.5*x[1], 1.5*x[-1] - 0.5*x[-2], z_edges[0], z_edges[-1]),
	                     interpolation="bilinear", aspect="auto")
	ax.contour(x, z_centers, cdf.T, levels=[5, 95], colors=["black", "black"])
	plt.colorbar(picture, ax=ax, format=lambda x, pos: f"{x:.0f}%").set_label("Probability")
	ax.text(0, -1, "5%")
	ax.text(0, 1, "95%")
	ax.set_xlim(x[0], x[-1])
	ax.set_ylim(-5, 5)
	save_plot(fig, [ax], "figures/probability density 1d.png")

	# plot a histogram for every single pixel (not every single pixel)
	fig = plt.figure(facecolor="none", figsize=(5, 5))
	ax = fig.add_subplot(projection="3d")
	ax.set_axis_off()
	normalize = Normalize(vmin=-4.5, vmax=6.0)
	colors = height_colormap(normalize(mean(image, axis=0)))
	xi, yi = meshgrid(M, M, indexing="ij")
	zi = np.full_like(xi, 0)
	ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False)
	for i in range(0, image.shape[1], 10):
		for j in range(0, image.shape[2], 10):
			density, _ = histogram(image[:, i, j], bins=z_edges)
			ax.plot(i + density/density.max()*10,
			        full(z_centers.shape, j),
			        z_centers - z_edges[0], color=f"C{random.randint(0, 10)}", linewidth=1.0, zorder=100 - j)
	ax.set_xlim(0, M - 1)
	ax.set_ylim(0, M - 1)
	ax.set_zlim(0, z_edges[-1] - z_edges[0])
	ax.view_init(30, -75)
	save_plot(fig, [ax], "figures/histograms 2d.png")

	# plot a few samples from the distribution
	fig, axs = plt.subplots(
		3, 3, figsize=(5, 5), facecolor="none",
		gridspec_kw=dict(
			wspace=0, hspace=0,
		)
	)
	axs = ravel(axs)
	for i in range(9):
		plot_image(axs[i], image[i], vmin=-4.5, vmax=6.0, colormap=height_colormap)
	save_plot(fig, axs, "figures/samples.png")

	# plot a bunch of overlapping contours
	fig, ax = plt.subplots(facecolor="none", figsize=(5, 5))
	plot_image(ax, mean(image, axis=0), vmin=-4.5, vmax=6.0, colormap=height_colormap)
	for i in range(9):
		ax.contour(image[i, :, :].T, levels=[-1.5], colors="white", linewidths=1.0)
	save_plot(fig, [ax], "figures/contours.png")

	# instead of plotting the mean of the distribution, plot the amount over a certain level
	fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
	plot_image(ax, mean(image > -1.6, axis=0), vmin=0, vmax=1, colormap=probability_colormap, interpolation="bilinear")
	save_plot(fig, [ax], "figures/probability density 2d.png")

	# plot a contour with uncertainty
	fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
	plot_image(ax, mean(image, axis=0), vmin=-4.5, vmax=6.0, colormap=height_colormap)
	plot_contour(ax, image, level=-1.5)
	save_plot(fig, [ax], "figures/punchline.png")

	# plot multiple contours with uncertainty
	fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
	plot_contour(ax, image, level=-3.0, color=rainbow_colormap(.0))
	plot_contour(ax, image, level=-1.5, color=rainbow_colormap(.2))
	plot_contour(ax, image, level=0.0, color=rainbow_colormap(.4))
	plot_contour(ax, image, level=1.5, color=rainbow_colormap(.6))
	plot_contour(ax, image, level=3.0, color=rainbow_colormap(.8))
	plot_contour(ax, image, level=4.5, color=rainbow_colormap(1.0))
	ax.set_xlim(0, M - 1)
	ax.set_ylim(0, M - 1)
	save_plot(fig, [ax], "figures/multiple bands.png")

	# plot the contour level and contour band edges over the lineout band
	fig, ax = plt.subplots(facecolor="none", figsize=(7, 4))
	picture = plot_image(ax, cdf, vmin=0, vmax=100, colormap=probability_colormap,
	                     extent=(1.5*x[0] - 0.5*x[1], 1.5*x[-1] - 0.5*x[-2], z_edges[0], z_edges[-1]),
	                     interpolation="bilinear", aspect="auto")
	plt.colorbar(picture, ax=ax, format=lambda x, pos: f"{x:.0f}%").set_label("Probability")
	ax.contour(x, z_centers, cdf.T, levels=[5, 95], colors=["black", "black"])
	ax.axhline(-1.5, color="black", linestyle="dashed")
	intersections = []
	for z in [z05, z95]:
		for i in nonzero(diff(z > -1.5))[0]:
			order = arange(i, i + 2)[argsort([z[i], z[i + 1]])]
			x_crossing = float(interp(-1.5, z[order], x[order]))
			intersections.append(x_crossing)
	intersections = sorted(intersections)
	for i in range(0, len(intersections) - 1, 2):
		ax.axvspan(intersections[i], intersections[i + 1], color="#aaa")
	ax.set_xlim(x[0], x[-1])
	ax.set_ylim(-5, 5)
	ax.grid()
	save_plot(fig, [ax], "figures/contour lineout.png")

	plt.show()


def save_plot(fig, axs, filename):
	for ax in axs:
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
	# if len(axs) == 1:
	fig.tight_layout()
	fig.savefig(filename, dpi=300)


def plot_image(ax, image, *, vmin, vmax, colormap, **kwargs):
	thing = ax.imshow(image.T, cmap=colormap, origin="lower", vmin=vmin, vmax=vmax, **kwargs)
	ax.set_xlim(0, image.shape[0] - 1)
	ax.set_ylim(0, image.shape[1] - 1)
	return thing


def plot_contour(ax, images, level, credibility=.90, color="white"):
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
