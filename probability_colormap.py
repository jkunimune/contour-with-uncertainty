"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license,  visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from matplotlib.colors import ListedColormap
from numpy import linspace

cm_data = [convert_color(LabColor(l, 0, 0), sRGBColor).get_value_tuple() for l in linspace(100, 50, 257)]
probability_colormap = ListedColormap(cm_data)
