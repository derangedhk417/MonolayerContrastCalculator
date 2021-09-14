import os
import sys
sys.path.append("../")

from FPContrastCalculator import ContrastCalculator, ContrastCalculatorLoader

import code
import json
import argparse
import sys 
import os
import time
import csv


import matplotlib.pyplot as plt
import numpy             as np

from scipy.integrate   import simps as simpson
from scipy.interpolate import interp1d

if __name__ == '__main__':

	# Initialize a calculator.
	calculator = ContrastCalculatorLoader(
		["graphene.csv", "silica_maltison_1965.csv", "silicon_green_1995.csv"],
		"ICX282AQ.csv",
		3200,
		1.0,
		heights=[3.34e-10, 90e-9],
		lens={
			'NA' : 0.42,
			'spectral_domain' : [
				402e-9,
				698e-9
			]
		},
		wavelength_resolution=128,
		angle_resolution=128
	).getCalculator()


	# Load the results file so we can compare.
	with open("result.csv", 'r') as file:
		raw = file.read()

	lines = raw.split("\n")[1:]
	cells = np.array([[float(i) for i in j.split(",")] for j in lines])

	# Perform the calculation.
	domain  = cells[:, 0]
	r, g, b = [], [], []

	for i, t in enumerate(domain):
		heights    = calculator.heights
		heights[1] = t
		calculator.setHeights(heights)
		r0, g0, b0 = calculator.getContrast(1)
		r.append(-r0)
		g.append(-g0)
		b.append(-b0)

		print("%d / %d"%(i + 1, len(domain)))


	# Graph a comparison of our results with Jessen et. al.
	ours_red   = plt.scatter(domain, r, color='red', marker='+', s=100)
	ours_green = plt.scatter(domain, g, color='green', marker='+', s=100)
	ours_blue  = plt.scatter(domain, b, color='blue', marker='+', s=100)

	theirs_red,   = plt.plot(domain, cells[:, 1], color='red')
	theirs_green, = plt.plot(domain, cells[:, 2], color='green')
	theirs_blue,  = plt.plot(domain, cells[:, 3], color='blue')

	plt.legend([
		ours_red, ours_green, ours_blue,
		theirs_red, theirs_green, theirs_blue
	], [
		"Red (my calculation)", "Green (my calculation)", "Blue (my calculation)",
		"Red (Jessen et. al.)", "Green (Jessen et. al.)", "Blue (Jessen et. al.)"
	])

	plt.xlabel(r"$SiO_2$ Thickness [m]")
	plt.ylabel("Contrast")
	plt.title(r"Optical Contrast as a Function of $SiO_2$ Thickness")
	plt.show()




	# ======================================================================================
	# Here we construct another calculator meant to emulate the system used by Nagashio et. al.

	calculator = ContrastCalculatorLoader(
		["graphene.csv", "silica_maltison_1965.csv", "silicon_green_1995.csv"],
		"ICX282AQ.csv",
		3200,
		1.0,
		heights=[3.34e-10, 90e-9],
		lens={
			'NA' : 0.9,
			'spectral_domain' : [
				402e-9,
				698e-9
			]
		},
		wavelength_resolution=128,
		angle_resolution=128
	).getCalculator()

	# Load the results file so we can compare.
	with open("result2.csv", 'r') as file:
		raw = file.read()

	lines = raw.split("\n")[1:]
	cells = np.array([[float(i) for i in j.split(",")] for j in lines])

	# Perform the calculation.
	domain  = cells[:, 0]
	r, g, b = [], [], []

	for i, t in enumerate(domain):
		heights    = calculator.heights
		heights[1] = t
		calculator.setHeights(heights)
		r0, g0, b0 = calculator.getContrast(1)
		r.append(-r0)
		g.append(-g0)
		b.append(-b0)

		print("%d / %d"%(i + 1, len(domain)))


	# Graph a comparison of our results with Nagashio et. al.
	# We need to average across the channels to compare to their greyscale camera.
	ours = (np.array(r) + np.array(g) + np.array(b)) / 3

	ours_sc    = plt.scatter(domain, ours, marker='+', s=100)
	theirs_pl, = plt.plot(domain, cells[:, 1], color='red')

	plt.legend([
		ours_sc, theirs_pl
	], [
		"my calculation", "Nagashio et. al."
	])

	plt.xlabel(r"$SiO_2$ Thickness [m]")
	plt.ylabel("Contrast")
	plt.title(r"Optical Contrast as a Function of $SiO_2$ Thickness")
	plt.show()