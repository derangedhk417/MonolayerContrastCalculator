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

# Process the command line arguments supplied to the program.
def preprocess(args_specification):
	parser = argparse.ArgumentParser(description=args_specification['description'])

	types = {'str': str, 'int': int, 'float': float}

	for argument in args_specification['arguments']:

		spec = argument['spec']
		if 'type' in spec:
			spec['type'] = types[spec['type']]
		parser.add_argument(
			*argument['names'], 
			**spec
		)

	args = parser.parse_args()

	return args


class ContrastCalculator:
	def __init__(self, n, d, camera, source, NA, wavelength_range):
		"""
			n : A list of refractive indices for each material in order from the
			    topmost material to the bottom material. Each element can be 
			    either a complex(n, k) or a tuple of arrays. The first array 
			    should correspond to the wavelength and the second array should
			    correspond to the complex valued refactive index of the material
			    at that wavelength (in meters).
			d : The thickness of each material in the stack. This value should
			    be given in meters. You do not need to include the thickness of
			    the bottom layer, and it will not be used.

			camera : A tuple of three "functions". These correspond to the R, G
			         and B channel response functions of the camera. Each should
			         itself be a tuple of two arrays. The first corresponding to
			         wavelength and the second corresponding to the response of
			         the camera at that wavelength (unitless). Wavelengths should
			         be given in meters. These functions will be normalized 
			         before use.
			source : This should be a tuple of two "functions". The first is the
			         color spectrum of the source light. If a real number is 
			         specified, this will be calculated using Planck's law. 
			         Otherwise, this should be a tuple of two arrays where the
			         first array corresponds to the wavelength and the second 
			         array corresponds to the intensity. This will be normalized.
			         Wavelength should be in meters. The second function, should
			         describe the intensity of the source as a function of the 
			         angle of incidence at which it is hitting the sample. The 
			         first array in this tuple should correspond to angles in 
			         radians and the second array should correspond to the 
			         intensity of the source at that incident angle. This will
			         be normalized automatically.

			NA : Numerical Aperture of the lens. This program assumes it's 
			     operating in air (n = 1)
		"""

		self.d = d

		self.wvCount = 256 # The number of subdivisions of the wavelength domain
		self.tCount  = 256 # Number of subdivisions of the theta domain
		self.angle   = np.arcsin(NA)

		# Determine the smallest domain of wavelength values across all of the
		# different functions that are supplied. We will convert all functions
		# to the same domain. 
		bounds    = self._findWavelengthDomain(n, camera, source)
		bounds[0] = max(bounds[0], wavelength_range[0])
		bounds[1] = min(bounds[1], wavelength_range[1])
		x         = np.linspace(bounds[0], bounds[1], self.wvCount)
		t         = np.linspace(0, self.angle, self.tCount)

		# Air
		self.immersion_medium = complex(1.0003, 0.0)

		# Now we need to convert everything that wasn't specified as an array 
		# into and array so we can interpolate it. We use this interpolation
		# to resample everything so that the domain of all wavelength dependent
		# functions is the same array.
		self._resampleWavelengthData(n, camera, source, x)

		# If the value given for the source intensity as a function of angle
		# is a scalar instead of a tuple of arrays, then we assume that it's
		# the sigma value in a gaussian function. Here, we calculate a set of 
		# values for it.
		self.tDomain = t
		if type(source[1]) is not tuple:
			self.sourceIntensity = np.exp(
				-np.square(self.tDomain) / (2 * np.square(source[1]))
			)
		else:
			# Resample this to match the specified domain for the angle.
			interp = interp1d(source[1][0], source[1][1], kind='cubic')
			self.sourceIntensity = interp(self.tDomain)

		# We now have all of our data in a form that can be numerically 
		# integrated over very efficiently. Now we normalize everything.
		# self._normalize()

		# All of the data is now uniform along it's relevant x-axis and 
		# normalized. We are now equipped to calculate the overall intensity
		# of light as detected by the camera on each color channel.

	def getContrast(self, substrate_idx):
		rs, gs, bs = self.getIntensity(substrate_idx)
		rt, gt, bt = self.getIntensity(0)

		rc = -(rs - rt) / rs
		gc = -(gs - gt) / gs
		bc = -(bs - bt) / bs

		return rc, gc, bc

	def getIntensity(self, idx):
		# We need to define a function that will calculate the raw intensity
		# of reflected light as a function of wavelength and incident angle.

		def getTransmissionAngle(t, n0, n1):
			return np.arcsin((n0 / n1) * np.sin(t))

		def partialReflection_p(t0, t1, n0, n1):
			numerator   = n1 * np.cos(t0) - n0 * np.cos(t1)
			denominator = n1 * np.cos(t0) + n0 * np.cos(t1)
			return numerator / denominator

		def partialReflection_s(t0, t1, n0, n1):
			numerator   = n0 * np.cos(t0) - n1 * np.cos(t1)
			denominator = n0 * np.cos(t0) + n1 * np.cos(t1)
			return numerator / denominator

		def reflectionCoefficient_p(t0, indices, w):
			n0  = self.immersion_medium
			n1  = indices[idx]
			t1  = getTransmissionAngle(t0, n0, n1)
			r0  = partialReflection_p(t0, t1, n0, n1)

			if idx == len(self.refractiveData) - 1:
				return r0

			phi = 4 * np.pi * n1 * np.cos(t1) * self.d[idx] / w

			inner = _reflectionCoefficient_p(indices, t1, w, n1, idx + 1)

			ex = np.exp(-1j * phi)
			return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def _reflectionCoefficient_p(indices, t0, w, n0, i):
			N = len(self.refractiveData)

			n1 = indices[i]
			t1 = getTransmissionAngle(t0, n0, n1)
			r0 = partialReflection_p(t0, t1, n0, n1)

			if i == N - 1:
				return r0
			else:
				phi   = 4 * np.pi * n1 * np.cos(t1) * self.d[i] / w
				inner = _reflectionCoefficient_p(indices, t1, w, n1, i + 1)
				ex    = np.exp(-1j * phi)

				return (r0 + inner * ex) / (1 + r0 * inner * ex)

		def reflectionCoefficient_s(t0, indices, w):
			n0  = self.immersion_medium
			n1  = indices[idx]
			t1  = getTransmissionAngle(t0, n0, n1)
			r0  = partialReflection_s(t0, t1, n0, n1)

			if idx == len(self.refractiveData) - 1:
				return r0

			phi = 4 * np.pi * n1 * np.cos(t1) * self.d[idx] / w

			inner = _reflectionCoefficient_s(indices, t1, w, n1, idx + 1)

			ex = np.exp(-1j * phi)
			return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def _reflectionCoefficient_s(indices, t0, w, n0, i):
			N = len(self.refractiveData)

			n1 = indices[i]
			t1 = getTransmissionAngle(t0, n0, n1)
			r0 = partialReflection_s(t0, t1, n0, n1)

			if i == N - 1:
				return r0
			else:
				phi   = 4 * np.pi * n1 * np.cos(t1) * self.d[i] / w
				inner = _reflectionCoefficient_s(indices, t1, w, n1, i + 1)
				ex    = np.exp(-1j * phi)

				return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def innerIntegrand(t, indices, w):
			Rp = reflectionCoefficient_p(t, indices, w)
			Rs = reflectionCoefficient_s(t, indices, w)
			I = (Rp.real**2 + Rp.imag**2 + Rs.real**2 + Rs.imag**2) / 2

			return I * self.sourceIntensity

		def angleIntegral(w):
			index   = np.where(self.wvDomain == w)[0][0]
			indices = np.array([layer[index] for layer in self.refractiveData])
			# Get the source intensity and the refractive index of each layer 
			# at this wavelength.
			y = innerIntegrand(self.tDomain, indices, w)

			return simpson(y, self.tDomain)

		rawIntensity = []
		for w in self.wvDomain:
			rawIntensity.append(angleIntegral(w))

		def getChannel(channel):
			channelIntensity = None
			if channel == 'r':
				channelIntensity = self.redResponse
			elif channel == 'g':
				channelIntensity = self.greenResponse
			elif channel == 'b':
				channelIntensity = self.blueResponse

			integrand = np.array(rawIntensity) * channelIntensity * self.sourceSpectrum

			return simpson(integrand, self.wvDomain)

		return getChannel('r'), getChannel('g'), getChannel('b')

	def _findWavelengthDomain(self, n, camera, source):
		wvMin = 0.0
		wvMax = 1.0

		# Check refractive indices.
		for ni in n:
			if not type(ni) is complex:
				# This isn't a single value, so its a tuple of two arrays.
				lower = np.array(ni[0]).min()
				upper = np.array(ni[0]).max()
				wvMin = max(wvMin, lower)
				wvMax = min(wvMax, upper)

		# Check the camera sensitivity curves.
		R_min = np.array(camera[0][0]).min()
		R_max = np.array(camera[0][0]).max()
		G_min = np.array(camera[1][0]).min()
		G_max = np.array(camera[1][0]).max()
		B_min = np.array(camera[2][0]).min()
		B_max = np.array(camera[2][0]).max()

		wvMin = max(wvMin, R_min)
		wvMin = max(wvMin, G_min)
		wvMin = max(wvMin, B_min)

		wvMax = min(wvMax, R_max)
		wvMax = min(wvMax, G_max)
		wvMax = min(wvMax, B_max)

		if type(source[0]) is tuple:
			# Check the source spectrum.
			sourceMin = np.array(source[0][0]).min()
			sourceMax = np.array(source[0][0]).max()

			wvMin = max(wvMin, sourceMin)
			wvMax = min(wvMax, sourceMax)

		return [wvMin, wvMax]

	def _resampleWavelengthData(self, n, camera, source, x):
		# Start with refractive indices.
		refractiveIndices = []

		for ni in n:
			if type(ni) is complex:
				y = np.ones(self.wvCount) * ni
				refractiveIndices.append((x, y))
			else:
				refractiveIndices.append(ni)

		sourceSpectrum = None
		# Now convert the source spectrum if only a color temperature was 
		# specified.
		if type(source[0]) is not tuple:
			if type(source[0]) is str:
				# Treat this as a FWHM and center wavelength for a "monochromatic" source.
				center, fwhm    = [float(i) for i in source[0].split(',')]
				s               = fwhm / np.sqrt(2 * np.log(2))
				def gaussian(x, s, x0):
					A = (1 / (s * np.sqrt(2*np.pi)))
					return A * np.exp(-np.square(x - x0) / (2 * np.square(s)))
				sourceSpectrum = (x, gaussian(x, s, center))
				# plt.plot(*sourceSpectrum)
				# plt.show()
			else:
				h = 6.62607015e-34
				c = 299792458
				k = 1.380649e-23
				def Planck(wv, T):
					res = (2 * h * c / (wv**3))
					res = res * (1 / (np.exp((h * c) / (wv * k * T)) - 1))
					return res

				sourceSpectrum = (x, Planck(x, source[0]))
		else:
			sourceSpectrum = source[0]

		# We should now have all of functions with wavelength for an independent
		# variable in the form of arrays. Now we interpolate each of them and
		# use the interpolation to resample them so they all have the same 
		# wavelength values.

		self.wvDomain       = x
		self.refractiveData = []
		for ni in refractiveIndices:
			interp = interp1d(ni[0], ni[1], kind="cubic")
			self.refractiveData.append(interp(self.wvDomain))

		# Refractive index data is now in the proper form for efficient numeric
		# integration. Now we do the same for the color response curve of the
		# camera.

		Rinterp = interp1d(camera[0][0], camera[0][1], kind='cubic')
		Ginterp = interp1d(camera[1][0], camera[1][1], kind='cubic')
		Binterp = interp1d(camera[2][0], camera[2][1], kind='cubic')

		self.redResponse   = Rinterp(self.wvDomain)
		self.greenResponse = Ginterp(self.wvDomain)
		self.blueResponse  = Binterp(self.wvDomain)

		# We've now resampled the color responses as well. Next we handle the
		# spectrum of the source.
		spectrumInterp = interp1d(
			sourceSpectrum[0], 
			sourceSpectrum[1], 
			kind='cubic'
		)
		self.sourceSpectrum = spectrumInterp(self.wvDomain)

		# At this point, every input function that has wavelength as an
		# independent variable has been resampled so that they all have the same
		# x-coordinate values and so that their domain is equal to that of the
		# lowest common denominator.

	def _normalize(self):
		redConstant   = simpson(self.redResponse,   self.wvDomain)
		greenConstant = simpson(self.greenResponse, self.wvDomain)
		blueConstant  = simpson(self.blueResponse,  self.wvDomain)

		self.redResponse   = self.redResponse   / redConstant
		self.greenResponse = self.greenResponse / greenConstant
		self.blueResponse  = self.blueResponse  / blueConstant

		spectrumConstant = simpson(self.sourceSpectrum, self.wvDomain)
		self.sourceSpectrum = self.sourceSpectrum / spectrumConstant

		intensityConstant = simpson(self.sourceIntensity, self.tDomain)
		self.sourceIntensity = self.sourceIntensity / intensityConstant

def getCSVFloats(path):
	with open(path, 'r') as file:
		r = csv.reader(file, delimiter=",")
		data = list(r)

	rows = [[float(c) for c in r] for r in data[1:]]
	return rows

if __name__ == '__main__':
	# Load the arguments file. 
	with open("Contrast.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# Load all of the relevant data files in preparation for initializing the
	# calculator.
	refractive_data = []
	for n in args.refractive_indices:
		try:
			c = complex(n)
			refractive_data.append(c)
		except:
			rows = getCSVFloats(n)
			wv   = np.array([i[0] for i in rows]) * 1e-6
			n0   = np.array([i[1] for i in rows])
			try:
				k0   = np.array([i[2] for i in rows])
			except:
				# No k data.
				k0 = np.ones(len(n0)) * 0.0

			if args.plot_data:
				plt.plot(wv, n0)
				plt.plot(wv, k0)
				plt.show()

			n0 = [complex(n0[i], k0[i]) for i in range(len(n0))]
			refractive_data.append((wv, n0))

	color_data = getCSVFloats(args.camera)

	wv = [i[0] for i in color_data]
	r  = [i[1] for i in color_data]
	g  = [i[2] for i in color_data]
	b  = [i[3] for i in color_data]

	if args.plot_data:
		plt.plot(wv, r, color="red")
		plt.plot(wv, g, color="green")
		plt.plot(wv, b, color="blue")
		plt.show()

	camera = ((wv, r), (wv, g), (wv, b))

	try:
		s = float(args.source_spectrum)
		source_spectrum = s
	except:
		if ',' in args.source_spectrum:
			source_spectrum = args.source_spectrum
		else:
			data = getCSVFloats(args.source_spectrum)
			wv   = [i[0] for i in data]
			I    = [i[1] for i in data]
			source_spectrum = (wv, I)

			if args.plot_data:
				plt.plot(wv, I)
				plt.show()

	try:
		g = float(args.source_angle_dependence)
		source_angle_dependence = g
	except:
		data = getCSVFloats(args.source_angle_dependence)
		angle = [i[0] for i in data]
		I     = [i[1] for i in data]
		source_angle_dependence = (wv, I)
		if args.plot_data:
			plt.plot(wv, I)
			plt.show()

	source = (source_spectrum, source_angle_dependence)
	args.wavelength_range = [
		args.wavelength_range[0] * 1e-9,
		args.wavelength_range[1] * 1e-9
	]

	# Color Temperature Variance Test
	# rs, gs, bs = [], [], []
	# K          = np.linspace(2000, 4000, 256)

	# for i, k in enumerate(K):
	# 	print("%d / %d"%(i + 1, len(K)))
	# 	source = (k, source[1])
	# 	calculator = ContrastCalculator(
	# 		refractive_data, args.thicknesses, camera, 
	# 		source, args.numerical_aperture, args.wavelength_range
	# 	)

	# 	r, g, b = calculator.getContrast(args.substrate_index)
	# 	rs.append(r)
	# 	gs.append(g)
	# 	bs.append(b)

	# plt.plot(K, rs, color='red')
	# plt.plot(K, gs, color='green')
	# plt.plot(K, bs, color='blue')
	# plt.show()

	# # NA Variance Test
	# rs, gs, bs = [], [], []
	# NAs        = np.linspace(0.1, 0.8, 64)

	# for i, NA in enumerate(NAs):
	# 	print("%d / %d"%(i + 1, len(NAs)))
	# 	args.numerical_aperture = NA
	# 	calculator = ContrastCalculator(
	# 		refractive_data, args.thicknesses, camera, 
	# 		source, args.numerical_aperture, args.wavelength_range
	# 	)

	# 	r, g, b = calculator.getContrast(args.substrate_index)
	# 	rs.append(r)
	# 	gs.append(g)
	# 	bs.append(b)

	# plt.plot(NAs, rs, color='red')
	# plt.plot(NAs, gs, color='green')
	# plt.plot(NAs, bs, color='blue')
	# plt.show()

	calculator = ContrastCalculator(
		refractive_data, args.thicknesses, camera, 
		source, args.numerical_aperture, args.wavelength_range
	)

	# start   = time.time_ns()
	# r, g, b = calculator.getContrast(args.substrate_index)
	# end     = time.time_ns()

	# print("Calculation took: %fms"%((end - start) * 1e-6))
	# print("r = %f, g = %f, b = %f"%(r, g, b))

	# # Thickness variance test
	# rs, gs, bs = [], [], []
	# thicknesses = np.linspace(3.33e-10, 50*3.33e-10, 51)
	# calculator = ContrastCalculator(
	# 	refractive_data, args.thicknesses, camera, 
	# 	source, args.numerical_aperture, args.wavelength_range
	# )

	# for i, t in enumerate(thicknesses):
	# 	print("%d / %d"%(i + 1, len(thicknesses)))
	# 	calculator.immersion_medium = complex(1.42, 0.0)
	# 	calculator.d[0] = t

	# 	r, g, b = calculator.getContrast(args.substrate_index)
	# 	rs.append(r)
	# 	gs.append(g)
	# 	bs.append(b)

	# plt.scatter(thicknesses, rs, color="red", s=2)
	# plt.scatter(thicknesses, gs, color="green", s=2)
	# plt.scatter(thicknesses, bs, color="blue", s=2)
	# # plt.plot(thicknesses, rs, color="red")
	# # plt.plot(thicknesses, gs, color="green")
	# # plt.plot(thicknesses, bs, color="blue")
	# #plt.axhline(0.102)
	# plt.xlabel(r"$h-BN\;Thickness\;[\AA]$")
	# plt.ylabel("Optical Contrast")
	# plt.title(r"Optical Contrast of h-BN as a Function of Thickness on PDMS")
	# plt.show()

	# Monochromatic source test
	rs, gs, bs = [], [], []
	centers = np.linspace(435e-9, 655e-9, 64)

	for i, c in enumerate(centers):
		print("%d / %d"%(i + 1, len(centers)))
		calculator = ContrastCalculator(
			refractive_data, args.thicknesses, camera, 
			(','.join([str(c), str(10e-9)]), source[1]), 
			args.numerical_aperture, args.wavelength_range
		)
		calculator.immersion_medium = complex(1.40, 0.0)
		r, g, b = calculator.getContrast(args.substrate_index)
		rs.append(r)
		gs.append(g)
		bs.append(b)

	plt.plot(centers, rs, color="red")
	plt.plot(centers, gs, color="green")
	plt.plot(centers, bs, color="blue")
	#plt.axhline(0.102)
	plt.xlabel(r"$h-BN\;Thickness\;[\AA]$")
	plt.ylabel("Optical Contrast")
	plt.title(r"Optical Contrast of h-BN as a Function of Source Wavelength")
	plt.show()

	# Immersion Medium Test
	# rs, gs, bs = [], [], []
	# indices = np.linspace(1.0, 2.0, 1024)

	# for i, n in enumerate(indices):
	# 	print("%d / %d"%(i + 1, len(indices)))
	# 	calculator.immersion_medium = complex(n, 0.0)
	# 	r, g, b = calculator.getContrast(args.substrate_index)
	# 	rs.append(r)
	# 	gs.append(g)
	# 	bs.append(b)

	# plt.yscale('log')
	# plt.plot(indices, rs, color="red")
	# plt.plot(indices, gs, color="green")
	# plt.plot(indices, bs, color="blue")
	# #plt.axhline(0.102)
	# plt.axhline(0.2)
	# plt.xlabel(r"Refractive Index of Immersion Media")
	# plt.ylabel("Optical Contrast")
	# plt.title(r"Optical Contrast of h-BN as a Function of Immersion Media on PDMS")
	# plt.show()







	



