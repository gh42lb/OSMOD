#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import sounddevice as sd
import numpy as np
#import matplotlib.pyplot as plt
import threading
import wave
import sys
import random

from numpy import pi
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from scipy.io.wavfile import write, read

from modulators import ModulatorPSK 
from demodulators import DemodulatorPSK 
from queue import Queue
from datetime import datetime, timedelta

"""
MIT License

Copyright (c) 2022-2025 Lawrence Byng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class OsmodTest(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod

  def testRoutines(self, values, mode, routine_type, chunk_num, carrier_separation_override, amplitude):
    self.debug.info_message("testRoutines")
    try:

      if routine_type == 'Interpolation':
        self.testInterpolate(mode)
      elif routine_type == 'Calculate Phase Angles':
        self.testCalcPhaseAngles(mode)
      elif routine_type == 'Compare Modes':
        self.testCompareModes(mode)
      elif routine_type == 'Calculate Rotation Tables':
        self.createRotationTables(values, mode, chunk_num, carrier_separation_override, amplitude, mode, "full")
        #self.createRotationTables(values, mode, chunk_num, carrier_separation_override, amplitude, mode, "partial")
      elif routine_type == 'Calculate Constellation Shift Tables':
        self.createConstellationShiftTables(values, mode, chunk_num, carrier_separation_override, amplitude)
        """constellation_shift_values = [0.023, 0.124, 0.246, 0.312, 0.627, 0.676, 0.732, 0.886, 1.586, 2.071, 3.19, 3.641, 3.652, 3.673, 3.71, 4.034, 4.11,
         4.176, 4.392, 4.516, 4.523, 4.603, 4.626, 4.712, 4.784, 5.915, 5.94, 5.945, 6.252, 6.405, 6.539, 6.575, 6.62, 6.765, 6.811, 6.936, 6.958, 7.065,
         7.644, 7.994, 8.269, 8.414, 8.542, 8.966, 9.29, 11.164, 11.215, 11.846, 11.95, 11.961, 12.119, 12.392, 12.656, 12.798, 13.0, 13.159, 13.167, 13.203,
         13.457, 13.634, 13.933, 14.547, 14.713, 15.183, 15.556, 15.815, 15.91, 16.116, 17.502, 17.547, 18.408, 18.687, 18.944, 19.089, 19.356, 19.777, 19.824,
         20.172, 20.356, 20.391, 20.435, 20.634, 20.86, 21.085, 21.106, 21.405, 21.91, 22.103, 22.312, 22.378, 22.456, 22.466, 22.735, 22.746, 23.069, 23.084,
         23.366, 23.554, 23.587, 23.756, 23.976, 24.523, 24.879, 25.033, 25.148, 26.089, 26.327, 26.44, 26.632, 27.164, 27.506, 27.526, 27.585, 28.271, 28.445,
         28.84, 28.924, 28.955, 29.386, 29.801, 29.85, 30.164, 30.752, 31.286, 31.583, 32.093, 32.185, 32.27, 32.322, 32.336, 32.74, 33.316, 33.419, 33.56, 33.878,
         33.932, 34.182, 34.319, 34.407, 34.664, 34.751, 35.078, 35.275, 35.876, 36.202, 36.231, 36.273, 36.375, 37.113, 37.655, 37.662, 37.697, 37.792, 37.893, 37.955,
         38.327, 38.347, 38.358, 38.455, 38.484, 38.983, 39.649, 40.19, 40.364, 40.642, 40.679, 40.709, 40.787, 40.813, 40.906, 40.942, 41.234, 41.254, 41.474, 41.517,
         41.527, 41.887, 42.164, 42.62, 43.874, 43.955, 44.23, 44.66, 45.095, 45.838, 45.858, 46.304, 46.387, 46.91, 47.14, 47.162, 47.256, 47.498, 47.595, 47.867, 48.004,
         48.326, 48.375, 48.511, 48.568, 48.903, 49.025, 49.095, 49.098, 49.506, 49.903, 49.951, 50.101, 50.323, 51.237, 51.457, 51.49, 51.551, 51.773, 51.965, 52.079,
         53.628, 53.632, 54.068, 54.64, 54.873, 54.957, 54.985, 55.21, 55.465, 55.808, 55.818, 55.84, 55.914, 56.024, 56.025, 56.54, 56.744, 57.104, 57.125, 57.218, 57.445,
         57.522, 57.551, 57.766, 58.529, 58.911, 59.028, 59.153, 59.617, 60.413, 60.816, 61.032, 61.335, 61.63, 62.034, 62.296, 62.402, 62.467, 62.968, 63.001, 63.177, 63.225, 63.357, 63.477, 63.54, 63.66, 63.731, 64.203, 64.525, 64.725, 64.96, 65.145, 65.228, 65.312, 65.347, 65.831, 66.229, 66.302, 66.544, 67.574, 68.083, 68.367, 68.372, 68.519, 68.921, 69.18, 69.602, 69.644, 69.822, 70.112, 70.844, 71.279, 71.29, 71.365, 71.658, 71.795, 71.844, 71.915, 72.206, 72.336, 72.371, 72.735, 72.837, 73.146, 73.153, 73.216, 73.225, 73.654, 74.335, 74.691, 75.592, 75.618, 75.892, 76.718, 76.897, 77.147, 77.3, 77.769, 78.064, 78.13, 78.29, 78.35, 78.458, 78.555, 78.825, 79.124, 79.133, 79.272, 79.555, 79.612, 80.339, 80.435, 80.646, 80.866, 81.033, 81.053, 81.131, 81.191, 81.223, 81.764, 81.786, 81.946, 82.175, 82.221, 82.645, 82.821, 82.867, 83.275, 84.179, 84.245, 84.344, 84.662, 85.255, 85.359, 85.591, 85.923, 86.08, 86.083, 86.354, 86.538, 86.823, 87.939, 88.386, 88.389, 88.629, 88.781, 88.85, 88.86, 88.984, 89.081, 89.931, 90.114, 90.151, 90.593, 91.369, 91.448, 91.76, 92.22, 92.414, 92.644, 93.0, 93.04, 93.226, 93.391, 93.608, 94.618, 95.238, 95.538, 95.547, 96.065, 96.275, 96.571, 96.64, 97.007, 97.359, 97.383, 97.448, 97.594, 97.649, 97.812, 98.635, 98.703, 99.126, 99.264, 99.301, 99.537, 99.716, 99.935]
        """

        #self.calculateMinimumIncrement(constellation_shift_values)


    except:
      self.debug.error_message("Exception in testRoutines: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def testCompareModes(self, mode):
    self.debug.info_message("testCompareModes")

    #test_normalized_key_values = {}

    """ recursive... multi level inheritance"""
    #def processInheritFrom(mode):
    #  self.debug.info_message("processInheritFrom mode: " + str(mode))
    #  if 'inherit_from' in self.modulation_initialization_block[mode]:
    #    self.processInheritFrom(self.modulation_initialization_block[mode]['inherit_from'])
    #  for param in self.modulation_initialization_block[mode]:
    #    self.optional_param_values[param] = self.modulation_initialization_block[mode][param]

    #def getParam(mode, param_name):
    #  if param_name in self.modulation_initialization_block[mode]:
    #    param_value = self.modulation_initialization_block[mode][param_name]
    #    return param_value  
    #  else:
    #    param_value = self.optional_param_values[param_name]
    #    return param_value 

    #def processOptionalInitParams(mode):
    #  for param in self.optional_param_values:
    #    if param in self.modulation_initialization_block[mode]:
    #      self.optional_param_values[param] = self.modulation_initialization_block[mode][param]

    def compare_values(dict_a, dict_b):
      for key, value in dict_a.items():
        if value != dict_b[key]:
          self.debug.info_message("mismatch values for key: " + str(key) )
          self.debug.info_message("value a: " + str(value) )
          self.debug.info_message("value b: " + str(dict_b[key]) )


    def a_in_b(dict_a, dict_b):
      for key, value in dict_a.items():
        if key not in dict_b:
        #  self.debug.info_message("found key: " + str(key) )
        #else:
          self.debug.info_message("missing key: " + str(key) )

    def loop_all_keys(loop_mode, init_block):
      nonlocal normalized_key_values

      self.debug.info_message("iterating keys at new level" )
      for key, value in init_block[loop_mode].items():
        self.debug.info_message("key: " + str(key) + " value: " + str(value) )
        if key == "inherit_from":
          loop_all_keys(value, init_block)
        else:
          normalized_key_values[key] = value
      self.debug.info_message("completed iterating keys at this level" )

    def display_key_values(key_values):
      for key, value in key_values.items():
        self.debug.info_message("key: " + str(key) + " value: " + str(value) )

    try:
      #resetOptionalInitParamDefaults(mode)
      #processOptionalInitParams(mode)
      #processInheritFrom(mode)

      normalized_key_values = {}
      loop_all_keys(mode, self.osmod.modulation_initialization_block)
      test_normalized_key_values = normalized_key_values
      self.debug.info_message("test_normalized_key_values: " )
      display_key_values(test_normalized_key_values)

      normalized_key_values = {}
      loop_all_keys("LB28-6400-I3-B", self.osmod.prodparams.getInitializationBlock())
      prod_normalized_key_values = normalized_key_values
      self.debug.info_message("prod_normalized_key_values: " )
      display_key_values(prod_normalized_key_values)

      self.debug.info_message("test - prod" )
      a_in_b(test_normalized_key_values, prod_normalized_key_values)

      self.debug.info_message("prod - test" )
      a_in_b(prod_normalized_key_values, test_normalized_key_values)

      self.debug.info_message("prod - test - compare values" )
      compare_values(prod_normalized_key_values, test_normalized_key_values)

    except:
      self.debug.error_message("Exception in testRoutines: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calculateMinimumIncrement(self, constellation_shift_values):
    #self.debug.info_message("calculateMinimumIncrement")
    try:
      min_diff = 100.0
      for i in range(len(constellation_shift_values) - 1):
        diff = constellation_shift_values[i+1] - constellation_shift_values[i]
        #if diff < min_diff:
        min_diff = min(min_diff, diff)
        #  self.debug.info_message("constellation_shift_values[i]: " + str(constellation_shift_values[i]) )
      #self.debug.info_message("min_diff: " + str(min_diff) )

      float_min_diff = float("{:.3f}".format(min_diff))

      self.debug.info_message("minimum increment: " + str(float_min_diff) )

      discrete_jump_list = []
      for i in range(len(constellation_shift_values) ):
        float_value = float("{:.3f}".format(constellation_shift_values[i]))

        #value = ((float_value / float_min_diff) * 100) % 100
        #value = float_value / float_min_diff
        value = float_value 

        float_result = float("{:.3f}".format(value))

        discrete_jump_list.append(float_result)
        #self.debug.info_message("float_lo, float_hi: " + str(float_lo) + ", " + str(float_hi))

      discrete_jump_set = set(discrete_jump_list)
      sorted_unique_jumps = sorted(list(discrete_jump_set))
      self.debug.info_message("sorted_unique_jumps: " + str(sorted_unique_jumps))

    except:
      self.debug.error_message("Exception in calculateMinimumIncrement: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calculateInterPulsePhaseDelta(self, frequency, num_pulses):
    samples_per_wavelength = self.osmod.sample_rate / frequency
    wavelength_per_sample = 1 / samples_per_wavelength
    pulse_length_in_samples = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
    waves = wavelength_per_sample * pulse_length_in_samples  

    zero_delta_pulse_length = 1 / wavelength_per_sample
    nearest_to_required = (pulse_length_in_samples // zero_delta_pulse_length) * zero_delta_pulse_length
    self.debug.error_message("zero_delta_pulse_length: " + str(zero_delta_pulse_length))
    self.debug.error_message("nearest_to_required: " + str(nearest_to_required))

    """ nearest frequency with zero delta """
    zero_delta_wavelength_per_sample = int(waves) / pulse_length_in_samples
    zero_delta_frequency = zero_delta_wavelength_per_sample * self.osmod.sample_rate
    self.debug.error_message("zero_delta_wavelength_per_sample: " + str(zero_delta_wavelength_per_sample))
    self.debug.error_message("zero_delta_frequency: " + str(zero_delta_frequency))
    zero_delta_wavelength_per_sample = (int(waves)+1) / pulse_length_in_samples
    zero_delta_frequency = zero_delta_wavelength_per_sample * self.osmod.sample_rate
    self.debug.error_message("+1 zero_delta_wavelength_per_sample: " + str(zero_delta_wavelength_per_sample))
    self.debug.error_message("+1 zero_delta_frequency: " + str(zero_delta_frequency))
    zero_delta_wavelength_per_sample = (int(waves)-1) / pulse_length_in_samples
    zero_delta_frequency = zero_delta_wavelength_per_sample * self.osmod.sample_rate
    self.debug.error_message("-1 zero_delta_wavelength_per_sample: " + str(zero_delta_wavelength_per_sample))
    self.debug.error_message("-1 zero_delta_frequency: " + str(zero_delta_frequency))

    waves = wavelength_per_sample * pulse_length_in_samples  
    self.debug.error_message("test 0: " + str(((waves * 8) ) % 8))
    waves = wavelength_per_sample * pulse_length_in_samples * 2 
    self.debug.error_message("test 1: " + str(((waves * 8) ) % 8))
    waves = wavelength_per_sample * pulse_length_in_samples * 3 
    self.debug.error_message("test 2: " + str(((waves * 8) ) % 8))
    waves = wavelength_per_sample * pulse_length_in_samples * (((num_pulses // 3) * 3) +1)
    self.debug.error_message("test 3: " + str(((waves * 8) ) % 8))


    waves = wavelength_per_sample * pulse_length_in_samples * num_pulses
    inter_pulse_phase = (waves * 2 * np.pi) % np.pi
    self.debug.error_message("inter_pulse_phase: " + str(inter_pulse_phase))

    return ((waves * 8) + 4 ) % 8
    

  def testCalcPhaseAngles(self, mode):
    self.debug.info_message("testCalcPhaseAngles")

    try:
      self.osmod.setInitializationBlock(mode)

      center_frequency = 1400
      #frequency = self.osmod.calcCarrierFrequencies(center_frequency, carrier_separation_override)
      #self.debug.info_message("center frequency: " + str(center_frequency))
      #self.debug.info_message("carrier frequencies: " + str(frequency))
      frequency = [1300, 1315]

      pulse_length_in_samples = self.osmod.symbol_block_size / self.osmod.pulses_per_block

      def calcPhases(freq, offset_ratio):
        """ calculate the phases for the pulses at fixed pulse distance from pulse C """
        """ pulse separation is equivalent to pulse_length """
        offset_samples = pulse_length_in_samples * offset_ratio
        wavelength_in_samples   = self.osmod.sample_rate / freq
        phase_for_pulse_A = (((2*pulse_length_in_samples) - wavelength_in_samples + offset_samples) %  wavelength_in_samples ) / wavelength_in_samples
        phase_for_pulse_B = ((pulse_length_in_samples - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        phase_for_pulse_C = ((0 - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        phase_for_pulse_D = ((-pulse_length_in_samples - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        self.debug.info_message("phase_for_pulse_A: " + str(phase_for_pulse_A))
        self.debug.info_message("phase_for_pulse_B: " + str(phase_for_pulse_B))
        self.debug.info_message("phase_for_pulse_C: " + str(phase_for_pulse_C))
        self.debug.info_message("phase_for_pulse_D (A): " + str(phase_for_pulse_D))
        return phase_for_pulse_A, phase_for_pulse_B

      def calcForEachFreq(offset_ratio):
        """ calculate first frequency """
        self.debug.info_message("frequency[0]: " + str(frequency[0]))
        phase_for_pulse_A, phase_for_pulse_B = calcPhases(frequency[0], offset_ratio)
        """ calculate second frequency """
        self.debug.info_message("frequency[1]: " + str(frequency[1]))
        phase_for_pulse_A, phase_for_pulse_B = calcPhases(frequency[1], offset_ratio)


      self.debug.info_message("calculate for standing wave at sample C")
      calcForEachFreq(0)

      self.debug.info_message("calculate for standing wave at sample C + 1/4 pulse length")
      calcForEachFreq(0.25)

      self.debug.info_message("calculate for standing wave at sample C + 1/2 pulse length")
      calcForEachFreq(0.5)

      self.debug.info_message("calculate for standing wave at sample C - 1/2 pulse length")
      calcForEachFreq(-0.5)

      self.debug.info_message("calculate for standing wave at sample C + 3/4 pulse length")
      calcForEachFreq(0.75)

      self.debug.info_message("calculate for standing wave at sample C - 3/4 pulse length")
      calcForEachFreq(-0.75)

      self.debug.info_message("calculate for standing wave at sample C + 1/8 pulse length")
      calcForEachFreq(0.125)

      self.debug.info_message("calculate for standing wave at sample C + 3/8 pulse length")
      calcForEachFreq(0.375)

      self.debug.info_message("calculate for standing wave at sample C + 5/8 pulse length")
      calcForEachFreq(0.625)

      self.debug.info_message("calculate for standing wave at sample C + 7/8 pulse length")
      calcForEachFreq(0.875)


    except:
      self.debug.error_message("Exception in testCalcPhaseAngles: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """
Info: persistent_lower: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
Info: persistent_higher: [33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
  """

  def testInterpolate(self, mode):
    self.debug.info_message("testInterpolate")
    try:
      persistent_lower  = [0] * 20
      persistent_higher = [0] * 20

      persistent_lower[0]  = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
      persistent_higher[0] = [33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

      persistent_lower[1]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 60, 61, 62, 63]
      persistent_higher[1] = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58]

      persistent_lower[2]  = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 45, 47, 48, 50, 53, 55]
      persistent_higher[2] = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

      persistent_lower[3]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 56, 57, 58, 59, 60, 61, 62, 63]
      persistent_higher[3] = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 58]

      persistent_lower[4]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 27, 38, 40, 42, 63]
      persistent_higher[4] = [32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59, 60]

      persistent_lower[5]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
      persistent_higher[5] = [31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

      persistent_lower[6]  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]
      persistent_higher[6] = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

      persistent_lower[7]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 52]
      persistent_higher[7] = [1, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

      persistent_lower[8]  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54]
      persistent_higher[8] = [0, 1, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

      persistent_lower[9]  = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 49, 51, 52]
      persistent_higher[9] = [0, 1, 2, 3, 4, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

      persistent_lower[10]  = [48, 49, 50, 51, 52, 53, 54]
      persistent_higher[10] = [16, 17, 18, 19, 20, 21, 22]

      persistent_lower[11]  = [50, 52, 54, 56]
      persistent_higher[11] = [15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

      persistent_lower[12]  = [46, 47, 48, 49, 50, 51, 52, 53, 54]
      persistent_higher[12] = [16, 17, 18, 19, 20, 21, 22, 23]

      persistent_lower[13]  = [26, 27, 28, 29, 30, 31, 32, 33, 34]
      persistent_higher[13] = [0, 1, 56, 57, 58, 59, 60, 61, 62, 63]

      persistent_lower[14]  = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
      persistent_higher[14] = [0, 1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

      persistent_lower[15]  = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
      persistent_higher[15] = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

      persistent_lower[16]  = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 46, 47, 48, 49, 50]
      persistent_higher[16] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]


      test_index = 16 - self.osmod.test_counter
      self.debug.info_message("test_index: " + str(test_index))

      #test_index = 13
      interpolated_lower, interpolated_higher, shift_amount = self.osmod.interpolator.interpolatePulseTrain([persistent_lower[test_index], persistent_higher[test_index]])

    except:
      self.debug.error_message("Exception in testInterpolate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def testInterpolate2(self, mode):
    self.debug.info_message("testInterpolate")
    try:
      """ initialize the block"""
      self.osmod.setInitializationBlock(mode)
      max_occurrences_lower = [4, 2, 9, 10, 1, 0, 3, 26, 5, 27, 32, 22, 19, 63, 20, 45, 28, 11, 36, 23, 33, 30, 38, 21, 54, 25, 8, 49, 18, 39, 55, 6]
      max_occurrences_higher = [49, 23, 50, 52, 51, 44, 24, 25, 45, 28, 29, 60, 59, 38, 48, 40, 39, 63, 30, 61, 0, 58, 35, 43, 21, 12, 13, 22, 27, 2, 41, 53]

      in_both_lists = []
      for i in max_occurrences_lower:
        if i in max_occurrences_higher and i not in in_both_lists:
          in_both_lists.append(i)
      for i in max_occurrences_higher:
        if i in max_occurrences_lower and i not in in_both_lists:
          in_both_lists.append(i)
      self.debug.info_message("in_both_lists: " + str(in_both_lists))

      for i in in_both_lists:
        max_occurrences_lower.remove(i)
        max_occurrences_higher.remove(i)


      interpolated_lower  = self.osmod.interpolator.interpolate_contiguous_items(max_occurrences_lower)
      interpolated_higher = self.osmod.interpolator.interpolate_contiguous_items(max_occurrences_higher)
      self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))
      sorted_lower  = self.osmod.interpolator.sort_interpolated(interpolated_lower)
      sorted_higher = self.osmod.interpolator.sort_interpolated(interpolated_higher)
      self.debug.info_message("sorted_lower: " + str(sorted_lower))
      self.debug.info_message("sorted_higher: " + str(sorted_higher))

      interpolated_lower  = sorted_lower
      interpolated_higher = sorted_higher
      """ if either of the inetrpolated lists is greater than the other list, fill out the other interpolated list"""
      if len(interpolated_lower) > len(interpolated_higher):
        for i in range(interpolated_lower[0],interpolated_lower[-1]):
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_higher:
            interpolated_higher.append(partner_index)
      elif len(interpolated_higher) > len(interpolated_lower):
        for i in range(interpolated_higher[0],interpolated_higher[-1]):
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_lower:
            interpolated_lower.append(partner_index)

      self.debug.info_message("re-processed lower: "  + str(interpolated_lower))
      self.debug.info_message("re-processed higher: " + str(interpolated_higher))


    except:
      self.debug.error_message("Exception in testInterpolate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def testRoutine1(self, mode, form_gui):
    self.debug.info_message("testRoutine1")
    self.window = form_gui.window
    self.test_double_carrier_8psk(mode)

  def testRoutine2(self, mode, form_gui, values, noise_mode, text_num, chunk_num, carrier_separation_override, amplitude):
    self.debug.info_message("testRoutine2")
    self.window = form_gui.window
    self.values = values
    self.test1(mode, noise_mode, text_num, chunk_num, carrier_separation_override, amplitude)



  #def createPhaseRotationTables(self, values, mode, chunk_num, carrier_separation_override, amplitude, tablename, full_partial):
  #  self.debug.info_message("createPhaseRotationTables")




  def createRotationTables(self, values, mode, chunk_num, carrier_separation_override, amplitude, tablename, full_partial):
    self.debug.info_message("createRotationTables")

    try:
      self.osmod.startTimer('init')
      pulse_length   = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      fine_tune_adjust    = [0] * 2
      fine_tune_adjust[0] = 0
      fine_tune_adjust[1] = 0

      """ initialize the block"""
      self.osmod.setInitializationBlock(mode)

      self.osmod.extrapolate = 'no'

      """ figure out the carrier frequencies"""
      center_frequency = values['slider_frequency']

      self.debug.info_message("center_frequency: " + str(center_frequency))
      self.debug.info_message("carrier_separation_override: " + str(carrier_separation_override))

      frequency = self.osmod.calcCarrierFrequencies(center_frequency, carrier_separation_override)

      """ convert text to bits"""
      text = 'aaaaaaaa' + " peter piper picked a peck of pickled peppercorn "
      bit_groups, sent_bitstring, binary_array_pre_fec = self.osmod.text_encoder(text)
      data2 = self.osmod.modulation_object.modulate(frequency, bit_groups)

      self.osmod.modulation_object.writeFileWav(mode + ".wav", data2)
      audio_array = self.osmod.modulation_object.readFileWav(mode + ".wav")
      noise_free_signal = audio_array*0.00001 * float(amplitude)

      how_many_blocks = max(1, int(len(text) // int(chunk_num)))
      audio_block = np.array_split( noise_free_signal , how_many_blocks, axis=0)

      """ filter the output signal """
      tx_filter_params = self.osmod.tx_filter
      audio_block = self.osmod.modulation_object.apply_filter(audio_block, tx_filter_params, center_frequency)


      base_signal = audio_block[0].copy()
      half = int(self.osmod.pulses_per_block / 2)

      #rotation_dict = {}

      rotation_dict = self.osmod.opd.readRotationTablesFromFile(tablename)


      if full_partial == "full":
        loop_start = 3
        loop__end = half + 1
      elif full_partial == "partial":
        loop_start = half - 3   #18 #half - 6
        loop__end  = half + 1
        #self.osmod.rotation_increments = 100000

      #for pulse_train_length in range(3,33):
      for pulse_train_length in range(loop_start,loop__end):

        test_limit = half - pulse_train_length

        interpolated_lower = []
        interpolated_higher = []
        for interp in range(0, pulse_train_length):
          interpolated_lower.append(interp)
          interpolated_higher.append(interp + half)

        self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
        self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

        #rotation_dict[pulse_train_length] = []
        rotation_dict[str(int(center_frequency)) + "_" + str(pulse_train_length)] = []

        for offset in range(0, test_limit + 1):
          test_signal = (base_signal[offset * pulse_length:]).copy()

          pulse_start_index = 0
          block_count = 0

          """ ignore compression / expansion """
          num_full_blocks = int(len(test_signal) // self.osmod.symbol_block_size)
          total_block_residual_lower  = [0] * num_full_blocks
          total_block_residual_higher = [0] * num_full_blocks
          residuals = [total_block_residual_lower, total_block_residual_higher]

          fft_filtered_lower, fft_filtered_higher = self.osmod.demodulation_object.receive_pre_filters_filter_wave(pulse_start_index, test_signal, frequency)

          audio_array1 = (fft_filtered_lower.real + fft_filtered_lower.imag ) / 2
          audio_array2 = (fft_filtered_higher.real + fft_filtered_higher.imag) / 2

          self.osmod.demodulation_object.downconvert_I3_RelExp(pulse_start_index, [fft_filtered_lower, fft_filtered_higher], frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)

          where1_pulse_a = int(np.median(interpolated_lower))
          where1_pulse_b = int(np.min(interpolated_lower))
          where1_pulse_c = int(np.max(interpolated_lower))
          where2_pulse_a = int(np.median(interpolated_higher))
          where2_pulse_b = int(np.min(interpolated_higher))
          where2_pulse_c = int(np.max(interpolated_higher))

          self.osmod.demodulation_object.receive_pre_filters_average_data_intra_triple(pulse_start_index, fft_filtered_lower, fft_filtered_higher, None, interpolated_lower, interpolated_higher, fine_tune_adjust)
          intlist_lower, decoded_bitstring_1, decoded_intvalues1, intra_triple_charts = self.osmod.demodulation_object.extractPhaseValuesIntraTriple(fft_filtered_lower, pulse_start_index, frequency[0], pulse_length, where1_pulse_a * pulse_length, where1_pulse_b * pulse_length, where1_pulse_c * pulse_length, fine_tune_adjust, 0, residuals)
          intlist_higher, decoded_bitstring_2, decoded_intvalues2, intra_triple_charts = self.osmod.demodulation_object.extractPhaseValuesIntraTriple(fft_filtered_higher, pulse_start_index, frequency[1], pulse_length, where2_pulse_a * pulse_length, where2_pulse_b * pulse_length, where2_pulse_c * pulse_length, fine_tune_adjust, 1, residuals)

          rotation_lo = self.osmod.detector.rotation_angles[0]
          rotation_hi = self.osmod.detector.rotation_angles[1]

          rotation_dict[str(int(center_frequency)) + "_" + str(pulse_train_length)].append((rotation_lo, rotation_hi))

      self.debug.info_message("rotation_dict: " + str(rotation_dict))


      self.osmod.opd.writeRotationTablesToFile(tablename, rotation_dict)

      #self.debug.info_message("rotation_lo: " + str(rotation_lo))
      #self.debug.info_message("rotation_hi: " + str(rotation_hi))

    except:
      self.debug.error_message("Exception in createRotationTables: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))




  def createConstellationShiftTables(self, values, mode, chunk_num, carrier_separation_override, amplitude):
    self.debug.info_message("createConstellationShiftTables")

    try:
      half = int(self.osmod.pulses_per_block / 2)
      #constellation_shift_dict = {}

      """
      self.createRotationTables(values, mode, chunk_num, carrier_separation_override, amplitude, "constellation_test", "partial")
      """
      current_rotation_table = self.osmod.opd.readRotationTablesFromFile("constellation_test")

      #current_rotation_table = self.osmod.rotation_tables
      if current_rotation_table != None:
        #for pulse_train_length in range(3,half + 1):
        for key, value in current_rotation_table.items():
          discrete_jump_list = []

          for item_count in range(0, len(value)):
            #self.debug.info_message("value item: " + str(value[item_count]))
            item_lo = value[item_count][0]
            item_hi = value[item_count][1]
            shift_lo = ((item_lo / (np.pi / 4)) * 100.0) % 100
            shift_hi = ((item_hi / (np.pi / 4)) * 100.0) % 100
            #self.debug.info_message("shift_lo, shift_hi: " + str(shift_lo) + ", " + str(shift_hi))

            float_lo = float("{:.3f}".format(shift_lo))
            float_hi = float("{:.3f}".format(shift_hi))

            discrete_jump_list.append(float_lo)
            discrete_jump_list.append(float_hi)

          discrete_jump_set = set(discrete_jump_list)
          sorted_unique_jumps = sorted(list(discrete_jump_set))
          self.debug.info_message("Scale " + str(key) + " :- ")

          self.calculateMinimumIncrement(sorted_unique_jumps)
          #self.debug.info_message("float_lo, float_hi: " + str(float_lo) + ", " + str(float_hi))



        #self.debug.info_message("sorted_unique_jumps: " + str(sorted_unique_jumps))


    except:
      self.debug.error_message("Exception in createConstellationShiftTables: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  #test = [0.023, 0.124, 0.246, 0.312, 0.627, 0.676, 0.732, 0.886, 1.586, 2.071, 3.19, 3.641, 3.652, 3.673, 3.71, 4.034, 4.11, 4.176, 4.392, 4.516, 4.523, 4.603, 4.626, 4.712, 4.784, 5.915, 5.94, 5.945, 6.252, 6.405, 6.539, 6.575, 6.62, 6.765, 6.811, 6.936, 6.958, 7.065, 7.644, 7.994, 8.269, 8.414, 8.542, 8.966, 9.29, 11.164, 11.215, 11.846, 11.95, 11.961, 12.119, 12.392, 12.656, 12.798, 13.0, 13.159, 13.167, 13.203, 13.457, 13.634, 13.933, 14.547, 14.713, 15.183, 15.556, 15.815, 15.91, 16.116, 17.502, 17.547, 18.408, 18.687, 18.944, 19.089, 19.356, 19.777, 19.824, 20.172, 20.356, 20.391, 20.435, 20.634, 20.86, 21.085, 21.106, 21.405, 21.91, 22.103, 22.312, 22.378, 22.456, 22.466, 22.735, 22.746, 23.069, 23.084, 23.366, 23.554, 23.587, 23.756, 23.976, 24.523, 24.879, 25.033, 25.148, 26.089, 26.327, 26.44, 26.632, 27.164, 27.506, 27.526, 27.585, 28.271, 28.445, 28.84, 28.924, 28.955, 29.386, 29.801, 29.85, 30.164, 30.752, 31.286, 31.583, 32.093, 32.185, 32.27, 32.322, 32.336, 32.74, 33.316, 33.419, 33.56, 33.878, 33.932, 34.182, 34.319, 34.407, 34.664, 34.751, 35.078, 35.275, 35.876, 36.202, 36.231, 36.273, 36.375, 37.113, 37.655, 37.662, 37.697, 37.792, 37.893, 37.955, 38.327, 38.347, 38.358, 38.455, 38.484, 38.983, 39.649, 40.19, 40.364, 40.642, 40.679, 40.709, 40.787, 40.813, 40.906, 40.942, 41.234, 41.254, 41.474, 41.517, 41.527, 41.887, 42.164, 42.62, 43.874, 43.955, 44.23, 44.66, 45.095, 45.838, 45.858, 46.304, 46.387, 46.91, 47.14, 47.162, 47.256, 47.498, 47.595, 47.867, 48.004, 48.326, 48.375, 48.511, 48.568, 48.903, 49.025, 49.095, 49.098, 49.506, 49.903, 49.951, 50.101, 50.323, 51.237, 51.457, 51.49, 51.551, 51.773, 51.965, 52.079, 53.628, 53.632, 54.068, 54.64, 54.873, 54.957, 54.985, 55.21, 55.465, 55.808, 55.818, 55.84, 55.914, 56.024, 56.025, 56.54, 56.744, 57.104, 57.125, 57.218, 57.445, 57.522, 57.551, 57.766, 58.529, 58.911, 59.028, 59.153, 59.617, 60.413, 60.816, 61.032, 61.335, 61.63, 62.034, 62.296, 62.402, 62.467, 62.968, 63.001, 63.177, 63.225, 63.357, 63.477, 63.54, 63.66, 63.731, 64.203, 64.525, 64.725, 64.96, 65.145, 65.228, 65.312, 65.347, 65.831, 66.229, 66.302, 66.544, 67.574, 68.083, 68.367, 68.372, 68.519, 68.921, 69.18, 69.602, 69.644, 69.822, 70.112, 70.844, 71.279, 71.29, 71.365, 71.658, 71.795, 71.844, 71.915, 72.206, 72.336, 72.371, 72.735, 72.837, 73.146, 73.153, 73.216, 73.225, 73.654, 74.335, 74.691, 75.592, 75.618, 75.892, 76.718, 76.897, 77.147, 77.3, 77.769, 78.064, 78.13, 78.29, 78.35, 78.458, 78.555, 78.825, 79.124, 79.133, 79.272, 79.555, 79.612, 80.339, 80.435, 80.646, 80.866, 81.033, 81.053, 81.131, 81.191, 81.223, 81.764, 81.786, 81.946, 82.175, 82.221, 82.645, 82.821, 82.867, 83.275, 84.179, 84.245, 84.344, 84.662, 85.255, 85.359, 85.591, 85.923, 86.08, 86.083, 86.354, 86.538, 86.823, 87.939, 88.386, 88.389, 88.629, 88.781, 88.85, 88.86, 88.984, 89.081, 89.931, 90.114, 90.151, 90.593, 91.369, 91.448, 91.76, 92.22, 92.414, 92.644, 93.0, 93.04, 93.226, 93.391, 93.608, 94.618, 95.238, 95.538, 95.547, 96.065, 96.275, 96.571, 96.64, 97.007, 97.359, 97.383, 97.448, 97.594, 97.649, 97.812, 98.635, 98.703, 99.126, 99.264, 99.301, 99.537, 99.716, 99.935]



  def test1(self, mode, noise_mode, text_num, chunk_num, carrier_separation_override, amplitude):
    self.debug.info_message("test1")

    try:

      self.osmod.startTimer('init')

      """ initialize the block"""
      self.osmod.setInitializationBlock(mode)

      """ figure out the carrier frequencies"""
      center_frequency = self.values['slider_frequency']
      #center_frequency = 1400
      frequency = self.osmod.calcCarrierFrequencies(center_frequency, carrier_separation_override)
      self.debug.info_message("center frequency: " + str(center_frequency))
      self.debug.info_message("carrier frequencies: " + str(frequency))

      """ convert text to bits"""
      text_examples = [0] * 16
      text_examples[0]  = " cq wh6ggo "
      text_examples[1]  = " cqcqcqcqcqcq wh6ggo "
      text_examples[2]  = " cqcqcqcqcqcqcqcqcqcqcq wh6ggo "
      text_examples[3]  = " peter piper picked a peck of pickled peppercorn "
      text_examples[4]  = "jack be nimble jack be quick jack jump over the candlestick"
      text_examples[5]  = "row row row your boat gently down the stream merrily merrily merrily merrily life is but a dream"
      text_examples[6]  = "hickory dickory dock the mouse ran up the clock the clock struch one the mouse ran down hickory dicory dock"
      text_examples[7]  = "its raining its pouring the old man is snoring he bumped his head and went to bed and he couldnt get up in the morning"
      text_examples[8]  = "jack and jill went up the hill to fetch a pail of water jack fell down and broke his crown and jill came tumbling after"
      text_examples[9]  = "humpty dumpty dat on a wall humpty dumpty had a great fall all the kings forses and all the kings men coudnt put humpty together again"
      text_examples[10]  = "a wise old owl sat in an oak the more he heard the less he spoke the less he spoke the more he heard why arent we all like that wise old bird"
      text_examples[11]  = "hey diddle diddle the cat and the fiddle the cow jumped over the moon the little dog laughed to see such fun and the dish ran away with the spoon"
      text_examples[12]  = "baa baa black sheep have you any wool yes sir yes sir three bags full one for the master and one for the dame and one for the little boy who lives down the lane"
      text_examples[13] = "twinkle twinkle little bat how i wonder what youre at up above the world you fly like a tea tray in the sky twinkle twinkle little bat how i wonder what youre at"
      text_examples[14] = "i can read on a boat i can read with a goat i can read on a train i can read in the rain i can read with a fox i can read in a box i can read with a mouse i can read in a house i can read here or there i can read anywhere"
      text_examples[15] = "the queen of hearts she made some tarts all on a summers day the knave of hearts he stole the tarts and took them clean away the king of hearts called for the tarts and beat the knave full sore the knave of hearts brought back the tarts and vowed hed steal no more"

      """ add start sequence character and trailing space """
      if self.osmod.start_seq == '2_of_3':
        text = 'aaa' + text_examples[int(text_num)] + ' '
      elif self.osmod.start_seq == '2_of_4' or self.osmod.start_seq == '3_of_4':
        text = 'aaaa' + text_examples[int(text_num)] + ' '
      elif self.osmod.start_seq == '2_of_5' or self.osmod.start_seq == '3_of_5' or self.osmod.start_seq == '4_of_5':
        text = 'aaaaa' + text_examples[int(text_num)] + ' '
      elif self.osmod.start_seq == '2_of_6':
        text = 'aaaaaa' + text_examples[int(text_num)] + ' '
      elif self.osmod.start_seq == '2_of_7':
        text = 'aaaaaaa' + text_examples[int(text_num)] + ' '
      elif self.osmod.start_seq == '2_of_8':
        text = 'aaaaaaaa' + text_examples[int(text_num)] + ' '
      else:
        text = 'aaa' + text_examples[int(text_num)] + ' '


      #text = text_examples[int(text_num)] + ' '

      #text = 'gngngngngngngngngngngngngngn '
      #text = 'aaaaajjjjjbbbbbpppppcccccxxxxx'
      #text = 'aaaaajjjjjaaaaajjjjjaaaaajjjjjaaaaajjjjj'
      #text = 'qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'
      #text = 'aaaaaaabcdefghijklmnopqrstuvwxyz123456789'
      #text = 'aabbccddeeffgghhiijjkkll'
      #text = 'abcdefghabcdefghabcdefghabcdefgh '
      #text = 'aaabbccddeeffgghhiijjkkllmmnnoo'
      #text = 'aaaaaaaaaaaaaaaaaaaaaa'
      #text = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
      #text = 'aaaaaaaaaaaabbbbbbbbbbbbcccccccccccc'
      #text = 'aaaaaaaaaayyyyyyyyyykkkkkkkk'
 
      self.debug.info_message("encoding text: " + str(text))

      bit_groups, sent_bitstring, binary_array_pre_fec = self.osmod.text_encoder(text)
      data2 = self.osmod.modulation_object.modulate(frequency, bit_groups)

      """ filter the output signal """
      tx_filter_params = self.osmod.tx_filter
      data2 = self.osmod.modulation_object.apply_filter(data2, tx_filter_params, center_frequency)

      """
      signal_width = 48 # 50 actually works at 1.5 AWGN to 0.06 BER
      for _ in range(5):
        sig1 = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(data2, center_frequency + signal_width/2)
        data2 = sig1
      for _ in range(5):
        sig2 = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(data2, center_frequency - signal_width/2)
        data2 = sig2
      """

      """ write to file """
      self.debug.info_message("size of signal data: " + str(len(data2)))
      #if len(data2) < 10000000:
      self.osmod.modulation_object.writeFileWav(mode + ".wav", data2)

      """ read file """
      use_audio_sample = self.osmod.form_gui.window['cb_test_routine_use_audio_sample'].get()
      if use_audio_sample:
        audio_sample_name = self.osmod.form_gui.window['combo_audio_sample_name'].get()
        audio_array = self.osmod.modulation_object.readFileWav(audio_sample_name) #v* 0.00000001
      else:
        audio_array = self.osmod.modulation_object.readFileWav(mode + ".wav") #'8psktest11.wav')

      self.debug.info_message("audio data type: " + str(audio_array.dtype))
      self.debug.info_message("demodulating")
      total_audio_length = len(audio_array)

      """ add noise for testing..."""
      #noise_free_signal = audio_array*0.00001
      noise_free_signal = audio_array*0.00001 * float(amplitude)   #* 0.7


      noise_free_signal = self.osmod.modulation_object.adjustFrequencyShiftAndDopplerShift(noise_free_signal, self.values, center_frequency)


      frequency_test_lower  = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1380, 1385)
      frequency_test_higher = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1414, 1424)
      """ freq diff for 37 showing as 36.056 from fft values. changes depending on mode"""
      difference = ((frequency_test_higher - frequency_test_lower) - (self.osmod.resample_params[2] - self.osmod.resample_params[1])) * 10000
      self.debug.info_message("difference: " + str(difference))

      #partial_result = difference + 9439.0 # this value changes depending on mode
      partial_result = difference # - 0.344827586 # this value changes depending on mode
      self.debug.info_message("partial_result: " + str(partial_result))
      calculated_freq_offset = partial_result / 250    #255  #271
      self.debug.info_message("calculated_freq_offset: " + str(calculated_freq_offset))

      self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1180, 1185)
      self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1214, 1224)

      """ 9439 is at 0 freq diff , 9466 is at 0.1 freq diff, 9490 is at 0.2 freq diff"""
      """ (difference - 9439) / 255 """

      """
      frequency_test = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1380, 1385)
      adjust_ratio = frequency_test / 1382.5
      self.debug.info_message("adjust_ratio: " + str(adjust_ratio))
      #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000, 8000 * adjust_ratio)
      noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000 / adjust_ratio, 8000)
      self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, 1380, 1385)
      """


      self.debug.info_message("noise mode: " + str(noise_mode))
      value = float(noise_mode)

      audio_array = noise_free_signal
      if self.window['cb_enable_awgn'].get():
        audio_array = self.osmod.modulation_object.addAWGN(audio_array, value, frequency)
      if self.window['cb_enable_timing_noise'].get():
        audio_array = self.osmod.modulation_object.addTimingNoise(audio_array)
      if self.window['cb_enable_phase_noise'].get():
        audio_array = self.osmod.modulation_object.addPhaseNoise2(audio_array)

      self.debug.info_message("size of noise data: " + str(len(audio_array)))
      #if len(audio_array) < 10000000:
      self.osmod.modulation_object.writeFileWav(mode + "_with_noise.wav", audio_array)

      audio_array_with_unfiltered_noise = audio_array.copy()


      """ receive section """

      """ filter the input signal """
      rx_filter_params = self.osmod.rx_filter
      audio_array = self.osmod.modulation_object.apply_filter(audio_array, rx_filter_params, center_frequency)

      """ reset the remainder"""
      self.osmod.demod_2fsk8psk.remainder = np.array([])

      """ split into blocks for testing..."""
      #self.osmod.startTimer('test12_demod_timer')

      rcvd_bitstring_1 = []
      rcvd_bitstring_2 = []

      how_many_blocks = max(1, int(len(text) // int(chunk_num)))


      simulate_random_phase_shift = self.osmod.form_gui.window['cb_simulatephaseshift'].get()
      simulate_random_phase_shift_add = self.osmod.form_gui.window['cb_simulatephaseshift_add'].get()
      simulate_random_phase_shift_remove = self.osmod.form_gui.window['cb_simulatephaseshift_remove'].get()
      simulate_mid_signal_phase_delay_thirds = self.osmod.form_gui.window['cb_simulatemidsignalphasedelatthirds'].get()

      mid_signal_phase_shift_amount = int(self.osmod.form_gui.window['in_midsignalphaseshiftamount'].get())

      third = int(len(audio_array) / 3)
      first_delay_location  = third 
      second_delay_location = third * 2

      #delay_samples = 20
      delay_samples = mid_signal_phase_shift_amount #3
      phase_delay = np.zeros(delay_samples, dtype = audio_array.dtype)

      if simulate_random_phase_shift:
        self.debug.info_message("simulating random phase shift")

        if simulate_mid_signal_phase_delay_thirds:
          self.debug.info_message("mid signal phase shift")
          first_part  = audio_array[0:first_delay_location]
          second_part = audio_array[first_delay_location:second_delay_location]
          third_part  = audio_array[second_delay_location:]
          new_array = audio_array.copy()
          new_array[0:first_delay_location] = first_part
          new_array[first_delay_location:first_delay_location+delay_samples]      = phase_delay
          new_array[first_delay_location+delay_samples:second_delay_location+delay_samples]   = second_part
          new_array[second_delay_location+delay_samples:second_delay_location+(delay_samples * 2)] = phase_delay
          new_array[second_delay_location+(delay_samples * 2):]   = third_part[0:(delay_samples * -2)]
          audio_array = new_array

        random_start_index1 = 0
        if simulate_random_phase_shift_remove:
          random_start_index1 = random.randint(0,1000)
          self.debug.info_message("additive random phase shift")
          audio_array = audio_array[random_start_index1:]

        random_start_index2 = 0
        if simulate_random_phase_shift_add:
          random_start_index2 = random.randint(0,1000)
          self.debug.info_message("subtractive random phase shift")
          audio_array = np.concatenate((np.zeros(random_start_index2, dtype = audio_array.dtype), audio_array))

        pulse_length = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
        self.debug.info_message("random offset modulo pulse_length: " + str((random_start_index2 - random_start_index1 + pulse_length) % pulse_length) )
        self.debug.info_message("random offset modulo 3*pulse_length: " + str((random_start_index2 - random_start_index1 + (3*pulse_length)) % (3*pulse_length)) ) 
 

      audio_block = np.array_split( audio_array , how_many_blocks, axis=0)
      for block_count in range (how_many_blocks):
        self.debug.info_message("num_divisor: " + str(how_many_blocks))
        self.debug.info_message("calling demodulate_2fsk_8psk. block count: " + str(block_count))
        decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = self.osmod.demodulation_object.demodulate_2fsk_8psk(audio_block[block_count], frequency)

        self.debug.info_message("appending bitstrings")
        rcvd_bitstring_1.append(decoded_bitstring_1)
        rcvd_bitstring_2.append(decoded_bitstring_2)

      self.debug.info_message("complete")

      #self.debug.info_message("elapsed time: " + str(self.osmod.getDuration('test12_demod_timer')))
      self.debug.info_message("text len: " + str(len(text)))

      self.debug.info_message("total_audio_length: " + str(total_audio_length))
      total_seconds = total_audio_length / self.osmod.sample_rate
      self.debug.info_message("total_seconds: " + str(total_seconds))
      characters_per_second = len(text) / total_seconds
      self.debug.info_message("characters per second: " + str(characters_per_second))
      bits_per_second = characters_per_second * 6
      self.debug.info_message("bits per second (baud): " + str(bits_per_second))

      count = 0
      error = 0

      if self.osmod.msg_type == ocn.MSGTYPE_VARIABLE_LENGTH:
        msg_start = 0
        msg_len   = len(sent_bitstring[0])
        total_num_bits = msg_len * 6

      elif self.osmod.msg_type == ocn.MSGTYPE_FIXED_LENGTH and self.osmod.FEC != ocn.FEC_NONE:
        msg_start = int(self.osmod.msg_sections[0]) + int(self.osmod.msg_sections[1])
        #msg_end   = len(rcvd_bitstring_1)   #msg_start + min(len(rcvd_bitstring_1[0])-msg_start, int(self.osmod.msg_sections[2]))
        msg_len   = min(len(binary_array_pre_fec // 6) - msg_start, self.osmod.msg_sections[2])
        total_num_bits = msg_len * 6

      elif self.osmod.msg_type == ocn.MSGTYPE_FIXED_LENGTH and self.osmod.FEC == ocn.FEC_NONE:
        msg_start = 0
        msg_len   = len(sent_bitstring[0])
        total_num_bits = msg_len * 6



      #self.debug.info_message("len(rcvd_bitstring_1[0]): " + str(len(rcvd_bitstring_1[0])))
      self.debug.info_message("msg_start: " + str(msg_start))
      #self.debug.info_message("msg_end: " + str(msg_end))
      self.debug.info_message("msg_len: " + str(msg_len))


      if self.osmod.FEC == ocn.FEC_NONE:
        for block in range(0, len(rcvd_bitstring_1)):
          for bits1, bits2 in zip(rcvd_bitstring_1[block], rcvd_bitstring_2[block]):
            if count >= msg_start and count < msg_start + msg_len:
              if bits1[0:1] !=  sent_bitstring[0][count][0:1]:
                error = error + 1
              if bits1[1:2] !=  sent_bitstring[0][count][1:2]:
                error = error + 1
              if bits1[2:3] !=  sent_bitstring[0][count][2:3]:
                error = error + 1
              if bits2[0:1] !=  sent_bitstring[1][count][0:1]:
                error = error + 1
              if bits2[1:2] !=  sent_bitstring[1][count][1:2]:
                error = error + 1
              if bits2[2:3] !=  sent_bitstring[1][count][2:3]:
                error = error + 1
            #self.debug.info_message("error: " + str(error))
            count = count + 1
      else:
        self.debug.info_message("binary_array_pre_fec: " + str(binary_array_pre_fec))
        self.debug.info_message("binary_array_post_fec: " + str(binary_array_post_fec))

        for bits1, bits2 in zip(binary_array_pre_fec, binary_array_post_fec):
          if count // 6 >= msg_start and count // 6 < msg_start + msg_len:
            if bits1 !=  bits2:
              error = error + 1
          count = count + 1


      self.debug.info_message("total error: " + str(error))
      self.debug.info_message("num bits: " + str(len(sent_bitstring[0])*6))
      ber = error / (msg_len * 6)
      """ last char is padding so can be ignored """
      #ber = error/(len(rcvd_bitstring_1)*6)

      self.debug.info_message("BER: " + str(ber))
      #total_num_bits = len(sent_bitstring[0])*6

      ebn0_db, ebn0, SNR_equiv_db = self.osmod.mod_2fsk8psk.calculate_EbN0(audio_array_with_unfiltered_noise, frequency, total_num_bits, bits_per_second, noise_free_signal, center_frequency) 
      self.osmod.form_gui.window['text_ber_value'].update("BER: " + str(ber))
      self.osmod.form_gui.window['text_ebn0_value'].update("Eb/N0: " + str(ebn0))
      self.osmod.form_gui.window['text_ebn0db_value'].update("Eb/N0 (dB): " + str(ebn0_db))
      self.osmod.form_gui.window['text_snr_value'].update("SNR dB: " + str(SNR_equiv_db))

      self.osmod.getSummary()

      if self.window['cb_enable_awgn'].get():
        noise_factor = float(noise_mode)
      else:
        noise_factor = 0.0

      chunk_size = self.window['combo_chunk_options'].get()


      standingwave_pattern = "---"
      standingwave_location = 0.0
      offsets_override_checked = self.osmod.form_gui.window['cb_override_standingwaveoffsets'].get()
      if offsets_override_checked:
        standingwave_pattern = self.osmod.form_gui.window['combo_standingwave_pattern'].get()
        standingwave_location = float(self.osmod.form_gui.window['in_standingwavelocation'].get())
      elif self.osmod.i3_offsets_type == ocn.OFFSETS_MANUAL:
        standingwave_pattern  = self.osmod.i3_parameters[3]
        standingwave_location = float(self.osmod.i3_parameters[4])

      #extract_type = ''
      #extract_selection = self.osmod.form_gui.window['combo_intra_extract_type'].get() 
      #if extract_selection == 'Type 5':
      #  extract_type = 'gaussian'
      #extract_type = self.osmod.I3_extract

      
      #"""
      sw_pattern_override_checked = self.osmod.form_gui.window['cb_override_standingwavepattern'].get()
      if sw_pattern_override_checked:
        preset_sw_pattern = self.osmod.form_gui.window['combo_selectstandingwavepattern'].get()
      else:
        preset_sw_pattern = self.osmod.i3_offsets_type

      chunk_size = self.osmod.form_gui.window['combo_chunk_options'].get() 

      rrc_alpha_override_checked = self.osmod.form_gui.window['cb_override_rrc_alpha'].get()
      if rrc_alpha_override_checked:
        rrc_alpha = self.osmod.form_gui.window['in_rrc_alpha'].get()
      else:
        rrc_alpha = self.osmod.parameters[1]

      rrc_t_override_checked = self.osmod.form_gui.window['cb_override_rrc_t'].get()
      if rrc_t_override_checked:
        rrc_t = self.osmod.form_gui.window['in_rrc_t'].get()
      else:
        rrc_t = self.osmod.parameters[2]

      costas_override_checked = self.osmod.form_gui.window['cb_override_costasloop'].get()
      if costas_override_checked:
        costas_damping = self.osmod.form_gui.window['in_costasloop_dampingfactor'].get()
        costas_loop_bandwidth = self.osmod.form_gui.window['in_costasloop_loopbandwidth'].get()
        costas_k1 = self.osmod.form_gui.window['in_costasloop_K1'].get()
        costas_k2 = self.osmod.form_gui.window['in_costasloop_K2'].get()
      else:
        costas_damping        = self.osmod.parameters[6]
        costas_loop_bandwidth = self.osmod.parameters[7]
        costas_k1             = self.osmod.parameters[8]
        costas_k2             = self.osmod.parameters[9]

      extract_type = self.osmod.form_gui.window['combo_intra_extract_type'].get() 


      pulse_train_sigma = 5.0
      override_pulse_train_sigma = self.osmod.form_gui.window['cb_overridepulsetrainsigma'].get()
      if override_pulse_train_sigma:
        pulse_train_sigma = float(self.osmod.form_gui.window['in_pulsetrainsigma'].get())

      downconvert_shift = self.osmod.downconvert_shift
      override_downconvert_shift = self.osmod.form_gui.window['cb_overridedownconvertshift'].get()
      if override_downconvert_shift:
          downconvert_shift = float(self.osmod.form_gui.window['in_downconvertshift'].get())


      fdmsep = self.osmod.FDM_parameters[1]
      fdmsep_override_checked = self.osmod.form_gui.window['cb_overridefdmseparation'].get()
      if fdmsep_override_checked:
        fdmsep = float(self.osmod.form_gui.window['in_fdmseparation'].get())


      polynomial_override_checked = self.osmod.form_gui.window['cb_overridegeneratorpolynomials'].get()
      if polynomial_override_checked:
        gp1     = self.osmod.form_gui.window['in_fecgeneratorpolynomial1'].get()
        gp2     = self.osmod.form_gui.window['in_fecgeneratorpolynomial2'].get()
        gpdepth = self.osmod.form_gui.window['in_fecgeneratorpolynomialdepth'].get()

      else:
        gp1 = 0 #self.osmod.fec_params[1]
        gp2 = 0 #self.osmod.fec_params[2]
        gpdepth = 0


      detector_threshold_1 = 1
      detector_threshold_2 = 1
      basebandconv_freq_delta = 1
      #"""

      rotation_lo = self.osmod.detector.rotation_angles[0]
      rotation_hi = self.osmod.detector.rotation_angles[1]

      disposition = self.osmod.detector.disposition

      pulse_train_length = self.osmod.detector.pulse_train_length

      csv_data = [mode, ebn0_db, SNR_equiv_db, ber, characters_per_second, bits_per_second, noise_factor, standingwave_pattern, standingwave_location, preset_sw_pattern, chunk_size, rrc_alpha, rrc_t, extract_type, pulse_train_sigma, detector_threshold_1, detector_threshold_2, basebandconv_freq_delta, costas_damping, costas_loop_bandwidth, costas_k1, costas_k2, rotation_lo, rotation_hi, pulse_train_length, disposition, downconvert_shift,gp1,gp2, gpdepth, fdmsep]
      #csv_data = [mode, ebn0_db, SNR_equiv_db, ber, characters_per_second, bits_per_second, noise_factor, standingwave_pattern, standingwave_location]
      self.osmod.analysis.writeDataToFile(csv_data)

      self.debug.info_message("strongest frequency is: " + str(self.osmod.modulation_object.getStrongestFrequency(audio_array, 500, 2000)))
      self.debug.info_message("strongest frequencies over range is: " + str(self.osmod.modulation_object.getStrongestFrequencies(audio_array, 20, 500, 2000)))

      #timestamp = datetime.utcnow().strftime('%y%m%d%H%M%S')
      timestamp = datetime.utcnow().strftime('%Y/%m/%d %H:%M')
      self.debug.info_message("timestamp UTC: " + str(timestamp))


    except:
      self.debug.error_message("Exception in test1: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
