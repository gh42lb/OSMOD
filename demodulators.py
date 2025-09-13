#!/usr/bin/env python

import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
#import matplotlib.pyplot as plt
from numpy import pi
from scipy.signal import butter, filtfilt, firwin, TransferFunction, lfilter, lfiltic
from modem_core_utils import ModemCoreUtils
from scipy import stats
from scipy.fft import fft, fftfreq
from collections import Counter
import ctypes
import cmath
from scipy import signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import savgol_filter, hilbert
from scipy.signal import correlate, find_peaks

from osmod_c_interface import ptoc_float_array, ptoc_double_array, ptoc_float, ctop_int, ptoc_int_array, ptoc_numpy_int_array, ptoc_double

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


class DemodulatorPSK(ModemCoreUtils):

  plotfirstonly = False

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_DEMOD)
    self.debug.info_message("__init__")
    self.remainder = np.array([])
    self.block_start_candidates = []
    super().__init__(osmod)

  def convertToInt(self, samples, threshold, offset):

    override_threshold_checked = self.osmod.form_gui.window['cb_override_extractionthreshold'].get()
    if override_threshold_checked:
      threshold = int(self.osmod.form_gui.window['in_extractionthreshold'].get())

    int_values = []
    for i in range(len(samples)):
      if samples[i]>offset + threshold:
        int_values.append(1)
      elif samples[i]<(-1*threshold)+offset:
        int_values.append(-1)
      else:
        int_values.append(0)
    return int_values

  """ This method works fine"""
  def decodeDisplay8psk(self, frequency, signal_a, signal_b):
    symbol_to_bits = {'1:1': [0,0,0,], '1:0': [0,0,1], '1:-1': [0,1,0], '0:-1': [0,1,1], '-1:-1': [1,0,0], '-1:0': [1,0,1], '-1:1': [1,1,0], '0:1': [1,1,1] } 

    """ threshold (800) makes a huge difference to decoding accuracy! """
    decoded_values_real = self.processDownsampleMinMax(frequency, signal_a, 800, 0.0)
    decoded_values_imag = self.processDownsampleMinMax(frequency, signal_b, 800, 0.0)
    self.debug.info_message("decoded values real: " + str(decoded_values_real))
    self.debug.info_message("decoded values imag: " + str(decoded_values_imag))

    #if self.plotfirstonly==False:
    #  self.plotfirstonly = True
    #  self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_oneoffour'))
    #  self.osmod.form_gui.plotQueue.put((signal_b, 'chart_canvas_twooffour'))


    for i in range(0, len(decoded_values_real)):
      lookup_string = str(decoded_values_real[i]) + ':' + str(decoded_values_imag[i])
      if lookup_string == '0:0':
        self.osmod.has_invalid_decodes = True
        self.debug.error_message("invalid decode: " + lookup_string)
        self.osmod.form_gui.window['ml_txrx_recvtext'].print('*', end="", text_color='red', background_color = 'white')
      else:
        self.debug.info_message("looking up: " + lookup_string)
        self.debug.info_message("bit value: " + str(symbol_to_bits[lookup_string]))
        self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(symbol_to_bits[lookup_string]), end="", text_color='black', background_color = 'white')

  def displayTextResults(self, decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag):
    self.debug.info_message("displayTextResults" )

    symbol_to_bits = {'1:1': '000', '1:0': '001', '1:-1': '010', '0:-1': '011', '-1:-1': '100', '-1:0': '101', '-1:1': '110', '0:1': '111' } 

    rotation = 0
    #rotation = 3
    symbol_to_bits_offset = [0] * 8
    symbol_to_bits_offset[0] = {'1:1': '000', '1:0': '001', '1:-1': '010', '0:-1': '011', '-1:-1': '100', '-1:0': '101', '-1:1': '110', '0:1': '111' } 
    symbol_to_bits_offset[1] = {'1:1': '001', '1:0': '010', '1:-1': '011', '0:-1': '100', '-1:-1': '101', '-1:0': '110', '-1:1': '111', '0:1': '000' } 
    symbol_to_bits_offset[2] = {'1:1': '010', '1:0': '011', '1:-1': '100', '0:-1': '101', '-1:-1': '110', '-1:0': '111', '-1:1': '000', '0:1': '001' } 
    symbol_to_bits_offset[3] = {'1:1': '011', '1:0': '100', '1:-1': '101', '0:-1': '110', '-1:-1': '111', '-1:0': '000', '-1:1': '001', '0:1': '010' } 
    symbol_to_bits_offset[4] = {'1:1': '100', '1:0': '101', '1:-1': '110', '0:-1': '111', '-1:-1': '000', '-1:0': '001', '-1:1': '010', '0:1': '011' } 
    symbol_to_bits_offset[5] = {'1:1': '101', '1:0': '110', '1:-1': '111', '0:-1': '000', '-1:-1': '001', '-1:0': '010', '-1:1': '011', '0:1': '100' } 
    symbol_to_bits_offset[6] = {'1:1': '110', '1:0': '111', '1:-1': '000', '0:-1': '001', '-1:-1': '010', '-1:0': '011', '-1:1': '100', '0:1': '101' } 
    symbol_to_bits_offset[7] = {'1:1': '111', '1:0': '000', '1:-1': '001', '0:-1': '010', '-1:-1': '011', '-1:0': '100', '-1:1': '101', '0:1': '110' } 
    symbol_rotation = {'1:1': 0, '0:1': 1, '-1:1': 2, '-1:0': 3, '-1:-1': 4, '0:-1': 5, '1:-1': 6, '1:0': 7}
    #symbol_rotation = {'1:1': 0, '0:1': 7, '-1:1': 6, '-1:0': 5, '-1:-1': 4, '0:-1': 3, '1:-1': 2, '1:0': 1}
    """
      self.osmod.sequence_start_character_detected_low = False
      self.osmod.sequence_start_character_rotation_low = 0
      self.osmod.sequence_start_character_detected_high = False
      self.osmod.sequence_start_character_rotation_high = 0
    """

    """ use_rotate enables detection via start sequence"""
    use_rotate = False

    #if self.osmod.phase_align == 'start_seq' and self.osmod.start_seq == '2_of_3':
    #  use_rotate = True

    #rotation = self.osmod.phase_rotation
    #rotation = 7
    #self.osmod.sequence_start_character_detected_low = True
    #self.osmod.sequence_start_character_detected_high = True
    #self.osmod.sequence_start_character_rotation_low  = self.osmod.phase_rotation[0]
    #self.osmod.sequence_start_character_rotation_high = self.osmod.phase_rotation[1]



    try:

      if self.osmod.phase_align == 'start_seq' and self.osmod.start_seq == '2_of_3':
        lookup_low_1  = str(decoded_values1_real[0]) + ':' + str(decoded_values1_imag[0])
        lookup_high_1 = str(decoded_values2_real[0]) + ':' + str(decoded_values2_imag[0])
        lookup_low_2  = str(decoded_values1_real[1]) + ':' + str(decoded_values1_imag[1])
        lookup_high_2 = str(decoded_values2_real[1]) + ':' + str(decoded_values2_imag[1])
        lookup_low_3  = str(decoded_values1_real[2]) + ':' + str(decoded_values1_imag[2])
        lookup_high_3 = str(decoded_values2_real[2]) + ':' + str(decoded_values2_imag[2])
        if lookup_low_1 != '0:0' and lookup_high_1 != '0:0':
          if lookup_low_2 != '0:0' and lookup_high_2 != '0:0':
            if lookup_low_1 == lookup_low_2 and lookup_high_1 == lookup_high_2:
              self.osmod.sequence_start_character_rotation_low  = symbol_rotation[lookup_low_1]
              self.osmod.sequence_start_character_rotation_high = symbol_rotation[lookup_high_1]
              use_rotate = True
          if lookup_low_3 != '0:0' and lookup_high_3 != '0:0':
            if lookup_low_1 == lookup_low_3 and lookup_high_1 == lookup_high_3:
              self.osmod.sequence_start_character_rotation_low  = symbol_rotation[lookup_low_1]
              self.osmod.sequence_start_character_rotation_high = symbol_rotation[lookup_high_1]
              use_rotate = True
        if lookup_low_2 != '0:0' and lookup_high_2 != '0:0':
          if lookup_low_3 != '0:0' and lookup_high_3 != '0:0':
            if lookup_low_2 == lookup_low_3 and lookup_high_2 == lookup_high_3:
              self.osmod.sequence_start_character_rotation_low  = symbol_rotation[lookup_low_2]
              self.osmod.sequence_start_character_rotation_high = symbol_rotation[lookup_high_2]
              use_rotate = True


      decoded_bitstring_1 = []
      decoded_bitstring_2 = []

      for i in range(0, len(decoded_values1_real)):
        lookup_string1 = str(decoded_values1_real[i]) + ':' + str(decoded_values1_imag[i])
        lookup_string2 = str(decoded_values2_real[i]) + ':' + str(decoded_values2_imag[i])

        if lookup_string1 == '0:0' or lookup_string2 == '0:0':
          self.osmod.has_invalid_decodes = True
          self.debug.error_message("invalid decode: " + lookup_string1)
          self.osmod.form_gui.window['ml_txrx_recvtext'].print('*', end="", text_color='red', background_color = 'white')
        else:
          #if use_rotate == True:
          #  """ rotation is based on 000,000 being the first character i.e. character 'a' """
          #  if self.osmod.sequence_start_character_detected_low == False:
          #    self.osmod.sequence_start_character_detected_low = True
          #    self.osmod.sequence_start_character_rotation_low = symbol_rotation[lookup_string1]
          #  if self.osmod.sequence_start_character_detected_high == False:
          #    self.osmod.sequence_start_character_detected_high = True
          #    self.osmod.sequence_start_character_rotation_high = symbol_rotation[lookup_string2]



          self.debug.info_message("looking up: " + lookup_string1)
          #self.debug.info_message("bit value: " + str(symbol_to_bits[lookup_string1]))

          self.debug.info_message("bit value: " + str(symbol_to_bits_offset[rotation][lookup_string1]))

          if self.osmod.process_debug == True:
            #self.osmod.form_gui.window['ml_txrx_recvtext'].print('[' + str(symbol_to_bits[lookup_string1]), end="", text_color='green', background_color = 'white')
            #self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(symbol_to_bits[lookup_string2]) + ']', end="", text_color='green', background_color = 'white')
            if use_rotate == False:
              self.osmod.form_gui.window['ml_txrx_recvtext'].print('[' + str(symbol_to_bits_offset[rotation][lookup_string1]), end="", text_color='green', background_color = 'white')
              self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(symbol_to_bits_offset[rotation][lookup_string2]) + ']', end="", text_color='green', background_color = 'white')
            else:
              self.osmod.form_gui.window['ml_txrx_recvtext'].print('[' + str(symbol_to_bits_offset[self.osmod.sequence_start_character_rotation_low][lookup_string1]), end="", text_color='green', background_color = 'white')
              self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(symbol_to_bits_offset[self.osmod.sequence_start_character_rotation_high][lookup_string2]) + ']', end="", text_color='green', background_color = 'white')

          #decoded_bitstring_1.append(str(symbol_to_bits[lookup_string1]))
          #decoded_bitstring_2.append(str(symbol_to_bits[lookup_string2]))
          if use_rotate == False:
            decoded_bitstring_1.append(str(symbol_to_bits_offset[rotation][lookup_string1]))
            decoded_bitstring_2.append(str(symbol_to_bits_offset[rotation][lookup_string2]))
          else:
            decoded_bitstring_1.append(str(symbol_to_bits_offset[self.osmod.sequence_start_character_rotation_low][lookup_string1]))
            decoded_bitstring_2.append(str(symbol_to_bits_offset[self.osmod.sequence_start_character_rotation_high][lookup_string2]))

      for i in range(0, len(decoded_bitstring_1)):
        lookup_string = str(decoded_bitstring_1[i]) + str(decoded_bitstring_2[i])

        """ decimal index of 6 bit binary code (base 64) """
        binary = int(lookup_string, 2)

        char = self.b64_charfromindex_list[binary]
        self.debug.info_message("found char: " + str(char))
        self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(char), end="", text_color='blue', background_color = 'white')

      return decoded_bitstring_1, decoded_bitstring_2
    except:
      sys.stdout.write("Exception in displayTextResults: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")



  """ searches for the first occurrence of 2 sequential characters...characters 'b' and 'i' equiv to (low,hi) [000,001] and [001,000] """
  def locateStartSequence(self, signal1, signal2, median_block_offset, frequency, num_waves, where1, where2):
    self.debug.info_message("locateStartSequence" )
    try:
      self.debug.info_message("locating start sequence" )
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      phase_match_offset_lower = 0
      phase_match_offset_higher = 0
      found_match_lower = False
      found_match_higher = False

      """ rebase lower frequency signal just prior to extraction"""
      max1 = np.max(np.abs(signal1[0]))
      max2 = np.max(np.abs(signal1[1]))
      ratio = max(max1, max2) / self.osmod.parameters[3]
      signal1[0] = signal1[0]/ratio
      signal1[1] = signal1[1]/ratio

      """ rebase higher frequency signal just prior to extraction"""
      max1 = np.max(np.abs(signal1[0]))
      max2 = np.max(np.abs(signal1[1]))
      ratio = max(max1, max2) / self.osmod.parameters[3]
      signal2[0] = signal2[0]/ratio
      signal2[1] = signal2[1]/ratio

      """
          middle = x + where
          decoded_values1.append(sum(signal1[middle-(term1):middle+(term1)-1]) / (2*term1))
          decoded_values2.append(sum(signal2[middle-(term1):middle+(term1)-1]) / (2*term1))

      """
      num_points_low  = int(self.osmod.sample_rate / frequency[0])
      num_points_high = int(self.osmod.sample_rate / frequency[1])
      term1_low  = num_points_low  * num_waves
      term1_high = num_points_high * num_waves
      test_phase = [0]*8
      count = 0
      for x in range(median_block_offset, len(signal1[0]), self.osmod.symbol_block_size):
        if x <= len(signal1[0]) - self.osmod.symbol_block_size:
          middle1 = x + where1
          middle2 = x + where1 + int(self.osmod.symbol_block_size/2)   #x + where2
          """ 1st character 000,001 aka 1:1 1:0"""
          """ 2nd character 001,000 aka 1:0 1:1"""

          #for offset in range(-5000, 5000):
          for offset in range(- (2*pulse_length), (2*pulse_length)):

            if found_match_lower == False:
              """ test this block"""
              test_phase[0] = sum(signal1[0][middle1 - term1_low + offset:middle1 + term1_low + offset -1]) / (2*term1_low)
              test_phase[1] = sum(signal1[1][middle1 - term1_low + offset:middle1 + term1_low + offset -1]) / (2*term1_low) 
              test_phase[2] = sum(signal2[0][middle2 - term1_low + offset:middle2 + term1_low + offset -1]) / (2*term1_low)
              test_phase[3] = sum(signal2[1][middle2 - term1_low + offset:middle2 + term1_low + offset -1]) / (2*term1_low)
              if test_phase[0] > self.osmod.parameters[0] and test_phase[1] > self.osmod.parameters[0]: #1:1
                if test_phase[2] > self.osmod.parameters[0]: # 1:
                  if test_phase[3] <= self.osmod.parameters[0] and test_phase[3] >= -1 * self.osmod.parameters[0]: # :0
                    self.debug.info_message("found phase match lower!!!!" )
                    self.debug.info_message("offset: " + str(offset) )
                    phase_match_offset_lower = x
                    found_match_lower = True
            if found_match_higher == False:
              """ test the next block"""
              test_phase[4] = sum(signal1[0][self.osmod.symbol_block_size + middle1 - term1_high + offset:self.osmod.symbol_block_size + middle1 + term1_high + offset -1]) / (2*term1_high)
              test_phase[5] = sum(signal1[1][self.osmod.symbol_block_size + middle1 - term1_high + offset:self.osmod.symbol_block_size + middle1 + term1_high + offset -1]) / (2*term1_high)
              test_phase[6] = sum(signal2[0][self.osmod.symbol_block_size + middle2 - term1_high + offset:self.osmod.symbol_block_size + middle2 + term1_high + offset -1]) / (2*term1_high)
              test_phase[7] = sum(signal2[1][self.osmod.symbol_block_size + middle2 - term1_high + offset:self.osmod.symbol_block_size + middle2 + term1_high + offset -1]) / (2*term1_high)
              if test_phase[4] > self.osmod.parameters[0]: # 1:
                if test_phase[5] <= self.osmod.parameters[0] and test_phase[5] >= -1 * self.osmod.parameters[0]: # :0
                  if test_phase[6] > self.osmod.parameters[0] and test_phase[7] > self.osmod.parameters[0]: #1:1
                    self.debug.info_message("found phase match higher!!!!" )
                    self.debug.info_message("offset: " + str(offset) )
                    phase_match_offset_higher = x
                    found_match_higher = True
                  #break
        if found_match_lower and found_match_higher:
          break
        count = count + 1
        if count == 5:
          break
      if not (found_match_lower and found_match_higher):
        self.debug.info_message("no match for phase sync" )

      #if not found_match:
      #  self.debug.info_message("no phase match found" )

    except:
      sys.stdout.write("Exception in locateStartSequence: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def displayChartResults(self, signal_a, signal_b, color1, color2, erase, chart_name, which_chart):
    if which_chart == ocn.CHART_ONE:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_oneoffour', chart_name, color1, erase))
      self.osmod.form_gui.plotQueue.put((signal_b, 'chart_canvas_oneoffour', chart_name, color2, False))
    elif which_chart == ocn.CHART_TWO:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_twooffour', chart_name, color1, erase))
      self.osmod.form_gui.plotQueue.put((signal_b, 'chart_canvas_twooffour', chart_name, color2, False))
    elif which_chart == ocn.CHART_THREE:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_threeoffour', chart_name, color1, erase))
      self.osmod.form_gui.plotQueue.put((signal_b, 'chart_canvas_threeoffour', chart_name, color2, False))
    elif which_chart == ocn.CHART_FOUR:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_fouroffour', chart_name, color1, erase))
      self.osmod.form_gui.plotQueue.put((signal_b, 'chart_canvas_fouroffour', chart_name, color2, False))
    elif which_chart == ocn.CHART_ONE_A:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_oneoffour', chart_name, color1, erase))
    elif which_chart == ocn.CHART_TWO_A:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_twooffour', chart_name, color1, erase))
    elif which_chart == ocn.CHART_THREE_A:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_threeoffour', chart_name, color1, erase))
    elif which_chart == ocn.CHART_FOUR_A:
      self.osmod.form_gui.plotQueue.put((signal_a, 'chart_canvas_fouroffour', chart_name, color1, erase))

  def getMode(self, int_array):
    #self.debug.info_message("getMode")

    try:
      #self.debug.info_message("type is: " + str(type(int_array)))
      if self.osmod.use_compiled_c_code == True:
        if isinstance(int_array, list):
          python_list = int_array
        elif isinstance(int_array, np.ndarray) and np.issubdtype(int_array.dtype, np.integer):
          python_list = [0]*len(int_array)
          for i in range(0, len(int_array)):
            python_list[i] = int(int_array[i])

        #self.debug.info_message("calling compiled C code")

        c_int_array = ptoc_int_array(python_list)

        self.osmod.compiled_lib.find_mode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.osmod.compiled_lib.find_mode.restype = ctypes.c_int
        mode = self.osmod.compiled_lib.find_mode(c_int_array, len(python_list))

        return mode

      else:
        return int(np.argmax(np.bincount(int_array)))

    except:
      self.debug.error_message("Exception in getMode: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def detectSampleOffset(self, signal):
    self.debug.info_message("detectSampleOffset" )
    try:
      max_list = []
      min_list = []
      all_list = []
      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        #test_min = np.min(test_peak)
        max_indices = np.where(test_peak == test_max)
        #min_indices = np.where(test_peak == test_min)
        self.debug.info_message("max indices: " + str(max_indices[0]))
        #self.debug.info_message("min indices: " + str(min_indices[0]))

        for x in range(0, len(max_indices[0]) ):
          max_list.append(int(max_indices[0][x]))

        #for item in list(max_indices[0]):
        #  max_list.append(item)
        #  all_list.append(item)
        #for item in list(min_indices[0]):
        #  min_list.append(item)
        #  all_list.append(item)

      self.debug.info_message("max_list: " + str(list(max_list)))
      #self.debug.info_message("min_list: " + str(list(min_list)))
      #self.debug.info_message("all_list: " + str(list(all_list)))
      if self.osmod.detector_function == 'median':
        self.debug.info_message("calculating median" )
        median_index_max = int(np.median(np.array(max_list)))
        #median_index_min = int(np.median(np.array(min_list)))
        #median_index_all = int(np.median(np.array(all_list)))
      elif self.osmod.detector_function == 'mode':
        self.debug.info_message("calculating mode" )
        #median_index_max = int(stats.mode(max_list).mode[0])
        median_index_max = self.getMode(max_list)
        #median_index_max = int(np.argmax(np.bincount(max_list)))
        #median_index_min = int(stats.mode(min_list).mode[0])
        #median_index_all = int(stats.mode(all_list).mode[0])
      self.debug.info_message("mean max index: " + str(median_index_max))
      #self.debug.info_message("mean min index: " + str(median_index_min))
      #self.debug.info_message("mean all index: " + str(median_index_all))

      pulse_size = (self.osmod.symbol_block_size*2)/self.osmod.pulses_per_block
      half_pulse_size = pulse_size / 2
      sample_start = int((median_index_max + half_pulse_size) % pulse_size)

      self.debug.info_message("sample_start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in detectSampleOffset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return sample_start


  def extractPhaseValuesIntraTriple(self, complex_signal, pulse_start_index, frequency, pulse_length, where1, where2, where3, fine_tune_adjust, low_high, residuals):
    self.debug.info_message("extractPhaseValuesIntraTriple" )
    diff_codes_group_1 = []
    diff_codes_group_2 = []
    angles_real_part = []
    angles_imag_part = []
    angles_list = []

    angle_a_real = np.array([0] * len(complex_signal), dtype = np.float32)
    angle_b_real = np.array([0] * len(complex_signal), dtype = np.float32)
    angle_c_real = np.array([0] * len(complex_signal), dtype = np.float32)
    angle_a_imag = np.array([0] * len(complex_signal), dtype = np.float32)
    angle_b_imag = np.array([0] * len(complex_signal), dtype = np.float32)
    angle_c_imag = np.array([0] * len(complex_signal), dtype = np.float32)


    graph_real_1 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_real_2 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_real_3 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_angle_1 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_angle_2 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_angle_3 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_imag_1 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_imag_2 =  np.array([0] * len(complex_signal), dtype = np.float32)
    graph_imag_3 =  np.array([0] * len(complex_signal), dtype = np.float32)

    sub_sample_a_real  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_b_real  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_c_real  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_a_imag  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_b_imag  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_c_imag  = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_a_angle = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_b_angle = np.array([0] * pulse_length, dtype = np.float32)
    sub_sample_c_angle = np.array([0] * pulse_length, dtype = np.float32)

    try:
      def extract_pulse_center():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag

        num_waves  = self.osmod.parameters[4]
        num_points = int(self.osmod.sample_rate / frequency)
        term1 = int(num_points*num_waves)
        half  = int(pulse_length/2)
        angle_a_real = sum(sub_sample_a_real[half - term1:half + term1 - 1]) / (2*term1)
        angle_b_real = sum(sub_sample_b_real[half - term1:half + term1 - 1]) / (2*term1)
        angle_c_real = sum(sub_sample_c_real[half - term1:half + term1 - 1]) / (2*term1)
        angle_a_imag = sum(sub_sample_a_imag[half - term1:half + term1 - 1]) / (2*term1)
        angle_b_imag = sum(sub_sample_b_imag[half - term1:half + term1 - 1]) / (2*term1)
        angle_c_imag = sum(sub_sample_c_imag[half - term1:half + term1 - 1]) / (2*term1)
        angles_real_part.append([angle_a_real, angle_b_real, angle_c_real])
        angles_imag_part.append([angle_a_imag, angle_b_imag, angle_c_imag])
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c

      def extract_pulse_all_filtered():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        angle_a_real = self.filteredSum(sub_sample_a_real[0:pulse_length], filter_ratio, filter_inc)
        angle_b_real = self.filteredSum(sub_sample_b_real[0:pulse_length], filter_ratio, filter_inc)
        angle_c_real = self.filteredSum(sub_sample_c_real[0:pulse_length], filter_ratio, filter_inc)
        angle_a_imag = self.filteredSum(sub_sample_a_imag[0:pulse_length], filter_ratio, filter_inc)
        angle_b_imag = self.filteredSum(sub_sample_b_imag[0:pulse_length], filter_ratio, filter_inc)
        angle_c_imag = self.filteredSum(sub_sample_c_imag[0:pulse_length], filter_ratio, filter_inc)
        angles_real_part.append([angle_a_real, angle_b_real, angle_c_real])
        angles_imag_part.append([angle_a_imag, angle_b_imag, angle_c_imag])
        return [angle_a_real, angle_b_real, angle_c_real], [angle_a_imag, angle_b_imag, angle_c_imag]

      def extract_pulse_all():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        num_waves  = self.osmod.parameters[4]
        num_points = int(self.osmod.sample_rate / frequency)
        term1 = int(num_points*num_waves)
        half  = int(pulse_length/2)
        angle_a_real = sum(sub_sample_a_real[0:pulse_length]) / pulse_length
        angle_b_real = sum(sub_sample_b_real[0:pulse_length]) / pulse_length
        angle_c_real = sum(sub_sample_c_real[0:pulse_length]) / pulse_length
        angle_a_imag = sum(sub_sample_a_imag[0:pulse_length]) / pulse_length
        angle_b_imag = sum(sub_sample_b_imag[0:pulse_length]) / pulse_length
        angle_c_imag = sum(sub_sample_c_imag[0:pulse_length]) / pulse_length
        angles_real_part.append([angle_a_real, angle_b_real, angle_c_real])
        angles_imag_part.append([angle_a_imag, angle_b_imag, angle_c_imag])
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c

      def extract_angles_center_average():
        num_waves  = self.osmod.parameters[4]
        num_points = int(self.osmod.sample_rate / frequency)
        term1 = int(num_points*num_waves)
        half  = int(pulse_length/2)
        angle_a = self.averageAngles3(np.angle(complex_signal[middle_a + half - term1:middle_a + half + term1 -1]), filter_ratio, filter_inc)
        angle_b = self.averageAngles3(np.angle(complex_signal[middle_b + half - term1:middle_b + half + term1 -1]), filter_ratio, filter_inc)
        angle_c = self.averageAngles3(np.angle(complex_signal[middle_c + half - term1:middle_c + half + term1 -1]), filter_ratio, filter_inc)
        return angle_a, angle_b, angle_c

      def extract_angles_all_average():
        angle_a = self.averageAngles3(np.angle(complex_signal[middle_a:middle_a+pulse_length]), filter_ratio, filter_inc)
        angle_b = self.averageAngles3(np.angle(complex_signal[middle_b:middle_b+pulse_length]), filter_ratio, filter_inc)
        angle_c = self.averageAngles3(np.angle(complex_signal[middle_c:middle_c+pulse_length]), filter_ratio, filter_inc)
        return angle_a, angle_b, angle_c

      def extract_angles_all_savitzky_golay():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        window_length = 7 #must be odd number
        polyorder = 3
        angle_a_real = sum(savgol_filter(sub_sample_a_real[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_b_real = sum(savgol_filter(sub_sample_b_real[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_c_real = sum(savgol_filter(sub_sample_c_real[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_a_imag = sum(savgol_filter(sub_sample_a_imag[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_b_imag = sum(savgol_filter(sub_sample_b_imag[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_c_imag = sum(savgol_filter(sub_sample_c_imag[0:pulse_length], window_length, polyorder)) / pulse_length
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c

      def extract_angles_all_chebyshev():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        self.debug.info_message("extract_angles_all_chebyshev" )
        sos = signal.cheby1(4, 5, 1/20, 'low', output='sos')
        angle_a_real = sum(signal.sosfilt(sos, sub_sample_a_real[0:pulse_length])) / pulse_length
        angle_b_real = sum(signal.sosfilt(sos, sub_sample_b_real[0:pulse_length])) / pulse_length
        angle_c_real = sum(signal.sosfilt(sos, sub_sample_c_real[0:pulse_length])) / pulse_length
        angle_a_imag = sum(signal.sosfilt(sos, sub_sample_a_imag[0:pulse_length])) / pulse_length
        angle_b_imag = sum(signal.sosfilt(sos, sub_sample_b_imag[0:pulse_length])) / pulse_length
        angle_c_imag = sum(signal.sosfilt(sos, sub_sample_c_imag[0:pulse_length])) / pulse_length
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c

      def extract_angles_all_gaussian():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        sigma = 5 # this works well
        angle_a_real = sum(gaussian_filter1d(sub_sample_a_real[0:pulse_length], sigma)) / pulse_length
        angle_b_real = sum(gaussian_filter1d(sub_sample_b_real[0:pulse_length], sigma)) / pulse_length
        angle_c_real = sum(gaussian_filter1d(sub_sample_c_real[0:pulse_length], sigma)) / pulse_length
        angle_a_imag = sum(gaussian_filter1d(sub_sample_a_imag[0:pulse_length], sigma)) / pulse_length
        angle_b_imag = sum(gaussian_filter1d(sub_sample_b_imag[0:pulse_length], sigma)) / pulse_length
        angle_c_imag = sum(gaussian_filter1d(sub_sample_c_imag[0:pulse_length], sigma)) / pulse_length
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c

      def extract_angles_center_gaussian():
        nonlocal angle_a_real
        nonlocal angle_b_real
        nonlocal angle_c_real
        nonlocal angle_a_imag
        nonlocal angle_b_imag
        nonlocal angle_c_imag
        num_waves  = self.osmod.parameters[4]
        num_points = int(self.osmod.sample_rate / frequency)
        term1 = int(num_points*num_waves)
        half  = int(pulse_length/2)
        sigma = 5
        angle_a_real = sum(gaussian_filter1d(sub_sample_a_real[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_b_real = sum(gaussian_filter1d(sub_sample_b_real[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_c_real = sum(gaussian_filter1d(sub_sample_c_real[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_a_imag = sum(gaussian_filter1d(sub_sample_a_imag[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_b_imag = sum(gaussian_filter1d(sub_sample_b_imag[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_c_imag = sum(gaussian_filter1d(sub_sample_c_imag[half - term1:half + term1 -1], sigma)) / (2*term1)
        angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
        angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
        angle_c = np.angle(angle_c_real + 1j * angle_c_imag)
        return angle_a, angle_b, angle_c



      num_points = int(self.osmod.sample_rate / frequency)
      term1 = num_points*1

      two_times_pi = np.pi * 2

      extraction_factor = float(self.osmod.form_gui.window['in_intra_extract_factor'].get())

      filter_ratio    = float(self.osmod.form_gui.window['in_intra_extract_filterratio'].get())
      filter_inc      = float(self.osmod.form_gui.window['in_intra_extract_filterinc'].get())
      search_accuracy = float(self.osmod.form_gui.window['in_intra_extract_searchaccuracy'].get())

      enable_display = self.osmod.form_gui.window['cb_display_phases'].get()


      selected_type = self.osmod.form_gui.window['combo_intra_extract_type'].get()
      if selected_type == 'Type 1':
        extraction_type = ocn.INTRA_EXTRACT_TYPE1
      elif selected_type == 'Type 2':
        extraction_type = ocn.INTRA_EXTRACT_TYPE2
      elif selected_type == 'Type 3':
        extraction_type = ocn.INTRA_EXTRACT_TYPE3
      elif selected_type == 'Type 4':
        extraction_type = ocn.INTRA_EXTRACT_TYPE4
      elif selected_type == 'Type 5':
        extraction_type = ocn.INTRA_EXTRACT_TYPE5
      elif selected_type == 'Type 6':
        extraction_type = ocn.INTRA_EXTRACT_TYPE6
      elif selected_type == 'Type 7':
        extraction_type = ocn.INTRA_EXTRACT_TYPE7

      half_pulse = int(pulse_length / 2)

      fine_tune_pulse_start_index  = (pulse_start_index + fine_tune_adjust[low_high]) % pulse_length
      for x in range(fine_tune_pulse_start_index, len(complex_signal), self.osmod.symbol_block_size):
        if x <= len(complex_signal) - self.osmod.symbol_block_size:
          middle_a = x + where1
          middle_b = x + where2
          middle_c = x + where3

          for sig_a_count in range(0, pulse_length):
            sig_a = complex_signal[middle_a + sig_a_count]
            sub_sample_a_real[sig_a_count] = sig_a.real
            sub_sample_a_imag[sig_a_count] = sig_a.imag 
          for sig_b_count in range(0, pulse_length):
            sig_b = complex_signal[middle_b + sig_b_count]
            sub_sample_b_real[sig_b_count] = sig_b.real 
            sub_sample_b_imag[sig_b_count] = sig_b.imag 
          for sig_c_count in range(0, pulse_length):
            sig_c = complex_signal[middle_c + sig_c_count]
            sub_sample_c_real[sig_c_count] = sig_c.real 
            sub_sample_c_imag[sig_c_count] = sig_c.imag 


          if extraction_type == ocn.INTRA_EXTRACT_TYPE1:
            #angle_a = np.mean(np.angle(complex_signal[middle_a:middle_a+pulse_length]))
            #angle_b = np.mean(np.angle(complex_signal[middle_b:middle_b+pulse_length]))
            #angle_c = np.mean(np.angle(complex_signal[middle_c:middle_c+pulse_length]))
            angle_a, angle_b, angle_c = extract_angles_all_chebyshev()
            angles_list.append(angle_a)

          elif extraction_type == ocn.INTRA_EXTRACT_TYPE2:
            angle_a, angle_b, angle_c = extract_angles_all_gaussian()
            #angle_a, angle_b, angle_c = extract_angles_center_average()     
            angles_list.append(angle_a)

          elif extraction_type == ocn.INTRA_EXTRACT_TYPE3:
            angle_a, angle_b, angle_c = extract_angles_all_average()     # this works well
            angles_list.append(angle_a)

          elif extraction_type == ocn.INTRA_EXTRACT_TYPE4:
            angle_a, angle_b, angle_c = extract_angles_all_savitzky_golay()    # this works well
            angles_list.append(angle_a)

          elif extraction_type == ocn.INTRA_EXTRACT_TYPE5:
            angle_a, angle_b, angle_c = extract_angles_center_gaussian()
            angles_list.append(angle_a)

            """ angles_real_part and angles_imag_part used for second decode"""
            #extracted_real, extracted_imag = extract_pulse_all()
            #angle_a_real = extracted_real[0]
            #angle_b_real = extracted_real[1]
            #angle_c_real = extracted_real[2]
            #angle_a_imag = extracted_imag[0]
            #angle_b_imag = extracted_imag[1]
            #angle_c_imag = extracted_imag[2]

            #extract_pulse_center()

            #angle_a = np.angle(angle_a_real + 1j * angle_a_imag)
            #angle_b = np.angle(angle_b_real + 1j * angle_b_imag)
            #angle_c = np.angle(angle_c_real + 1j * angle_c_imag)

            #angle_a, angle_b, angle_c = extract_angles_all_average()     # this works well
            #angle_a, angle_b, angle_c = extract_angles_all_savitzky_golay()    # this works well
            #angle_a, angle_b, angle_c = extract_angles_all_gaussian()

            #angle_a, angle_b, angle_c = extract_angles_all_chebyshev()
            #angle_a, angle_b, angle_c = extract_angles_all_whittaker_eilers()
            #angle_a, angle_b, angle_c = extract_angles_center()
            """ angles_list is used for first decode"""

            if enable_display:
              middles = [middle_a, middle_b, middle_c]
              for middle_num in range(0,3):
                for pulse_num in range(0, pulse_length):
                  graph_angle_1[middles[middle_num] + pulse_num] = (angle_a)
                  graph_angle_2[middles[middle_num] + pulse_num] = (angle_b)
                  graph_angle_3[middles[middle_num] + pulse_num] = (angle_c)

                  graph_real_1[middles[middle_num] + pulse_num] = angle_a_real
                  graph_real_2[middles[middle_num] + pulse_num] = angle_b_real
                  graph_real_3[middles[middle_num] + pulse_num] = angle_c_real
                  graph_imag_1[middles[middle_num] + pulse_num] = angle_a_imag
                  graph_imag_2[middles[middle_num] + pulse_num] = angle_b_imag
                  graph_imag_3[middles[middle_num] + pulse_num] = angle_c_imag



      if enable_display:
        """ rebase the primary ral and imag charts"""
        max1 = np.max(np.abs(graph_real_1))
        max2 = np.max(np.abs(graph_imag_1))
        min1 = np.min(np.abs(graph_real_1))
        min2 = np.min(np.abs(graph_imag_1))
        minmin = min(min1, min2)
        maxmax = max(max(max1, max2), abs(minmin))
        ratio = max(max1, max2) / self.osmod.parameters[3]
        graph_real_1 = graph_real_1/ratio
        graph_imag_1 = graph_imag_1/ratio
        graph_real_1[len(complex_signal)-2] = maxmax/ratio
        graph_imag_1[len(complex_signal)-2] = maxmax/ratio
        graph_real_1[len(complex_signal)-1] = minmin/ratio
        graph_imag_1[len(complex_signal)-1] = minmin/ratio


      all_charts = [graph_angle_1, graph_angle_2, graph_angle_3, graph_real_1, graph_real_2, graph_real_3, graph_imag_1, graph_imag_2, graph_imag_3]


      #self.debug.info_message("angles_real_part: " + str(angles_real_part) )
      #self.debug.info_message("angles_real_part: " + str(angles_real_part) )

      decoded_values_real = []
      decoded_values_imag = []
      #decoded_values_real, decoded_values_imag = self.analyzePhaseDifferencesThreshold(angles_real_part, angles_imag_part)
      decoded_intvalues = [decoded_values_real, decoded_values_imag]
      self.debug.info_message("decoded_intvalues: " + str(decoded_intvalues) )


      decoded_intlist, decoded_bitstring = self.deriveIntlistFromAngles(angles_list, low_high, residuals)

      return decoded_intlist, decoded_bitstring, decoded_intvalues, all_charts
    except:
      sys.stdout.write("Exception in extractPhaseValuesIntraTriple: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def deriveIntlistFromAngles(self, angles, low_high, residuals):
    self.debug.info_message("deriveIntlistFromAngles" )
    try:

      def autoRotate():
        nonlocal adjustment
        nonlocal rotation_phase_eighth

        rotation_char_phase = [0] * 8
        best_adjustment = 0.0
        best_count = 0
        best_rotation = 0
        range_values    = [0] * 8
        range_values[0] = []
        range_values[1] = []
        range_values[2] = []
        range_values[3] = []
        range_values[4] = []
        range_values[5] = []
        range_values[6] = []
        range_values[7] = []
        range_number = 0
        in_range = False
        character_count = 3
      
        if self.osmod.start_seq == '2_of_3':
          character_count = 3
        elif self.osmod.start_seq == '2_of_4':
          character_count = 4
        elif self.osmod.start_seq == '2_of_5':
          character_count = 5
        elif self.osmod.start_seq == '2_of_6':
          character_count = 6
        elif self.osmod.start_seq == '2_of_7':
          character_count = 7
        elif self.osmod.start_seq == '2_of_8':
          character_count = 8

        self.debug.info_message("character_count" + str(character_count))
       
        for adjustment in np.arange(0, np.pi / 4, (np.pi / 4) / 40):
          if self.osmod.sample_rate == 48000:
            for i in range(0, character_count):
              rotation_char_phase[i] = int((self.normalizeAngle(np_angles[i] - adjustment- residuals[low_high][i]) / (2 * np.pi)) * 8)
          else:
            for i in range(0, character_count):
              rotation_char_phase[i] = int((self.normalizeAngle(np_angles[i] - adjustment) / (2 * np.pi)) * 8)

          bincount_array = np.bincount(rotation_char_phase)
          prevalent_count = max(bincount_array)
          prevalent_phase = np.argmax(bincount_array)
          rotation_phase_eighth = prevalent_phase

          if prevalent_count > best_count:
            range_number = 0
            range_values[range_number] = []
            best_count = prevalent_count
            best_adjustment = adjustment
            best_rotation = prevalent_phase
            in_range = True
            range_values[range_number].append((best_adjustment, best_rotation))
          elif prevalent_count == best_count:
            if in_range == False:
              range_number = range_number + 1
              range_values[range_number] = []
              in_range = True
            best_count = prevalent_count
            best_adjustment = adjustment
            best_rotation = prevalent_phase
            range_values[range_number].append((best_adjustment, best_rotation))
          else:
            in_range = False

        if range_number == 0:
          combined_range_values = range_values[0]
        else:
          combined_range_values = []
          for range_group in range(1, -1, -1):
            for range_item in range(0, len(range_values[range_group])):
              combined_range_values.append(range_values[range_group][range_item])


        self.debug.info_message("range_values" + str(range_values))
        self.debug.info_message("combined_range_values" + str(combined_range_values))

        range_len = len(combined_range_values)
        if range_len > 0:
          mid_point      = int(range_len / 2)
          if mid_point > 0:
            adjustment     = combined_range_values[mid_point-1][0]
            rotation_phase_eighth = combined_range_values[mid_point-1][1]
          else:
            adjustment     = combined_range_values[0][0]
            rotation_phase_eighth = combined_range_values[0][1]
          self.debug.info_message("********************")
          self.debug.info_message("mid_point: " + str(mid_point))
          self.debug.info_message("rotation_phase_eighth: " + str(rotation_phase_eighth))
          self.debug.info_message("adjustment: " + str(adjustment))
          self.debug.info_message("total angle: " + str(rotation_phase_eighth - adjustment))
          self.debug.info_message("********************")

          rotation_angle = rotation_phase_eighth * (np.pi / 4)
          #self.osmod.detector.rotation_angles[low_high] = self.normalizeAngle(rotation_phase_eighth - adjustment)
          #self.osmod.detector.rotation_angles[low_high] = self.normalizeAngle(rotation_angle - adjustment)
          #self.osmod.detector.rotation_angles[low_high] = self.normalizeAngle(0 - rotation_angle - adjustment)
          self.osmod.detector.rotation_angles[low_high] = self.normalizeAngle(rotation_angle + adjustment)


      """ start of code thread..."""
      np_angles = np.array(angles)
      median = np.median(np_angles % 8)
      self.debug.info_message("median" + str(median))
      adjustment = median - (np.pi / 8)
      self.debug.info_message("adjustment" + str(adjustment))
      adjustment = (np.pi / 8)

      intlist = []
      decoded_bitstring = []
      rotation_phase_eighth = 0

      """ auto calibration and auto rotation """
      if self.osmod.extrapolate == 'no':
        if self.osmod.phase_align == 'start_seq' and (self.osmod.start_seq == '2_of_3' or self.osmod.start_seq == '2_of_4' or self.osmod.start_seq == '2_of_5' or self.osmod.start_seq == '2_of_6' or self.osmod.start_seq == '2_of_7' or self.osmod.start_seq == '2_of_8'):
          autoRotate()

        elif self.osmod.phase_align == 'fixed_rotation':
          rotation_phase_eighth = self.osmod.phase_rotation[low_high]
          self.debug.info_message("fixed rotation_phase_eighth: " + str(rotation_phase_eighth))

      elif self.osmod.extrapolate == 'yes':
        if self.extrapolate_step == ocn.EXTRAPOLATE_FIND_DISPOSITION_ROTATION:
          self.debug.info_message("extrapolate auto rotation")
          autoRotate()

        elif self.extrapolate_step == ocn.EXTRAPOLATE_FIXED_ROTATION_DECODE:

          """ FIXME add calibrate only instead of auto rotate + calibrate"""
          if self.osmod.post_extrapolate_calibrate == "yes":
            autoRotate()
          else:
            rotation_dict = self.osmod.rotation_tables
            if rotation_dict != None:
              self.debug.info_message("extrapolate fixed rotation")
              half = int(self.osmod.pulses_per_block / 2)
              active_table = rotation_dict[str(half)]
              rotation_phase_eighth = 0
              adjustment = active_table[0][low_high]
            else:
              autoRotate() # fail safe use auto rotate
          

      block_count = 0
      residual_amount = 0.0
      for angle in np_angles:
        #term1 = (angle - adjustment + (2*np.pi)) % (2*np.pi)
        #index = int((term1 / (2 * np.pi)) * 8)

        if self.osmod.sample_rate == 48000:
          residual_amount = residuals[low_high][block_count]

        #index = int((int((self.normalizeAngle(angle - adjustment) / (2 * np.pi)) * 8) - rotation_phase_eighth + 8) % 8)
        index = int((int((self.normalizeAngle(angle - adjustment - residual_amount) / (2 * np.pi)) * 8) - rotation_phase_eighth + 8) % 8)
        intlist.append(index)
        binary = format(index, "06b")[0:6]
        for i in range(0, len(binary), 6):
          triplet = binary[i+3:i + 6]
          decoded_bitstring.append(triplet)
        block_count = block_count + 1


      self.debug.info_message("intlist" + str(intlist))
      self.debug.info_message("decoded_bitstring" + str(decoded_bitstring))
      return intlist, decoded_bitstring
    except:
      sys.stdout.write("Exception in deriveIntlistFromAngles: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  """
        index = self.b64_indexfromchar_dict[char]
        self.debug.info_message("index: " + str(index) )
        binary = format(index, "06b")[0:6]
        self.debug.info_message("binary : " + str(binary) )
        for i in range(0, len(binary), 6):
          triplet1 = binary[i:i + 3]
          triplet2 = binary[i+3:i + 6]
          self.debug.info_message("appending triplet1: " + str(triplet1) )
          sent_triplets_1.append(triplet1)
          self.debug.info_message("appending triplet2: " + str(triplet2) )
          sent_triplets_2.append(triplet2)
          row1 = [int(binary[i]), int(binary[i+1]), int(binary[i+2])]
          row2 = [int(binary[i+3]), int(binary[i+4]), int(binary[i+5])]
          self.debug.info_message("row: " + str(row1) )
          self.debug.info_message("row: " + str(row2) )
          bit_triplets1.append(row1)
          bit_triplets2.append(row2)


  """
  def displayTextFromIntlist(self, decoded_intlist_1, decoded_intlist_2):
    self.debug.info_message("displayTextFromIntlist" )

    try:
      binary_array_post_fec = []

      if self.osmod.FEC != ocn.FEC_NONE:
        self.debug.info_message("processing FEC" )
        binary_string = ''
        for int_low, int_high in zip(decoded_intlist_1, decoded_intlist_2):
          #char = self.b64_charfromindex_list[(int_low*8) + (int_high)]
          index = (int_low*8) + (int_high)
          binary = format(index, "06b")[0:6]
          self.debug.info_message("binary : " + str(binary) )
          binary_string = binary_string + binary

        self.debug.info_message("binary_string : " + str(binary_string) )
        binary_array_pre_fec = np.fromstring(binary_string, 'u1') - ord('0')
        self.debug.info_message("binary_array_pre_fec : " + str(binary_array_pre_fec) )
        """LDPC code goes here """
        #binary_array_post_ldpc = binary_array_pre_ldpc

        if self.osmod.chunk_num == 0:
          binary_array_post_fec = self.osmod.fec.decodeFEC(binary_array_pre_fec[self.osmod.extrapolate_seqlen * 6:])
          binary_array_post_fec = np.append(binary_array_pre_fec[:self.osmod.extrapolate_seqlen * 6], binary_array_post_fec)
        else:
          binary_array_post_fec = self.osmod.fec.decodeFEC(binary_array_pre_fec)

        #binary_array_post_ldpc = self.osmod.ldpc.decodeLDPC(binary_array_pre_ldpc)

        self.debug.info_message("binary_array_post_fec : " + str(binary_array_post_fec) )
        post_binary_string = "".join(binary_array_post_fec.astype(str))
        self.debug.info_message("post_binary_string : " + str(post_binary_string) )

        self.osmod.form_gui.window['ml_txrx_recvtext'].print("  decoded FEC: ", end="", text_color='black', background_color = 'white')
        for six_bits_index in range(0, len(post_binary_string), 6):
          index = int(post_binary_string[six_bits_index:six_bits_index+6], 2)
          #int_low = int(post_binary_string[0:3])
          #int_high = int(post_binary_string[3:6])
          char = self.b64_charfromindex_list[index]
          self.debug.info_message("found char: " + str(char))
          self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(char), end="", text_color='black', background_color = 'white')

      else:
        self.osmod.form_gui.window['ml_txrx_recvtext'].print("  decoded: ", end="", text_color='black', background_color = 'white')
        for int_low, int_high in zip(decoded_intlist_1, decoded_intlist_2):
          char = self.b64_charfromindex_list[(int_low*8) + (int_high)]
          self.debug.info_message("found char: " + str(char))
          self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(char), end="", text_color='black', background_color = 'white')

    except:
      sys.stdout.write("Exception in displayTextFromIntlist: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return binary_array_post_fec


    """
    
      for i in range(0, len(decoded_bitstring_1)):
        lookup_string = str(decoded_bitstring_1[i]) + str(decoded_bitstring_2[i])
        binary = int(lookup_string, 2)
        char = self.b64_charfromindex_list[binary]
        self.debug.info_message("found char: " + str(char))
        self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(char), end="", text_color='black', background_color = 'white')

      return decoded_bitstring_1, decoded_bitstring_2
   
    
    
    """




  def extractPhaseValuesWithOffsetDouble(self, signal1, signal2, median_block_offset, frequency, num_waves, where):
    self.debug.info_message("extractPhaseValuesWithOffset" )

    """ rebase the phases just prior to extraction"""
    max1 = np.max(np.abs(signal1))
    max2 = np.max(np.abs(signal2))
    ratio = max(max1, max2) / self.osmod.parameters[3]
    signal1 = signal1/ratio
    signal2 = signal2/ratio


    decoded_values1 = []
    decoded_values2 = []
    try:
      num_points = int(self.osmod.sample_rate / frequency)
      term1 = num_points*num_waves
      for x in range(median_block_offset, len(signal1), self.osmod.symbol_block_size):
        if x <= len(signal1) - self.osmod.symbol_block_size:
          middle = x + where
          decoded_values1.append(sum(signal1[middle-(term1):middle+(term1)-1]) / (2*term1))
          decoded_values2.append(sum(signal2[middle-(term1):middle+(term1)-1]) / (2*term1))

    except:
      sys.stdout.write("Exception in extractPhaseValuesWithOffset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return [decoded_values1, decoded_values2]



  """ This method works fine"""
  def downconvert8pskToBaseband(self, carrier_frequency, signal, phase_offset):
    t = np.arange(len(signal)) / self.osmod.sample_rate
    carrier = np.exp(-1j * (2 * np.pi * carrier_frequency * t + phase_offset))
    downconverted_signal = signal * carrier
    return downconverted_signal



  def recoverBasebandSignalOptimized(self, frequency, signal):
    self.debug.info_message("recoverBasebandSignalOptimized")

    try:
      if self.osmod.use_compiled_c_code == True:
        self.debug.info_message("calling compiled C code")

        t = np.arange(len(signal)) / self.osmod.sample_rate
        recovered_signal1a  = np.zeros_like(signal)
        recovered_signal1b  = np.zeros_like(signal)
        phase_error_history = np.zeros_like(signal)

        c_signal             = ptoc_double_array(signal)
        c_recovered_signal1a = ptoc_double_array(recovered_signal1a)
        c_recovered_signal1b = ptoc_double_array(recovered_signal1b)
        c_frequency          = ptoc_float(frequency)
        c_t                  = ptoc_double_array(t)

        costas_override = self.osmod.form_gui.window['cb_override_costasloop'].get()
        if costas_override:
          c_costas_damping_factor = ptoc_double(np.float64(self.osmod.form_gui.window['in_costasloop_dampingfactor'].get()))
          c_costas_loop_bandwidth = ptoc_double(np.float64(self.osmod.form_gui.window['in_costasloop_loopbandwidth'].get()))
          c_costas_K1             = ptoc_double(np.float64(self.osmod.form_gui.window['in_costasloop_K1'].get()))
          c_costas_K2             = ptoc_double(np.float64(self.osmod.form_gui.window['in_costasloop_K2'].get()))
        else:
          c_costas_damping_factor = ptoc_double(np.float64(self.osmod.parameters[6]))
          c_costas_loop_bandwidth = ptoc_double(np.float64(self.osmod.parameters[7]))
          c_costas_K1             = ptoc_double(np.float64(self.osmod.parameters[8]))
          c_costas_K2             = ptoc_double(np.float64(self.osmod.parameters[9]))

        self.debug.info_message("dtype: "   + str(signal.dtype))
        self.debug.info_message("value 1: " + str(signal[0]))
        self.debug.info_message("value 2: " + str(signal[1]))
        self.debug.info_message("value 3: " + str(signal[2]))
        self.debug.info_message("value 4: " + str(signal[3]))
        self.debug.info_message("value 5: " + str(signal[4]))

        self.osmod.compiled_lib.costas_loop_8psk.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_float, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.osmod.compiled_lib.costas_loop_8psk(c_signal, c_recovered_signal1a, c_recovered_signal1b, c_frequency, c_t, signal.size, c_costas_damping_factor, c_costas_loop_bandwidth, c_costas_K1, c_costas_K2)

        return [recovered_signal1a, recovered_signal1b], phase_error_history

      else:
        return self.recoverBasebandSignalPythonCode(frequency, signal)

    except:
      self.debug.error_message("Exception in recoverBasebandSignalOptimized: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def convertToIntMerge(self, samples1_normal, samples1_rotate, samples2_normal, samples2_rotate, threshold, offset):
    int_values1 = []
    int_values2 = []
    translate_a = {'0:1':-1, '-1:1':-1, '-1:0':-1, '-1:-1':0, '1:1':0,  '1:0':1,  '1:-1':1, '0:-1':1, '0:0':0  }
    translate_b = {'0:1':-1, '-1:1':0 , '-1:0':1,  '-1:-1':1, '1:1':-1, '1:0':-1, '1:-1':0, '0:-1':1, '0:0':0   }
    for i, j, k, l in zip(samples1_normal, samples1_rotate, samples2_normal, samples2_rotate):
      if i > offset + threshold:
        temp1 = 1
      elif i < (-1*threshold)+offset:
        temp1 = -1
      else:
        temp1 = 0

      if k > offset + threshold:
        temp2 = 1
      elif k < (-1*threshold)+offset:
        temp2 = -1
      else:
        temp2 = 0

      """ need to pull these values from second stream"""
      if (temp1 == 1 and temp2 == 1) or (temp1 == 1 and temp2 == 0) or (temp1 == 1 and temp2 == -1) or (temp1 == 0 and temp2 == -1) :
        if j > offset + threshold:
          temp1 = 1
        elif j < (-1*threshold)+offset:
          temp1 = -1
        else:
          temp1 = 0

        if l > offset + threshold:
          temp2 = 1
        elif l < (-1*threshold)+offset:
          temp2 = -1
        else:
          temp2 = 0

        #int_values1.append(temp1)
        #int_values2.append(temp2)
        int_values1.append(translate_a[str(temp1)+':'+str(temp2)])
        int_values2.append(translate_b[str(temp1)+':'+str(temp2)])
      else:
        int_values1.append(translate_a[str(temp1)+':'+str(temp2)])
        int_values2.append(translate_b[str(temp1)+':'+str(temp2)])

    return int_values1, int_values2



  """ This method works fine"""
  def recoverBasebandSignalPythonCode(self, frequency, signal):
    self.debug.info_message("recoverBasebandSignalOptimized")

    try:
      t = np.arange(len(signal)) / self.osmod.sample_rate
      phase_estimate = 0
      frequency_estimate = 0
      phase_estimate = 0.0
      damping_factor = 1.0 / np.sqrt(2)
      loop_bandwidth = 10
      K1 = 2 * damping_factor * loop_bandwidth 
      K2 = loop_bandwidth**2 
      recovered_signal1a  = np.zeros_like(signal)
      recovered_signal1b  = np.zeros_like(signal)
      phase_error_history = np.zeros_like(signal)

      term5 = np.pi * 2 / 8
      term6 = 2 * np.pi
      for i, sample in enumerate(signal):
        term1 = term6 * (frequency + frequency_estimate) * t[i]
        term3 = term1 + phase_estimate
        term4 = term3 + term5
        recovered_signal1a[i] = sample * self.osmod.cosRad(term3)
        recovered_signal1b[i] = sample * self.osmod.sinRad(term3)
        recovered_signal2a    = sample * self.osmod.cosRad(term4)
        recovered_signal2b    = sample * self.osmod.sinRad(term4)
        recovered_signal2     = recovered_signal2a - recovered_signal2b

        I   = recovered_signal1a[i]
        Q   = recovered_signal1b[i]
        I45 = np.real(recovered_signal2)
        Q45 = np.imag(recovered_signal2)

        phase_normal = I * Q
        phase_45     = I45 * Q45
        phase_error  = (phase_normal) if (abs(phase_normal) < abs(phase_45)) else (phase_45)
        frequency_estimate += K2 * phase_error 
        phase_estimate += K1 * phase_error + frequency_estimate

      max1 = np.max(np.abs(recovered_signal1a))
      max2 = np.max(np.abs(recovered_signal1b))
      ratio = max(max1, max2) / self.osmod.parameters[3]
      recovered_signal1a = recovered_signal1a/ratio
      recovered_signal1b = recovered_signal1b/ratio

    except:
      self.debug.error_message("Exception in recoverBasebandSignalOptimized: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return [recovered_signal1a, recovered_signal1b], phase_error_history

 
  
  def decoder_8psk_callback(self, multi_block, window):
    self.debug.info_message("decoder_8psk_callback")
    output = self.demod_costas_3_8psk(multi_block, window)

    return



  def findPulseStartIndex(self, signal):
    self.debug.info_message("findPulseStartIndex" )

    try:
      if self.osmod.use_compiled_c_code == True:
        self.debug.info_message("calling compiled C code")

        c_signal             = ptoc_double_array(signal)

        self.osmod.compiled_lib.find_pulse_start_index.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.osmod.compiled_lib.find_pulse_start_index.restype = ctypes.c_int
        start_index = self.osmod.compiled_lib.find_pulse_start_index(c_signal, signal.size, self.osmod.parameters[5], self.osmod.symbol_block_size, self.osmod.pulses_per_block)

        return start_index
      else:
        return self.findPulseStartIndexInterpretedPython(signal)
    except:
      sys.stdout.write("Exception in findPulseStartIndex: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  """ This method returns the index of the most frequently occurring peak (not the start of the pulse!!)"""
  def findPulseStartIndexInterpretedPython(self, signal):
    self.debug.info_message("findPulseStartIndex" )
    try:
      all_list = []

      pulse_width = self.osmod.symbol_block_size / self.osmod.pulses_per_block

      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        test_min = np.min(test_peak)
        max_indices = np.where((test_peak*(100/test_max)) > self.osmod.parameters[5])
        min_indices = np.where((test_peak*(100/test_min)) > self.osmod.parameters[5])
        self.debug.info_message("RRC max indices: " + str(max_indices[0]))
        self.debug.info_message("RRC min indices: " + str(min_indices[0]))
        for x in range(0, len(max_indices[0]) ):
          all_list.append(max_indices[0][x] % pulse_width)
        for x in range(0, len(min_indices[0]) ):
          all_list.append(min_indices[0][x] % pulse_width)
      self.debug.info_message("RRC all indices: " + str(all_list))

      median_index_all = int(np.median(np.array(all_list)))
      self.debug.info_message("RRC Filter mean all index: " + str(median_index_all))

      sample_start = median_index_all
      self.debug.info_message("sample_start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in findPulseStartIndex: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return int(sample_start % pulse_width)


  """ returns the most frequently occuring value in input data list that is not in items_to_ignore list """
  def count_max_occurrences(self, data, items_to_ignore):
    #self.debug.info_message("count_max_occurrences" )
    try:
      if items_to_ignore == []:
        counts = Counter(data)
      else:
        counts = Counter(item for item in data if item not in items_to_ignore)

      if not counts:
        return []

      max_count = max(counts.values())
      #self.debug.info_message("max_counts: " + str({key: value for key, value in counts.items() if value == max_count}) )

      return_value = [key for key, value in counts.items() if value == max_count]

    except:
      sys.stdout.write("Exception in count_max_occurrences: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return return_value





  def receive_pre_filters_filter_wave(self, pulse_start_index, audio_block, frequency):

    """ calculate receive side RRC filter"""
    pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

    """ apply receive side RRC filter"""
    for block_count in range(0, int(len(audio_block) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(audio_block)):
          audio_block[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] = audio_block[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main

    self.osmod.getDurationAndReset('apply RRC to wave')

    """ fft bandpass filter"""
    fft_filtered_lower, masked_fft_lower   = self.bandpass_filter_fft(audio_block, frequency[0] + self.osmod.fft_filter[0], frequency[0] + self.osmod.fft_filter[1])
    fft_filtered_higher, masked_fft_higher = self.bandpass_filter_fft(audio_block, frequency[1] + self.osmod.fft_filter[2], frequency[1] + self.osmod.fft_filter[3] )

    self.osmod.getDurationAndReset('bandpass_filter_fft in receive_pre_filters_filter_wave')

    return fft_filtered_lower, fft_filtered_higher


  def receive_post_filters_filter_wave(self, pulse_start_index, audio_block, frequency, fine_tune_adjust):

    """ calculate receive side RRC filter"""
    pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

    audio_block_copy = audio_block.copy()

    """ apply receive side RRC filter"""
    fine_tune_pulse_start_index = (pulse_start_index + fine_tune_adjust[0]) % pulse_length
    #fine_tune_pulse_start_index = pulse_start_index
    for block_count in range(0, int(len(audio_block) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + fine_tune_pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(audio_block)):
          audio_block[offset + fine_tune_pulse_start_index+(pulse_count * pulse_length):offset + fine_tune_pulse_start_index + ((pulse_count+1) * pulse_length)] = audio_block[offset + fine_tune_pulse_start_index+(pulse_count * pulse_length):offset + fine_tune_pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main
    """ apply receive side RRC filter"""
    fine_tune_pulse_start_index = (pulse_start_index + fine_tune_adjust[1]) % pulse_length
    #fine_tune_pulse_start_index = pulse_start_index
    for block_count in range(0, int(len(audio_block_copy) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + fine_tune_pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(audio_block_copy)):
          audio_block_copy[offset + fine_tune_pulse_start_index+(pulse_count * pulse_length):offset + fine_tune_pulse_start_index + ((pulse_count+1) * pulse_length)] = audio_block_copy[offset + fine_tune_pulse_start_index+(pulse_count * pulse_length):offset + fine_tune_pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main
    self.osmod.getDurationAndReset('apply RRC to wave')

    """ fft bandpass filter"""
    fft_filtered_lower, masked_fft_lower   = self.bandpass_filter_fft(audio_block, frequency[0] + self.osmod.fft_filter[0], frequency[0] + self.osmod.fft_filter[1])
    fft_filtered_higher, masked_fft_higher = self.bandpass_filter_fft(audio_block_copy, frequency[1] + self.osmod.fft_filter[2], frequency[1] + self.osmod.fft_filter[3] )

    self.osmod.getDurationAndReset('bandpass_filter_fft in receive_post_filters_filter_wave')

    return fft_filtered_lower, fft_filtered_higher


  def receive_post_filters_filter_wave_NOT_USED(self, pulse_start_index, audio_block, frequency, fine_tune_adjust):

    """ calculate receive side RRC filter"""
    pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

    self.osmod.getDurationAndReset('apply RRC to wave')

    """ fft bandpass filter"""
    fft_filtered_lower, masked_fft_lower   = self.bandpass_filter_fft(audio_block, frequency[0] + self.osmod.fft_filter[0], frequency[0] + self.osmod.fft_filter[1])
    fft_filtered_higher, masked_fft_higher = self.bandpass_filter_fft(audio_block, frequency[1] + self.osmod.fft_filter[2], frequency[1] + self.osmod.fft_filter[3] )

    """ apply receive side RRC filter"""
    for block_count in range(0, int(len(fft_filtered_lower) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(fft_filtered_lower)):
          fft_filtered_lower[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] = fft_filtered_lower[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main
    """ apply receive side RRC filter"""
    for block_count in range(0, int(len(fft_filtered_higher) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(fft_filtered_higher)):
          fft_filtered_higher[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] = fft_filtered_higher[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main

    self.osmod.getDurationAndReset('bandpass_filter_fft in receive_post_filters_filter_wave')

    return fft_filtered_lower, fft_filtered_higher


  def receive_pre_filters_average_data(self, pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher):
    self.debug.info_message("receive_pre_filters_average_data")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)


      audio_block1[0:pulse_start_index] = np.zeros(pulse_start_index)
      audio_block2[0:pulse_start_index] = np.zeros(pulse_start_index)


      """ average each frequency block"""
      num_full_blocks = int((len(audio_block1) - pulse_start_index) // self.osmod.symbol_block_size)
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        """ first frequency"""
        lower_pulse = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        pulse_count = 0
        for index in interpolated_lower: 
          if True:
            pulse_count = pulse_count + 1
            lower_pulse = lower_pulse + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)]
        lower_pulse = lower_pulse / pulse_count

        """ second frequency"""
        higher_pulse = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        pulse_count = 0
        for index in interpolated_higher: 
          if True:
            pulse_count = pulse_count + 1
            higher_pulse = higher_pulse + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)]
        higher_pulse = higher_pulse / pulse_count

        """write the data back to the data stream """
        for i in range(0, self.osmod.pulses_per_block): 
          if i in interpolated_lower:
            audio_block1[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = lower_pulse
          else:
            audio_block1[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = np.zeros_like(lower_pulse)

          if i in interpolated_higher:
            audio_block2[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = higher_pulse
          else:
            audio_block2[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = np.zeros_like(higher_pulse)

      #max1 = np.max(np.abs(audio_block1))
      #max2 = np.max(np.abs(audio_block2))
      #ratio = max(max1, max2) / self.osmod.parameters[3]
      #audio_block1 = audio_block1/ratio
      #audio_block2 = audio_block2/ratio


      self.debug.info_message("len(audio_block): " + str(len(audio_block1)))
      last_valid = 0
      for remainder_data in range((num_full_blocks * self.osmod.pulses_per_block * pulse_length) + pulse_start_index, len(audio_block1)-pulse_length, pulse_length): 
        self.debug.info_message("remainder_data: " + str(remainder_data))
        audio_block1[remainder_data:remainder_data + pulse_length] = lower_pulse
        audio_block2[remainder_data:remainder_data + pulse_length] = higher_pulse
        last_valid = remainder_data + pulse_length

    except:
      self.debug.error_message("Exception in receive_pre_filters_average_data: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return [audio_block1, audio_block2], pulse_start_index 


  def initialPulseTrainProcessing(self, audio_array, frequency):
    self.debug.info_message("initialPulseTrainProcessing")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      def acquire_pulse_train_offsets(audio_block_index, accuracy):
        nonlocal productCount

        pulse_a = audio_array[offset + (index * pulse_length):offset + ((index+1) * pulse_length)]
        productCount = productCount + 1
        return self.alignPulseTrain([audio_array, audio_array], frequency, audio_block_index, pulse_a, offset + (index * pulse_length), accuracy)

      aligh_type = ocn.ALIGN_RETAIN_LOCATION
      #aligh_type = ocn.ALIGN_MOVE_TO_MID

      pulse_a_modified = np.zeros(pulse_length, dtype = audio_array.dtype)
      audio_array_copy = np.zeros(len(audio_array), dtype = audio_array.dtype)

      """ average each frequency block"""
      self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }

      num_full_blocks = int((len(audio_array) ) // self.osmod.symbol_block_size)
      self.non_pulse = []

      """ locate start of pulses """
      found_start = False
      block_count = 0
      start_block = 0
      start_pulse = 0
      while found_start == False and block_count < num_full_blocks:
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        self.pulse_train_offsets = []
        self.pulse_train_offsets_mid = []
        productCount = 0
        for index in range(0,self.osmod.pulses_per_block): 
          is_non_pulse = acquire_pulse_train_offsets(0, 25)
          if is_non_pulse:
            self.non_pulse.append(index)
          else:
            found_start = True
            start_block = block_count
            start_pulse = index
            break
        block_count = block_count + 1

      self.debug.info_message("self.non_pulse: " + str(self.non_pulse))
      self.debug.info_message("start_block: " + str(start_block))
      self.debug.info_message("start_pulse: " + str(start_pulse))

      start = (start_pulse * pulse_length) + (start_block * self.osmod.pulses_per_block * pulse_length)
      num_full_blocks = int((len(audio_array) - start ) // self.osmod.symbol_block_size)
      self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }

      self.debug.info_message("start: " + str(start))

      """ identify peaks over full signal sample """
      num_pulses = 0
      for block_count in range(0, num_full_blocks): 
        offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) + (start_pulse * pulse_length)

        self.pulse_train_alignment_struct['blocks'].append([])
        self.pulse_train_offsets = []
        self.pulse_train_offsets_mid = []
        productCount = 0
        for index in range(0,self.osmod.pulses_per_block): 
          is_non_pulse = acquire_pulse_train_offsets(0, 10)
          if is_non_pulse:
            self.non_pulse.append(index)
          else:
            num_pulses = num_pulses + 1

        if aligh_type == ocn.ALIGN_RETAIN_LOCATION:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets
        elif aligh_type == ocn.ALIGN_MOVE_TO_MID:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets_mid

      self.debug.info_message("pulse_train_alignment_struct: " + str(self.pulse_train_alignment_struct))

      median_index = self.getMode(self.pulse_train_alignment_struct['pulses'])
      self.debug.info_message("median_index: " + str(median_index))

      """ identify strongest of the set of peaks from previous step """
      sigma_template = 7
      test_signal = gaussian_filter(np.abs(audio_array[start:]), sigma=sigma_template)


      sum_points = []
      for location in range(0, pulse_length): 
        sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % pulse_length == location])
        sum_points.append(sum_at_location)

      self.debug.info_message("sum_points: " + str(sum_points))
      self.debug.info_message("max sum_points: " + str(np.max(sum_points)))
      self.debug.info_message("min sum_points: " + str(np.min(sum_points)))

      index_max = np.where(sum_points == np.max(sum_points))[0]
      index_min = np.where(sum_points == np.min(sum_points))[0]
      self.debug.info_message("index_max: " + str(index_max))
      self.debug.info_message("index_min: " + str(index_min))

      self.debug.info_message("diff: " + str(index_max - index_min))



      diff_array = np.array([0] * (len(test_signal) - pulse_length), dtype = np.float32) #np.array(len(test_signal) - pulse_length)
      for i in range(0, len(test_signal) - pulse_length):
        diff_array[i] = test_signal[i+77] - test_signal[i]


      sum_points = []
      for location in range(0, pulse_length): 
        sum_at_location = np.sum(diff_array[np.arange(len(diff_array)) % pulse_length == location])
        sum_points.append(sum_at_location)

      self.debug.info_message("sum_points: " + str(sum_points))
      self.debug.info_message("max sum_points: " + str(np.max(sum_points)))
      self.debug.info_message("min sum_points: " + str(np.min(sum_points)))

      test  = np.where((diff_array % pulse_length)  >= 0.99 * np.max(diff_array) ) [0]
      test2 = np.where((diff_array % pulse_length)  <= 0.99 * np.min(diff_array) ) [0]

      self.debug.info_message("test: " + str(test))
      self.debug.info_message("test2: " + str(test2))

      for i in range(0, 100):
        self.debug.info_message(" " + str(test[i] % pulse_length))



      sigma_diff = 7
      smoothed_diff_array = gaussian_filter(diff_array, sigma=sigma_diff)


      reversal_array_up = []
      reversal_array_down = []
      last_diff = smoothed_diff_array[0]
      if smoothed_diff_array[1] > last_diff:
        trend = ocn.DIFF_TREND_UP
      else:
        trend = ocn.DIFF_TREND_DOWN
      for i in range(1, len(smoothed_diff_array) ):
        change = smoothed_diff_array[i] - last_diff
        last_diff = smoothed_diff_array[i]
        if change < 0 and trend == ocn.DIFF_TREND_UP:
          reversal_array_down.append(i % (3*pulse_length))
          trend = ocn.DIFF_TREND_DOWN
        elif change > 0 and trend == ocn.DIFF_TREND_DOWN:
          reversal_array_up.append(i % (3*pulse_length))
          trend = ocn.DIFF_TREND_UP

      self.debug.info_message("reversal_array_up: " + str(reversal_array_up))
      self.debug.info_message("reversal_array_down: " + str(reversal_array_down))



      sum_points = []
      for location in range(0, pulse_length): 
        sum_at_location = np.sum(smoothed_diff_array[np.arange(len(smoothed_diff_array)) % pulse_length == location])
        sum_points.append(sum_at_location)

      self.debug.info_message("sum_points: " + str(sum_points))
      self.debug.info_message("max sum_points: " + str(np.max(sum_points)))
      self.debug.info_message("min sum_points: " + str(np.min(sum_points)))

      test  = np.where((smoothed_diff_array % pulse_length)  >= 0.99 * np.max(smoothed_diff_array) ) [0]
      test2 = np.where((smoothed_diff_array % pulse_length)  <= 0.99 * np.min(smoothed_diff_array) ) [0]

      self.debug.info_message("test: " + str(test))
      self.debug.info_message("test2: " + str(test2))

      for i in range(0, 100):
        self.debug.info_message(" " + str(test[i] % pulse_length))


    except:
      self.debug.error_message("Exception in initialPulseTrainProcessing: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def obtainSkewForPulseTrain(self, pulse_start_index, frequency, audio_block, interpolated_lower, interpolated_higher):
    self.debug.info_message("obtainSkewForPulseTrain")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      half = int(pulse_length / 2)

      converted_wave = [0] * 2

      if isinstance(audio_block[0], np.ndarray) and np.issubdtype(audio_block[0].dtype, np.complex128):
        converted_wave[0] = (audio_block[0].real + audio_block[0].imag ) / 2
      if isinstance(audio_block[1], np.ndarray) and np.issubdtype(audio_block[1].dtype, np.complex128):
        converted_wave[1] = (audio_block[1].real + audio_block[1].imag ) / 2

      self.pulse_history       = [0] * 2
      self.pulse_abc           = [0] * 3
      mode_pulse_a             = [0] * 2
      mode_pulse_b             = [0] * 2
      mode_pulse_c             = [0] * 2
      #self.pulse_train_history = [0] * 2

      #self.pulse_train_history[0] = []
      #self.pulse_train_history[1] = []

      num_full_blocks = int((len(converted_wave[0]) - pulse_start_index) // self.osmod.symbol_block_size)

      """ calculate for lower frequency """
      self.pulse_abc[0] = []
      self.pulse_abc[1] = []
      self.pulse_abc[2] = []

      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        self.pulse_history[0] = []
        self.pulse_history[1] = []

        for index in interpolated_lower: 
          self.centerPulsePeak(0, converted_wave[0], offset + pulse_start_index + (index * pulse_length), index % 3)

        #median_index = self.getMode(self.pulse_history[0])
        #self.debug.info_message("median_index low: " + str(median_index))
        #self.pulse_train_history[0].append(median_index)

      mode_pulse_a[0] = self.getMode(self.pulse_abc[0])
      mode_pulse_b[0] = self.getMode(self.pulse_abc[1])
      mode_pulse_c[0] = self.getMode(self.pulse_abc[2])


      """ calculate for higher frequency """
      self.pulse_abc[0] = []
      self.pulse_abc[1] = []
      self.pulse_abc[2] = []

      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        self.pulse_history[0] = []
        self.pulse_history[1] = []

        for index in interpolated_higher: 
          self.centerPulsePeak(1, converted_wave[1], offset + pulse_start_index + (index * pulse_length), index % 3)

        #median_index = self.getMode(self.pulse_history[1])
        #self.debug.info_message("median_index high: " + str(median_index))
        #self.pulse_train_history[1].append(median_index)

      mode_pulse_a[1] = self.getMode(self.pulse_abc[0])
      mode_pulse_b[1] = self.getMode(self.pulse_abc[1])
      mode_pulse_c[1] = self.getMode(self.pulse_abc[2])

      #self.debug.info_message("self.pulse_train_history[0]: " + str(self.pulse_train_history[0]))
      #self.debug.info_message("self.pulse_train_history[1]: " + str(self.pulse_train_history[1]))

      self.debug.info_message("mode_pulse_a[0]: " + str(mode_pulse_a[0]))
      self.debug.info_message("mode_pulse_b[0]: " + str(mode_pulse_b[0]))
      self.debug.info_message("mode_pulse_c[0]: " + str(mode_pulse_c[0]))
      self.debug.info_message("mode_pulse_a[1]: " + str(mode_pulse_a[1]))
      self.debug.info_message("mode_pulse_b[1]: " + str(mode_pulse_b[1]))
      self.debug.info_message("mode_pulse_c[1]: " + str(mode_pulse_c[1]))


      #self.debug.info_message("self.pulse_abc[0]: " + str(mode_pulse_a))
      #self.debug.info_message("self.pulse_abc[1]: " + str(mode_pulse_b))
      #self.debug.info_message("self.pulse_abc[2]: " + str(mode_pulse_c))

      rel_accuracy = 0.0
      #abs_accuracy = 4
      abs_accuracy = 6

      fine_tune_adjust    = [0] * 2
      fine_tune_adjust[0] = 0
      fine_tune_adjust[1] = 0
      sw_pulse            = [0] * 2
      sw_pulse[0]         = ocn.PULSE_A
      sw_pulse[1]         = ocn.PULSE_A

      debug_text_low_high = ['low', 'high']

      for low_high in range(0, 2):
        self.debug.info_message("processing: " + str(debug_text_low_high[low_high]))

        metric_1 = math.isclose(mode_pulse_a[low_high], mode_pulse_b[low_high], rel_tol = rel_accuracy, abs_tol = abs_accuracy)
        metric_2 = math.isclose(mode_pulse_b[low_high], mode_pulse_c[low_high], rel_tol = rel_accuracy, abs_tol = abs_accuracy)
        metric_3 = math.isclose(mode_pulse_c[low_high], mode_pulse_a[low_high], rel_tol = rel_accuracy, abs_tol = abs_accuracy)

        if metric_1 and not metric_2 and not metric_3:
          self.debug.info_message("pulse C is Standing Wave")
          fine_tune_adjust[low_high] = int(half - mode_pulse_c[low_high] + pulse_length) % pulse_length
          sw_pulse[low_high] = ocn.PULSE_C
        elif metric_2 and not metric_3 and not metric_1:
          self.debug.info_message("pulse A is Standing Wave")
          fine_tune_adjust[low_high] = int(half - mode_pulse_a[low_high] + pulse_length) % pulse_length
          sw_pulse[low_high] = ocn.PULSE_A
        elif metric_3 and not metric_1 and not metric_2:
          self.debug.info_message("pulse B is Standing Wave")
          fine_tune_adjust[low_high] = int(half - mode_pulse_b[low_high] + pulse_length) % pulse_length
          sw_pulse[low_high] = ocn.PULSE_B


      #fine_tune_adjust = int(half - ((mode_pulse_a + mode_pulse_b + mode_pulse_c) / 3) + pulse_length) % pulse_length
      self.debug.info_message("fine_tune_adjust[0]: " + str(fine_tune_adjust[0]))
      self.debug.info_message("fine_tune_adjust[1]: " + str(fine_tune_adjust[1]))
      self.debug.info_message("sw_pulse[0]: " + str(sw_pulse[0]))
      self.debug.info_message("sw_pulse[1]: " + str(sw_pulse[1]))

      return fine_tune_adjust, sw_pulse

    except:
      self.debug.error_message("Exception in obtainSkewForPulseTrain: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def alignPulseTrain(self, audio_block, frequency, low_hi_index, pulse, pulse_location, location_accuracy):

    #self.debug.info_message("alignPulseTrain")
    try:
      no_pulse = False

      if isinstance(pulse, np.ndarray) and np.issubdtype(pulse.dtype, np.complex128):
        converted_pulse = (pulse.real + pulse.imag ) / 2
        pulse = converted_pulse

      samples_per_wavelength = self.osmod.sample_rate / frequency[low_hi_index]
      #self.debug.info_message("samples_per_wavelength: " + str(samples_per_wavelength))
      required_frequency = self.osmod.sample_rate / int(samples_per_wavelength)
      #self.debug.info_message("required_frequency: " + str(required_frequency))

      def locateSignalPeaks(template, signal, factor):
        correlation = correlate(signal, template, mode='same')
        peak_threshold = factor * np.max(correlation)
        #peak_threshold = np.max(correlation)
        peak_indices = np.where(correlation > peak_threshold)[0]
        
        #peaks, _ = find_peaks(correlation, height = 0.9*np.max(correlation))
        #peaks, _ = find_peaks(correlation, height = 0.95*np.max(correlation))
        peaks, _ = find_peaks(correlation, height = np.max(correlation))

        return peak_indices, peaks


      """ first method for locating pulse center"""
      """ create an RRC pulse sample """
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      half = int(pulse_length / 2)

      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate
      term5 = 2 * np.pi * time
      term6 = term5 * frequency[low_hi_index]
      term8 = term6 
      symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
      rrc_pulse = symbol_wave1[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main

      sigma_template = 7
      template_wave     = rrc_pulse
      template_envelope = gaussian_filter(rrc_pulse, sigma=sigma_template)

      #self.debug.info_message("Test 1")
      peaks1, peaks2 = locateSignalPeaks(template_envelope, pulse, 0.99)
      #self.debug.info_message("peaks1: " + str(peaks1))
      #self.debug.info_message("peaks2: " + str(peaks2))

      #self.debug.info_message("Test 2")
      sigma_signal = 7
      peaks1, peaks2 = locateSignalPeaks(template_envelope, gaussian_filter(pulse, sigma=sigma_signal), 0.99)
      #self.debug.info_message("peaks1: " + str(peaks1))
      #self.debug.info_message("peaks2: " + str(peaks2))
      if len(peaks1) > 0:
        final_value_envelope = peaks1[0]
      else:
        no_pulse = True
        final_value_envelope = 0
      #self.debug.info_message("final_value_envelope: " + str(final_value_envelope))

      #self.debug.info_message("Test 3")
      peaks1, peaks2 = locateSignalPeaks(template_wave, pulse, 0.85)
      #self.debug.info_message("peaks1: " + str(peaks1))
      peaks_modulo = peaks1 % samples_per_wavelength
      #self.debug.info_message("peaks_modulo: " + str(peaks_modulo))
      peaks_modulo = peaks1 % int(samples_per_wavelength)
      #self.debug.info_message("peaks_modulo: " + str(peaks_modulo))

      #pulse_location_struct = {'four_points': [0,0,0,0], 'current_point_index':0, num_points_observed:0, 'locus': 0, 'diff': 0 }

      rel_accuracy = 0.0
      abs_accuracy = location_accuracy #6

      if len(self.pulse_train_alignment_struct['location_points']) == 0:
        self.pulse_train_alignment_struct['current_point_index'] = 0
        self.pulse_train_alignment_struct['location_points'].append(final_value_envelope)
        self.pulse_train_alignment_struct['locus'] = final_value_envelope
        self.pulse_train_alignment_struct['pulses'].append(final_value_envelope)
      else:
        found = False
        found_index = 0
        """ test direct and also with wrap around (use half pulse length offset)"""
        for i in range(0, len(self.pulse_train_alignment_struct['location_points'])):
          compare_1 = self.pulse_train_alignment_struct['location_points'][i]
          compare_2 = final_value_envelope
          if math.isclose(compare_1, compare_2, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
            found = True
            found_index = i
            break
        if found == False:
          for i in range(0, len(self.pulse_train_alignment_struct['location_points'])):
            compare_1 = (self.pulse_train_alignment_struct['location_points'][i] + half) % pulse_length
            compare_2 = (final_value_envelope + half) % pulse_length
            if math.isclose(compare_1, compare_2, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
              found = True
              found_index = i
              break

        if found:
          self.pulse_train_alignment_struct['current_point_index'] = found_index
          self.pulse_train_alignment_struct['locus'] = final_value_envelope
          self.pulse_train_alignment_struct['pulses'].append(final_value_envelope)
        else:
          self.pulse_train_alignment_struct['location_points'].append(final_value_envelope)
          self.pulse_train_alignment_struct['current_point_index'] = len(self.pulse_train_alignment_struct['location_points']) - 1
          self.pulse_train_alignment_struct['locus'] = final_value_envelope
          self.pulse_train_alignment_struct['pulses'].append(final_value_envelope)

      #self.pulse_train_offsets.append(final_value_envelope)
      self.pulse_train_offsets.append(0)
      mid_adjust = (half - final_value_envelope)
      self.pulse_train_offsets_mid.append(mid_adjust)

      #self.debug.info_message("peaks2: " + str(peaks2))


      #peaks_modulo = peaks % int(samples_per_wavelength)
      #self.debug.info_message("peaks_modulo: " + str(peaks_modulo))
      #mode = self.getMode(peaks_modulo)
      #self.debug.info_message("mode: " + str(mode))

      """ second method for locating pulse center
      sigma_value = 7
      #analytic_signal_1 = gaussian_filter(pulse, sigma=sigma_value)
      analytic_signal_1 = hilbert(pulse)
      envelope_1 = np.abs(analytic_signal_1)
      peak_max_1 = np.max(envelope_1)
      peak_min_1 = np.min(envelope_1)
      peak_index_max_1 = np.where(envelope_1 == peak_max_1)[0]
      peak_index_min_1 = np.where(envelope_1 == peak_min_1)[0]
      #self.debug.info_message("peak_index_max_1: " + str(peak_index_max_1))
      #self.debug.info_message("peak_index_min_1: " + str(peak_index_min_1))
      """

      #self.debug.info_message("return_value: " + str(return_value))

      #return (return_value + half) % pulse_length
      return no_pulse

    except:
      self.debug.error_message("Exception in alignPulseTrain: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      self.debug.info_message("i: " + str(i))
      self.debug.info_message("self.pulse_train_alignment_struct: " + str(self.pulse_train_alignment_struct))
      return no_pulse



  def downconvert_I3_RelExp(self, pulse_start_index, audio_block, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust):
    self.debug.info_message("downconvert_I3_RelExp")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      half = int(pulse_length / 2)
      time = np.arange(pulse_length) / self.osmod.sample_rate

      #downconvert_shift = 0.5
      downconvert_shift = self.osmod.downconvert_shift

      override_downconvert_shift = self.osmod.form_gui.window['cb_overridedownconvertshift'].get()
      if override_downconvert_shift:
        downconvert_shift = float(self.osmod.form_gui.window['in_downconvertshift'].get())
      shift_amount = int(downconvert_shift * pulse_length)

      complex_exponential_lower  = np.exp(-1j * 2 * np.pi * frequency[0] * time)
      complex_exponential_higher = np.exp(-1j * 2 * np.pi * frequency[1] * time)
      complex_exponential = [complex_exponential_lower, complex_exponential_higher]

      def multiply(sig1, sig2):
        return [cmath.rect(abs(c1) * abs(c2), cmath.phase(c1) + cmath.phase(c2)) for c1, c2 in zip(sig1, sig2)]

      def deriveProduct(low_hi_index, sig1, sig2):
        baseband_sig = complex_exponential[low_hi_index] * (sig1 + sig2)

        #baseband_sig = complex_exponential[low_hi_index] * sig1 + complex_exponential[low_hi_index] * sig2
        #baseband_sig = complex_exponential[low_hi_index] * sig2


        #baseband_sig = complex_exponential[low_hi_index] * sig2
        #return self.filter_low_pass(baseband_sig, 1200)


        return self.filter_low_pass(baseband_sig, frequency[low_hi_index] - 115)

        #filtered_signal, _ = self.lowpass_filter_fft(baseband_sig, frequency[low_hi_index] +100)
        #return filtered_signal



        #return self.filter_low_pass(baseband_sig, frequency[low_hi_index] - 110)
        #return baseband_sig



        #return baseband_sig
        #filtered_signal = self.lowpass_filter_fft(baseband_sig, 10000)
        #return ((filtered_signal.real + filtered_signal.imag ) / 2)

      def extract_process_writeback(audio_block_index):
        nonlocal has_last_pulse_a
        nonlocal has_last_pulse_b
        nonlocal has_last_pulse_c
        nonlocal last_pulse_a_location
        nonlocal last_pulse_b_location
        nonlocal last_pulse_c_location
        nonlocal max_a
        nonlocal max_b
        nonlocal max_c
        nonlocal last_pulse_a
        nonlocal last_pulse_b
        nonlocal last_pulse_c
        nonlocal last_pulse_a_modified
        nonlocal last_pulse_b_modified
        nonlocal last_pulse_c_modified
        nonlocal productCount

        if index % 3 == 0 and max_a < max_useable:
          if has_last_pulse_c == True:
            """ test code """
            #center_delta = self.pulse_train_alignment_struct['blocks'][block_count][productCount]
            center_delta = center
            pulse_a_modified[0:pulse_length] = audio_block_copy[audio_block_index][offset + center_delta + fine_tune_pulse_start_index + (index * pulse_length):offset + center_delta + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
            last_pulse_c_modified[0:pulse_length] = audio_block_copy[audio_block_index][last_pulse_c_location + center_delta:last_pulse_c_location + center_delta + pulse_length]
            max_a = max_a + 1
            product_pulse = deriveProduct(audio_block_index, pulse_a_modified, last_pulse_c_modified)
            productCount = productCount + 1
            audio_block[audio_block_index][offset + fine_tune_pulse_start_index+(index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)] = product_pulse
          last_pulse_a_location = offset + fine_tune_pulse_start_index + (index * pulse_length)
          has_last_pulse_a = True

        elif (index+2) % 3 == 0 and max_b < max_useable:
          if has_last_pulse_a == True:
            """ test code """
            #center_delta = self.pulse_train_alignment_struct['blocks'][block_count][productCount]
            center_delta = center
            pulse_b_modified[0:pulse_length] = audio_block_copy[audio_block_index][offset + center_delta + fine_tune_pulse_start_index + (index * pulse_length):offset + center_delta + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
            last_pulse_a_modified[0:pulse_length] = audio_block_copy[audio_block_index][last_pulse_a_location + center_delta:last_pulse_a_location + center_delta + pulse_length]
            max_b = max_b + 1
            product_pulse = deriveProduct(audio_block_index, pulse_b_modified, last_pulse_a_modified)
            productCount = productCount + 1
            audio_block[audio_block_index][offset + fine_tune_pulse_start_index+(index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)] = product_pulse
          last_pulse_b_location = offset + fine_tune_pulse_start_index + (index * pulse_length)
          has_last_pulse_b = True

        elif (index+1) % 3 == 0 and max_c < max_useable:
          if has_last_pulse_b == True:
            """ test code """
            #center_delta = self.pulse_train_alignment_struct['blocks'][block_count][productCount]
            center_delta = center

            pulse_c_modified[0:pulse_length] = audio_block_copy[audio_block_index][offset + center_delta + fine_tune_pulse_start_index + (index * pulse_length):offset + center_delta + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
            last_pulse_b_modified[0:pulse_length] = audio_block_copy[audio_block_index][last_pulse_b_location + center_delta:last_pulse_b_location + center_delta + pulse_length]
            max_c = max_c + 1
            product_pulse = deriveProduct(audio_block_index, pulse_c_modified, last_pulse_b_modified)
            productCount = productCount + 1
            audio_block[audio_block_index][offset + fine_tune_pulse_start_index+(index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)] = product_pulse
          last_pulse_c_location = offset + fine_tune_pulse_start_index + (index * pulse_length)
          has_last_pulse_c = True

        return



      def acquire_pulse_train_offsets(audio_block_index):
        nonlocal has_last_pulse_a
        nonlocal has_last_pulse_b
        nonlocal has_last_pulse_c
        nonlocal max_a
        nonlocal max_b
        nonlocal max_c
        nonlocal productCount

        if index % 3 == 0 and max_a < max_useable:
          pulse_a = audio_block[audio_block_index][offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
          if has_last_pulse_c == True:
            self.alignPulseTrain(audio_block, frequency, audio_block_index, pulse_a, offset + fine_tune_pulse_start_index + (index * pulse_length), 6)
            max_a = max_a + 1
            productCount = productCount + 1
          has_last_pulse_a = True

        elif (index+2) % 3 == 0 and max_b < max_useable:
          pulse_b = audio_block[audio_block_index][offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
          if has_last_pulse_a == True:
            self.alignPulseTrain(audio_block, frequency, audio_block_index, pulse_b, offset + fine_tune_pulse_start_index + (index * pulse_length), 6)
            max_b = max_b + 1
            productCount = productCount + 1
          has_last_pulse_b = True

        elif (index+1) % 3 == 0 and max_c < max_useable:
          pulse_c = audio_block[audio_block_index][offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)]
          if has_last_pulse_b == True:
            self.alignPulseTrain(audio_block, frequency, audio_block_index, pulse_c, offset + fine_tune_pulse_start_index + (index * pulse_length), 6)
            max_c = max_c + 1
            productCount = productCount + 1
          has_last_pulse_c = True

        return


      aligh_type = ocn.ALIGN_RETAIN_LOCATION
      #aligh_type = ocn.ALIGN_MOVE_TO_MID

      last_pulse_a_location = 0
      last_pulse_b_location = 0
      last_pulse_c_location = 0

      pulse_a_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)
      pulse_b_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)
      pulse_c_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)
      last_pulse_a_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)
      last_pulse_b_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)
      last_pulse_c_modified = np.zeros(pulse_length, dtype = audio_block[0].dtype)

      audio_block_copy = [0] * 2
      audio_block_copy[0] = np.zeros(len(audio_block[0]), dtype = audio_block[0].dtype)
      audio_block_copy[1] = np.zeros(len(audio_block[1]), dtype = audio_block[1].dtype)
      audio_block_copy[0][0:] = audio_block[0][0:]
      audio_block_copy[1][0:] = audio_block[1][0:]


      """ first frequency"""
      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_lower: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_lower = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable lower: " + str(max_useable_lower))

      """ second frequency"""
      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_higher: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_higher = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable higher: " + str(max_useable_higher))

      last_pulse_a = None
      last_pulse_b = None
      last_pulse_c = None


      """ average each frequency block"""
      self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }
      fine_tune_pulse_start_index = pulse_start_index #(pulse_start_index + fine_tune_adjust[0]) % pulse_length
      num_full_blocks = int((len(audio_block[0]) - fine_tune_pulse_start_index) // self.osmod.symbol_block_size)
      max_useable = max_useable_lower

      self.debug.info_message("LOC 1")

      #location_points': [97, 86, 6, 17]
      #ret_val = self.osmod.detector.detectStandingWavePulseNew(audio_block, frequency, 0, 0, ocn.LOCATE_PULSES_PEAK)
      #peak = ret_val[0]
      #self.debug.info_message("peak lower: " + str(peak))
      #center = half - peak

      #center = half - 97
      #center = 0

      center = shift_amount
      #center = half

      """
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        max_a = 0
        max_b = 0
        max_c = 0
        has_last_pulse_a = False
        has_last_pulse_b = False
        has_last_pulse_c = False
        self.pulse_train_alignment_struct['blocks'].append([])
        self.pulse_train_offsets = []
        self.pulse_train_offsets_mid = []
        productCount = 0
        for index in interpolated_lower: 
          acquire_pulse_train_offsets(0)
        for index in interpolated_lower: 
          if productCount < max_useable_lower * 3:
            acquire_pulse_train_offsets(0)

        if aligh_type == ocn.ALIGN_RETAIN_LOCATION:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets
        elif aligh_type == ocn.ALIGN_MOVE_TO_MID:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets_mid

      pulse_train_offsets_lower = self.pulse_train_alignment_struct
      self.debug.info_message("pulse_train_alignment_struct lower: " + str(self.pulse_train_alignment_struct))
      """
      self.debug.info_message("LOC 2")

      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        max_a = 0
        max_b = 0
        max_c = 0
        has_last_pulse_a = False
        has_last_pulse_b = False
        has_last_pulse_c = False
        productCount = 0
        for index in interpolated_lower: 
          extract_process_writeback(0)
        for index in interpolated_lower: 
          if productCount < max_useable_lower * 3:
            extract_process_writeback(0)
        
          #else:
          #  break

      self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }
      fine_tune_pulse_start_index = pulse_start_index #(pulse_start_index + fine_tune_adjust[1]) % pulse_length
      num_full_blocks = int((len(audio_block[1]) - fine_tune_pulse_start_index) // self.osmod.symbol_block_size)
      max_useable = max_useable_higher

      self.debug.info_message("LOC 3")


      #location_points': [99, 91, 10]
      #ret_val = self.osmod.detector.detectStandingWavePulseNew(audio_block, frequency, 0, 1, ocn.LOCATE_PULSES_PEAK)
      #peak = ret_val[0]
      #self.debug.info_message("peak higher: " + str(peak))
      #center = half - peak

      #center = half - 99
      #center = 0

      center = shift_amount
      #center = half

      """
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        max_a = 0
        max_b = 0
        max_c = 0
        has_last_pulse_a = False
        has_last_pulse_b = False
        has_last_pulse_c = False
        self.pulse_train_alignment_struct['blocks'].append([])
        self.pulse_train_offsets = []
        self.pulse_train_offsets_mid = []
        productCount = 0
        for index in interpolated_higher: 
          acquire_pulse_train_offsets(1)
        for index in interpolated_higher: 
          if productCount < max_useable_higher * 3:
            acquire_pulse_train_offsets(1)

        if aligh_type == ocn.ALIGN_RETAIN_LOCATION:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets
        elif aligh_type == ocn.ALIGN_MOVE_TO_MID:
          self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets_mid

      pulse_train_offsets_higher = self.pulse_train_alignment_struct
      self.debug.info_message("pulse_train_alignment_struct higher: " + str(self.pulse_train_alignment_struct))
      """

      self.debug.info_message("LOC 4")

      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length
        max_a = 0
        max_b = 0
        max_c = 0
        has_last_pulse_a = False
        has_last_pulse_b = False
        has_last_pulse_c = False
        productCount = 0
        for index in interpolated_higher: 
          extract_process_writeback(1)
        for index in interpolated_higher: 
          if productCount < max_useable_higher * 3:
            extract_process_writeback(1)


            
          #else:
          #  break

      fine_tune_pulse_start_index_lower  = (pulse_start_index + fine_tune_adjust[0]) % pulse_length
      fine_tune_pulse_start_index_higher = (pulse_start_index + fine_tune_adjust[1]) % pulse_length
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        for i in range(0, self.osmod.pulses_per_block): 
          if not (i in interpolated_lower):
            audio_block[0][offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = np.zeros(pulse_length, dtype = audio_block[0].dtype)
          if not (i in interpolated_higher):
            audio_block[1][offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = np.zeros(pulse_length, dtype = audio_block[0].dtype)


    except:
      self.debug.error_message("Exception in downconvert_I3_RelExp: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      self.debug.info_message("block_count: " + str(block_count))
      self.debug.info_message("index: " + str(index))


  """
  Analysis of the LB28 modes shows that calculation of the standing wave further is further complicated by the presence of a travelling wave
  The travelling wave moves along the pulse train and alters the peak max points to give false readings for where the standing wave is located
  To resolve this, concensus values across the entire 30 character sequqence of blocks must be obtained first.
  This method analyzes the maximum pulse and provides a count of the three A B and C
  """  
  def peak_pulse_quantifier(self, pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher):
    self.debug.info_message("peak_pulse_quantifier")
    try:
      peak_pulse_count_lower  = [0,0,0]
      peak_pulse_count_higher = [0,0,0]
      peak_pulse_count_total  = [0,0,0]

      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

      audio_block1[0:pulse_start_index] = np.zeros(pulse_start_index)
      audio_block2[0:pulse_start_index] = np.zeros(pulse_start_index)

      num_full_blocks = int((len(audio_block1) - pulse_start_index) // self.osmod.symbol_block_size)
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        """ first frequency"""
        lower_pulse_all      = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_a        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_b        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_c        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_all_real = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_a_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_b_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_c_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_all_imag = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_a_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_b_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
        lower_pulse_c_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])

        max_a = 0
        max_b = 0
        max_c = 0
        for index in interpolated_lower: 
          if index % 3 == 0:
            max_a = max_a + 1
          elif (index+2) % 3 == 0:
            max_b = max_b + 1
          elif (index+1) % 3 == 0:
            max_c = max_c + 1
        max_useable_lower = min(min(max_a, max_b), max_c)
        self.debug.info_message("max_useable_lower low: " + str(max_useable_lower))

        max_a = 0
        max_b = 0
        max_c = 0
        pulse_count = 0
        for index in interpolated_lower: 
          pulse_count = pulse_count + 1
          add_pulse_to_all = False
          if index % 3 == 0 and max_a < max_useable_lower:
            lower_pulse_a_real = lower_pulse_a_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_a_imag = lower_pulse_a_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_a = max_a + 1
            add_pulse_to_all = True
          elif (index+2) % 3 == 0 and max_b < max_useable_lower:
            lower_pulse_b_real = lower_pulse_b_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_b_imag = lower_pulse_b_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_b = max_b + 1
            add_pulse_to_all = True
          elif (index+1) % 3 == 0 and max_c < max_useable_lower:
            lower_pulse_c_real = lower_pulse_c_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_c_imag = lower_pulse_c_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_c = max_c + 1
            add_pulse_to_all = True

          if add_pulse_to_all:
            lower_pulse_all_real = lower_pulse_all_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_all_imag = lower_pulse_all_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag

        lower_pulse_a   = lower_pulse_a_real + 1j   * lower_pulse_a_imag
        lower_pulse_b   = lower_pulse_b_real + 1j   * lower_pulse_b_imag
        lower_pulse_c   = lower_pulse_c_real + 1j   * lower_pulse_c_imag
        lower_pulse_all = lower_pulse_all_real + 1j * lower_pulse_all_imag

        max_a = np.max(np.abs(lower_pulse_a))
        max_b = np.max(np.abs(lower_pulse_b))
        max_c = np.max(np.abs(lower_pulse_c))
        if max_a >= max_b and max_a >= max_c:
          self.debug.info_message("MAX A is largest pulse lower")
          peak_pulse_count_lower[ocn.PULSE_A] = peak_pulse_count_lower[ocn.PULSE_A] + 1
          max_pulse_lower = lower_pulse_a
          product_pulse_lower1 = lower_pulse_b
          product_pulse_lower2 = lower_pulse_c
          product_pulse_lower3 = lower_pulse_b + lower_pulse_c
        elif max_b >= max_a and max_b >= max_c:
          self.debug.info_message("MAX B is largest pulse lower")
          peak_pulse_count_lower[ocn.PULSE_B] = peak_pulse_count_lower[ocn.PULSE_B] + 1
          max_pulse_lower = lower_pulse_b
          product_pulse_lower1 = lower_pulse_c
          product_pulse_lower2 = lower_pulse_a
          product_pulse_lower3 = lower_pulse_c + lower_pulse_a
        elif max_c >= max_a and max_c >= max_b:
          self.debug.info_message("MAX C is largest pulse lower")
          peak_pulse_count_lower[ocn.PULSE_C] = peak_pulse_count_lower[ocn.PULSE_C] + 1
          max_pulse_lower = lower_pulse_c
          product_pulse_lower1 = lower_pulse_a
          product_pulse_lower2 = lower_pulse_b
          product_pulse_lower3 = lower_pulse_a + lower_pulse_b
        

        """ second frequency"""
        higher_pulse_all      = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_a        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_b        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_c        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_all_real = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_a_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_b_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_c_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_all_imag = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_a_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_b_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
        higher_pulse_c_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])

        max_a = 0
        max_b = 0
        max_c = 0
        for index in interpolated_higher: 
          if index % 3 == 0:
            max_a = max_a + 1
          elif (index+2) % 3 == 0:
            max_b = max_b + 1
          elif (index+1) % 3 == 0:
            max_c = max_c + 1
        max_useable_higher = min(min(max_a, max_b), max_c)
        self.debug.info_message("max_useable_higher high: " + str(max_useable_higher))

        max_a = 0
        max_b = 0
        max_c = 0
        pulse_count = 0
        for index in interpolated_higher: 
          pulse_count = pulse_count + 1
          add_pulse_to_all = False
          if index % 3 == 0 and max_a < max_useable_higher:
            higher_pulse_a_real = higher_pulse_a_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_a_imag = higher_pulse_a_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_a = max_a + 1
            add_pulse_to_all = True
          elif (index+2) % 3 == 0 and max_b < max_useable_higher:
            higher_pulse_b_real = higher_pulse_b_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_b_imag = higher_pulse_b_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_b = max_b + 1
            add_pulse_to_all = True
          elif (index+1) % 3 == 0 and max_c < max_useable_higher:
            higher_pulse_c_real = higher_pulse_c_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_c_imag = higher_pulse_c_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_c = max_c + 1
            add_pulse_to_all = True

          if add_pulse_to_all:
            higher_pulse_all_real = higher_pulse_all_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_all_imag = higher_pulse_all_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag

        higher_pulse_a   = higher_pulse_a_real + 1j   * higher_pulse_a_imag
        higher_pulse_b   = higher_pulse_b_real + 1j   * higher_pulse_b_imag
        higher_pulse_c   = higher_pulse_c_real + 1j   * higher_pulse_c_imag
        higher_pulse_all = higher_pulse_all_real + 1j * higher_pulse_all_imag

        max_a = np.max(np.abs(higher_pulse_a))
        max_b = np.max(np.abs(higher_pulse_b))
        max_c = np.max(np.abs(higher_pulse_c))
        if max_a >= max_b and max_a >= max_c:
          self.debug.info_message("MAX A is largest pulse higher")
          peak_pulse_count_higher[ocn.PULSE_A] = peak_pulse_count_higher[ocn.PULSE_A] + 1
          max_pulse_higher = higher_pulse_a
          product_pulse_higher1 = higher_pulse_b
          product_pulse_higher2 = higher_pulse_c
          product_pulse_higher3 = higher_pulse_b + higher_pulse_c
        elif max_b >= max_a and max_b >= max_c:
          self.debug.info_message("MAX B is largest pulse higher")
          peak_pulse_count_higher[ocn.PULSE_B] = peak_pulse_count_higher[ocn.PULSE_B] + 1
          max_pulse_higher = higher_pulse_b
          product_pulse_higher1 = higher_pulse_c
          product_pulse_higher2 = higher_pulse_a
          product_pulse_higher3 = higher_pulse_c + higher_pulse_a
        elif max_c >= max_a and max_c >= max_b:
          self.debug.info_message("MAX C is largest pulse higher")
          peak_pulse_count_higher[ocn.PULSE_C] = peak_pulse_count_higher[ocn.PULSE_C] + 1
          max_pulse_higher = higher_pulse_c
          product_pulse_higher1 = higher_pulse_a
          product_pulse_higher2 = higher_pulse_b
          product_pulse_higher3 = higher_pulse_a + higher_pulse_b


      self.debug.info_message("MAX A pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_A]))
      self.debug.info_message("MAX B pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_B]))
      self.debug.info_message("MAX C pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_C]))
      self.debug.info_message("MAX A pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_A]))
      self.debug.info_message("MAX B pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_B]))
      self.debug.info_message("MAX C pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_C]))

      peak_pulse_count_total[ocn.PULSE_A]  = peak_pulse_count_lower[ocn.PULSE_A] + peak_pulse_count_higher[ocn.PULSE_A]
      peak_pulse_count_total[ocn.PULSE_B]  = peak_pulse_count_lower[ocn.PULSE_B] + peak_pulse_count_higher[ocn.PULSE_B]
      peak_pulse_count_total[ocn.PULSE_C]  = peak_pulse_count_lower[ocn.PULSE_C] + peak_pulse_count_higher[ocn.PULSE_C]

      #"""
      if peak_pulse_count_lower[ocn.PULSE_A] >= peak_pulse_count_lower[ocn.PULSE_B] and peak_pulse_count_lower[ocn.PULSE_A] >= peak_pulse_count_lower[ocn.PULSE_C]:
        peak_lower = ocn.PULSE_A
      elif peak_pulse_count_lower[ocn.PULSE_B] >= peak_pulse_count_lower[ocn.PULSE_A] and peak_pulse_count_lower[ocn.PULSE_B] >= peak_pulse_count_lower[ocn.PULSE_C]:
        peak_lower = ocn.PULSE_B
      elif peak_pulse_count_lower[ocn.PULSE_C] >= peak_pulse_count_lower[ocn.PULSE_A] and peak_pulse_count_lower[ocn.PULSE_C] >= peak_pulse_count_lower[ocn.PULSE_B]:
        peak_lower = ocn.PULSE_C

      if peak_pulse_count_higher[ocn.PULSE_A] >= peak_pulse_count_higher[ocn.PULSE_B] and peak_pulse_count_higher[ocn.PULSE_A] >= peak_pulse_count_higher[ocn.PULSE_C]:
        peak_higher = ocn.PULSE_A
      elif peak_pulse_count_higher[ocn.PULSE_B] >= peak_pulse_count_higher[ocn.PULSE_A] and peak_pulse_count_higher[ocn.PULSE_B] >= peak_pulse_count_higher[ocn.PULSE_C]:
        peak_higher = ocn.PULSE_B
      elif peak_pulse_count_higher[ocn.PULSE_C] >= peak_pulse_count_higher[ocn.PULSE_A] and peak_pulse_count_higher[ocn.PULSE_C] >= peak_pulse_count_higher[ocn.PULSE_B]:
        peak_higher = ocn.PULSE_C
      #"""
      """
      if peak_pulse_count_total[ocn.PULSE_A] >= peak_pulse_count_total[ocn.PULSE_B] and peak_pulse_count_total[ocn.PULSE_A] >= peak_pulse_count_total[ocn.PULSE_C]:
        peak_total = ocn.PULSE_A
      elif peak_pulse_count_total[ocn.PULSE_B] >= peak_pulse_count_total[ocn.PULSE_A] and peak_pulse_count_total[ocn.PULSE_B] >= peak_pulse_count_total[ocn.PULSE_C]:
        peak_total = ocn.PULSE_B
      elif peak_pulse_count_total[ocn.PULSE_C] >= peak_pulse_count_total[ocn.PULSE_A] and peak_pulse_count_total[ocn.PULSE_C] >= peak_pulse_count_total[ocn.PULSE_B]:
        peak_total = ocn.PULSE_C
      """

      return peak_lower, peak_higher
      #return peak_total, peak_total
      #return ocn.PULSE_C, ocn.PULSE_C

    except:
      self.debug.error_message("Exception in peak_pulse_quantifier: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def peak_pulse_quantifier2(self, pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher):
    self.debug.info_message("peak_pulse_quantifier2")
    try:
      peak_pulse_count_lower  = [0,0,0]
      peak_pulse_count_higher = [0,0,0]
      peak_pulse_count_total  = [0,0,0]

      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

      audio_block1[0:pulse_start_index] = np.zeros(pulse_start_index)
      audio_block2[0:pulse_start_index] = np.zeros(pulse_start_index)

      """ first frequency"""
      lower_pulse_all      = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_a        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_b        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_c        = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_all_real = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_a_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_b_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_c_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_all_imag = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_a_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_b_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_c_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])

      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_lower: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_lower = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable_lower low: " + str(max_useable_lower))

      max_a = 0
      max_b = 0
      max_c = 0
      #pulse_count = 0

      num_full_blocks = int((len(audio_block1) - pulse_start_index) // self.osmod.symbol_block_size)
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        for index in interpolated_lower: 
          #pulse_count = pulse_count + 1
          add_pulse_to_all = False
          if index % 3 == 0 and max_a < max_useable_lower:
            lower_pulse_a_real = lower_pulse_a_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_a_imag = lower_pulse_a_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_a = max_a + 1
            add_pulse_to_all = True
          elif (index+2) % 3 == 0 and max_b < max_useable_lower:
            lower_pulse_b_real = lower_pulse_b_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_b_imag = lower_pulse_b_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_b = max_b + 1
            add_pulse_to_all = True
          elif (index+1) % 3 == 0 and max_c < max_useable_lower:
            lower_pulse_c_real = lower_pulse_c_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_c_imag = lower_pulse_c_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_c = max_c + 1
            add_pulse_to_all = True

          if add_pulse_to_all:
            lower_pulse_all_real = lower_pulse_all_real + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            lower_pulse_all_imag = lower_pulse_all_imag + audio_block1[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag


      lower_pulse_a   = lower_pulse_a_real + 1j   * lower_pulse_a_imag
      lower_pulse_b   = lower_pulse_b_real + 1j   * lower_pulse_b_imag
      lower_pulse_c   = lower_pulse_c_real + 1j   * lower_pulse_c_imag
      lower_pulse_all = lower_pulse_all_real + 1j * lower_pulse_all_imag

      max_a = np.max(np.abs(lower_pulse_a))
      max_b = np.max(np.abs(lower_pulse_b))
      max_c = np.max(np.abs(lower_pulse_c))

      if max_a >= max_b and max_a >= max_c:
        self.debug.info_message("MAX A is largest pulse lower")
        peak_pulse_count_lower[ocn.PULSE_A] = peak_pulse_count_lower[ocn.PULSE_A] + 1
        max_pulse_lower = lower_pulse_a
        product_pulse_lower1 = lower_pulse_b
        product_pulse_lower2 = lower_pulse_c
        product_pulse_lower3 = lower_pulse_b + lower_pulse_c
      elif max_b >= max_a and max_b >= max_c:
        self.debug.info_message("MAX B is largest pulse lower")
        peak_pulse_count_lower[ocn.PULSE_B] = peak_pulse_count_lower[ocn.PULSE_B] + 1
        max_pulse_lower = lower_pulse_b
        product_pulse_lower1 = lower_pulse_c
        product_pulse_lower2 = lower_pulse_a
        product_pulse_lower3 = lower_pulse_c + lower_pulse_a
      elif max_c >= max_a and max_c >= max_b:
        self.debug.info_message("MAX C is largest pulse lower")
        peak_pulse_count_lower[ocn.PULSE_C] = peak_pulse_count_lower[ocn.PULSE_C] + 1
        max_pulse_lower = lower_pulse_c
        product_pulse_lower1 = lower_pulse_a
        product_pulse_lower2 = lower_pulse_b
        product_pulse_lower3 = lower_pulse_a + lower_pulse_b
        

      """ second frequency"""
      higher_pulse_all      = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_a        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_b        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_c        = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_all_real = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_a_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_b_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_c_real   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_all_imag = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_a_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_b_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_c_imag   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])

      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_higher: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_higher = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable_higher high: " + str(max_useable_higher))

      max_a = 0
      max_b = 0
      max_c = 0
      #pulse_count = 0

      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        for index in interpolated_higher: 
          #pulse_count = pulse_count + 1
          add_pulse_to_all = False
          if index % 3 == 0 and max_a < max_useable_higher:
            higher_pulse_a_real = higher_pulse_a_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_a_imag = higher_pulse_a_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_a = max_a + 1
            add_pulse_to_all = True
          elif (index+2) % 3 == 0 and max_b < max_useable_higher:
            higher_pulse_b_real = higher_pulse_b_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_b_imag = higher_pulse_b_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_b = max_b + 1
            add_pulse_to_all = True
          elif (index+1) % 3 == 0 and max_c < max_useable_higher:
            higher_pulse_c_real = higher_pulse_c_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_c_imag = higher_pulse_c_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag
            max_c = max_c + 1
            add_pulse_to_all = True

          if add_pulse_to_all:
            higher_pulse_all_real = higher_pulse_all_real + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].real
            higher_pulse_all_imag = higher_pulse_all_imag + audio_block2[offset + pulse_start_index + (index * pulse_length):offset + pulse_start_index + ((index+1) * pulse_length)].imag


      higher_pulse_a   = higher_pulse_a_real + 1j   * higher_pulse_a_imag
      higher_pulse_b   = higher_pulse_b_real + 1j   * higher_pulse_b_imag
      higher_pulse_c   = higher_pulse_c_real + 1j   * higher_pulse_c_imag
      higher_pulse_all = higher_pulse_all_real + 1j * higher_pulse_all_imag

      max_a = np.max(np.abs(higher_pulse_a))
      max_b = np.max(np.abs(higher_pulse_b))
      max_c = np.max(np.abs(higher_pulse_c))
      if max_a >= max_b and max_a >= max_c:
        self.debug.info_message("MAX A is largest pulse higher")
        peak_pulse_count_higher[ocn.PULSE_A] = peak_pulse_count_higher[ocn.PULSE_A] + 1
        max_pulse_higher = higher_pulse_a
        product_pulse_higher1 = higher_pulse_b
        product_pulse_higher2 = higher_pulse_c
        product_pulse_higher3 = higher_pulse_b + higher_pulse_c
      elif max_b >= max_a and max_b >= max_c:
        self.debug.info_message("MAX B is largest pulse higher")
        peak_pulse_count_higher[ocn.PULSE_B] = peak_pulse_count_higher[ocn.PULSE_B] + 1
        max_pulse_higher = higher_pulse_b
        product_pulse_higher1 = higher_pulse_c
        product_pulse_higher2 = higher_pulse_a
        product_pulse_higher3 = higher_pulse_c + higher_pulse_a
      elif max_c >= max_a and max_c >= max_b:
        self.debug.info_message("MAX C is largest pulse higher")
        peak_pulse_count_higher[ocn.PULSE_C] = peak_pulse_count_higher[ocn.PULSE_C] + 1
        max_pulse_higher = higher_pulse_c
        product_pulse_higher1 = higher_pulse_a
        product_pulse_higher2 = higher_pulse_b
        product_pulse_higher3 = higher_pulse_a + higher_pulse_b


      self.debug.info_message("MAX A pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_A]))
      self.debug.info_message("MAX B pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_B]))
      self.debug.info_message("MAX C pulse lower count: "  + str(peak_pulse_count_lower[ocn.PULSE_C]))
      self.debug.info_message("MAX A pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_A]))
      self.debug.info_message("MAX B pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_B]))
      self.debug.info_message("MAX C pulse higher count: " + str(peak_pulse_count_higher[ocn.PULSE_C]))

      peak_pulse_count_total[ocn.PULSE_A]  = peak_pulse_count_lower[ocn.PULSE_A] + peak_pulse_count_higher[ocn.PULSE_A]
      peak_pulse_count_total[ocn.PULSE_B]  = peak_pulse_count_lower[ocn.PULSE_B] + peak_pulse_count_higher[ocn.PULSE_B]
      peak_pulse_count_total[ocn.PULSE_C]  = peak_pulse_count_lower[ocn.PULSE_C] + peak_pulse_count_higher[ocn.PULSE_C]

      """
      if peak_pulse_count_lower[ocn.PULSE_A] >= peak_pulse_count_lower[ocn.PULSE_B] and peak_pulse_count_lower[ocn.PULSE_A] >= peak_pulse_count_lower[ocn.PULSE_C]:
        peak_lower = ocn.PULSE_A
      elif peak_pulse_count_lower[ocn.PULSE_B] >= peak_pulse_count_lower[ocn.PULSE_A] and peak_pulse_count_lower[ocn.PULSE_B] >= peak_pulse_count_lower[ocn.PULSE_C]:
        peak_lower = ocn.PULSE_B
      elif peak_pulse_count_lower[ocn.PULSE_C] >= peak_pulse_count_lower[ocn.PULSE_A] and peak_pulse_count_lower[ocn.PULSE_C] >= peak_pulse_count_lower[ocn.PULSE_B]:
        peak_lower = ocn.PULSE_C

      if peak_pulse_count_higher[ocn.PULSE_A] >= peak_pulse_count_higher[ocn.PULSE_B] and peak_pulse_count_higher[ocn.PULSE_A] >= peak_pulse_count_higher[ocn.PULSE_C]:
        peak_higher = ocn.PULSE_A
      elif peak_pulse_count_higher[ocn.PULSE_B] >= peak_pulse_count_higher[ocn.PULSE_A] and peak_pulse_count_higher[ocn.PULSE_B] >= peak_pulse_count_higher[ocn.PULSE_C]:
        peak_higher = ocn.PULSE_B
      elif peak_pulse_count_higher[ocn.PULSE_C] >= peak_pulse_count_higher[ocn.PULSE_A] and peak_pulse_count_higher[ocn.PULSE_C] >= peak_pulse_count_higher[ocn.PULSE_B]:
        peak_higher = ocn.PULSE_C
      """
      #"""
      if peak_pulse_count_total[ocn.PULSE_A] >= peak_pulse_count_total[ocn.PULSE_B] and peak_pulse_count_total[ocn.PULSE_A] >= peak_pulse_count_total[ocn.PULSE_C]:
        peak_total = ocn.PULSE_A
      elif peak_pulse_count_total[ocn.PULSE_B] >= peak_pulse_count_total[ocn.PULSE_A] and peak_pulse_count_total[ocn.PULSE_B] >= peak_pulse_count_total[ocn.PULSE_C]:
        peak_total = ocn.PULSE_B
      elif peak_pulse_count_total[ocn.PULSE_C] >= peak_pulse_count_total[ocn.PULSE_A] and peak_pulse_count_total[ocn.PULSE_C] >= peak_pulse_count_total[ocn.PULSE_B]:
        peak_total = ocn.PULSE_C
      #"""

      #return peak_lower, peak_higher
      return peak_total, peak_total
      #return ocn.PULSE_C, ocn.PULSE_C

    except:
      self.debug.error_message("Exception in peak_pulse_quantifier2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def receive_pre_filters_average_data_intra_triple(self, pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust):
    self.debug.info_message("receive_pre_filters_average_data_intra_triple")
    try:
      #peak_lower, peak_higher = self.peak_pulse_quantifier(pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher)
      peak_lower  = ocn.PULSE_A  
      peak_higher = ocn.PULSE_A  
  
      #if peak_lowhigh != []:
      #  peak_lower  = peak_lowhigh[0]
      #  peak_higher = peak_lowhigh[1]
  
      self.debug.info_message("peak_lower after downconvert: " + str(peak_lower))
      self.debug.info_message("peak_higher after downconvert: " + str(peak_higher))

      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

      audio_block1[0:pulse_start_index] = np.zeros(pulse_start_index)
      audio_block2[0:pulse_start_index] = np.zeros(pulse_start_index)

      selected_type = self.osmod.form_gui.window['combo_intra_combine_type'].get()
      if selected_type == 'Type 1':
        combine_type = ocn.INTRA_COMBINE_TYPE1
      elif selected_type == 'Type 2':
        combine_type = ocn.INTRA_COMBINE_TYPE2
      elif selected_type == 'Type 3':
        combine_type = ocn.INTRA_COMBINE_TYPE3
      elif selected_type == 'Type 4':
        combine_type = ocn.INTRA_COMBINE_TYPE4
      elif selected_type == 'Type 5':
        combine_type = ocn.INTRA_COMBINE_TYPE5
      elif selected_type == 'Type 6':
        combine_type = ocn.INTRA_COMBINE_TYPE6
      elif selected_type == 'Type 7':
        combine_type = ocn.INTRA_COMBINE_TYPE7
      elif selected_type == 'Type 8':
        combine_type = ocn.INTRA_COMBINE_TYPE8
      elif selected_type == 'Type 9':
        combine_type = ocn.INTRA_COMBINE_TYPE9
      elif selected_type == 'Type 10':
        combine_type = ocn.INTRA_COMBINE_TYPE10

      def deriveCombinationPulses(block, fine_tune_pulse_start_index):
        nonlocal pulse_all_real
        nonlocal pulse_all_imag
        nonlocal pulse_a_real
        nonlocal pulse_a_imag
        nonlocal pulse_b_real
        nonlocal pulse_b_imag
        nonlocal pulse_c_real
        nonlocal pulse_c_imag
        nonlocal max_a
        nonlocal max_b
        nonlocal max_c
        nonlocal has_pulse_all
        nonlocal has_pulse_a
        nonlocal has_pulse_b
        nonlocal has_pulse_c
 
        add_pulse_to_all = False
        if index % 3 == 0 and max_a < max_useable:
          if has_pulse_a == True:
            pulse_a_real = pulse_a_real + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_a_imag = pulse_a_imag + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag
          else:
            has_pulse_a = True
            pulse_a_real = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_a_imag = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag

          max_a = max_a + 1
          add_pulse_to_all = True
        elif (index+2) % 3 == 0 and max_b < max_useable:
          if has_pulse_b == True:
            pulse_b_real = pulse_b_real + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_b_imag = pulse_b_imag + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag
          else:
            has_pulse_b = True
            pulse_b_real = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_b_imag = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag

          max_b = max_b + 1
          add_pulse_to_all = True
        elif (index+1) % 3 == 0 and max_c < max_useable:
          if has_pulse_c == True:
            pulse_c_real = pulse_c_real + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_c_imag = pulse_c_imag + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag
          else:
            has_pulse_c = True
            pulse_c_real = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_c_imag = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag

          max_c = max_c + 1
          add_pulse_to_all = True

        if add_pulse_to_all:
          if has_pulse_all == True:
            pulse_all_real = pulse_all_real + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_all_imag = pulse_all_imag + block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag
          else:
            has_pulse_all = True
            pulse_all_real = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].real
            pulse_all_imag = block[offset + fine_tune_pulse_start_index + (index * pulse_length):offset + fine_tune_pulse_start_index + ((index+1) * pulse_length)].imag

      """ first frequency"""
      lower_pulse_all  = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_a    = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_b    = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      lower_pulse_c    = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      """ second frequency"""
      higher_pulse_all = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_a   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_b   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])
      higher_pulse_c   = np.zeros_like(audio_block2[pulse_start_index:pulse_end_index])

      pulse_all_real   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_a_real     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_b_real     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_c_real     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_all_imag   = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_a_imag     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_b_imag     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])
      pulse_c_imag     = np.zeros_like(audio_block1[pulse_start_index:pulse_end_index])

      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_lower: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_lower = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable_lower low: " + str(max_useable_lower))

      max_a = 0
      max_b = 0
      max_c = 0
      for index in interpolated_higher: 
        if index % 3 == 0:
          max_a = max_a + 1
        elif (index+2) % 3 == 0:
          max_b = max_b + 1
        elif (index+1) % 3 == 0:
          max_c = max_c + 1
      max_useable_higher = min(min(max_a, max_b), max_c)
      self.debug.info_message("max_useable_higher high: " + str(max_useable_higher))

      """ average each frequency block"""
      fine_tune_pulse_start_index_lower  = (pulse_start_index + fine_tune_adjust[0]) % pulse_length
      fine_tune_pulse_start_index_higher = (pulse_start_index + fine_tune_adjust[1]) % pulse_length
      num_full_blocks = int((len(audio_block1) - pulse_start_index) // self.osmod.symbol_block_size)
      for block_count in range(0, num_full_blocks): 
        offset = (block_count * self.osmod.pulses_per_block) * pulse_length

        max_a = 0
        max_b = 0
        max_c = 0
        pulse_count = 0
        has_pulse_a   = False
        has_pulse_b   = False
        has_pulse_c   = False
        has_pulse_all = False
        for index in interpolated_lower: 
          pulse_count = pulse_count + 1
          max_useable = max_useable_lower
          deriveCombinationPulses(audio_block1, fine_tune_pulse_start_index_lower)

        lower_pulse_a   = pulse_a_real + 1j   * pulse_a_imag
        lower_pulse_b   = pulse_b_real + 1j   * pulse_b_imag
        lower_pulse_c   = pulse_c_real + 1j   * pulse_c_imag
        lower_pulse_all = pulse_all_real + 1j * pulse_all_imag

        max_a = np.max(np.abs(lower_pulse_a))
        max_b = np.max(np.abs(lower_pulse_b))
        max_c = np.max(np.abs(lower_pulse_c))

        if peak_lower == ocn.PULSE_A:
          #self.debug.info_message("MAX A is largest pulse lower")
          max_pulse_lower = lower_pulse_a
          product_pulse_lower1 = lower_pulse_b
          product_pulse_lower2 = lower_pulse_c
          product_pulse_lower3 = lower_pulse_b + lower_pulse_c
        elif peak_lower == ocn.PULSE_B:
          #self.debug.info_message("MAX B is largest pulse lower")
          max_pulse_lower = lower_pulse_b
          product_pulse_lower1 = lower_pulse_c
          product_pulse_lower2 = lower_pulse_a
          product_pulse_lower3 = lower_pulse_c + lower_pulse_a
        elif peak_lower == ocn.PULSE_C:
          #self.debug.info_message("MAX C is largest pulse lower")
          max_pulse_lower = lower_pulse_c
          product_pulse_lower1 = lower_pulse_a
          product_pulse_lower2 = lower_pulse_b
          product_pulse_lower3 = lower_pulse_a + lower_pulse_b


        max_a = 0
        max_b = 0
        max_c = 0
        pulse_count = 0
        has_pulse_a   = False
        has_pulse_b   = False
        has_pulse_c   = False
        has_pulse_all = False
        for index in interpolated_higher: 
          pulse_count = pulse_count + 1
          max_useable = max_useable_higher
          deriveCombinationPulses(audio_block2, fine_tune_pulse_start_index_higher)

        higher_pulse_a   = pulse_a_real + 1j   * pulse_a_imag
        higher_pulse_b   = pulse_b_real + 1j   * pulse_b_imag
        higher_pulse_c   = pulse_c_real + 1j   * pulse_c_imag
        higher_pulse_all = pulse_all_real + 1j * pulse_all_imag

        max_a = np.max(np.abs(higher_pulse_a))
        max_b = np.max(np.abs(higher_pulse_b))
        max_c = np.max(np.abs(higher_pulse_c))

        if peak_higher == ocn.PULSE_A:
          #self.debug.info_message("MAX A is largest pulse higher")
          max_pulse_higher = higher_pulse_a
          product_pulse_higher1 = higher_pulse_b
          product_pulse_higher2 = higher_pulse_c
          product_pulse_higher3 = higher_pulse_b + higher_pulse_c
        elif peak_higher == ocn.PULSE_B:
          #self.debug.info_message("MAX B is largest pulse higher")
          max_pulse_higher = higher_pulse_b
          product_pulse_higher1 = higher_pulse_c
          product_pulse_higher2 = higher_pulse_a
          product_pulse_higher3 = higher_pulse_c + higher_pulse_a
        elif peak_higher == ocn.PULSE_C:
          #self.debug.info_message("MAX C is largest pulse higher")
          max_pulse_higher = higher_pulse_c
          product_pulse_higher1 = higher_pulse_a
          product_pulse_higher2 = higher_pulse_b
          product_pulse_higher3 = higher_pulse_a + higher_pulse_b





        #for i in range(0, self.osmod.pulses_per_block): 
        #  if (i not in interpolated_lower) and (i not in interpolated_higher):
        #    audio_block1[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = np.zeros_like(lower_pulse_a)

        """write the data back to the data stream """
        for i in range(0, self.osmod.pulses_per_block): 
          if i in interpolated_lower:
            #if i % 3 == 0:
            if i // max_useable_lower == 0:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_a + lower_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + (lower_pulse_a + lower_pulse_b)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all - (lower_pulse_a + lower_pulse_b)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = (lower_pulse_all + (lower_pulse_a + lower_pulse_b)) - lower_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = max_pulse_lower
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_a - lower_pulse_b - lower_pulse_c 
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + lower_pulse_a - lower_pulse_b - lower_pulse_c
            #elif (i+2) % 3 == 0:
            elif i // max_useable_lower == 1:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_b + lower_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + (lower_pulse_b + lower_pulse_c)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all - (lower_pulse_b + lower_pulse_c)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = (lower_pulse_all + (lower_pulse_b + lower_pulse_c)) - lower_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = max_pulse_lower
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_b - lower_pulse_a - lower_pulse_c 
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + lower_pulse_b - lower_pulse_a - lower_pulse_c
            #elif (i+1) % 3 == 0:
            elif i // max_useable_lower == 2:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_c + lower_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + (lower_pulse_c + lower_pulse_a)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all - (lower_pulse_c + lower_pulse_a)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = (lower_pulse_all + (lower_pulse_c + lower_pulse_a)) - lower_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = max_pulse_lower
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = product_pulse_lower3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_c - lower_pulse_a - lower_pulse_b 
              #audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = lower_pulse_all + lower_pulse_c - lower_pulse_a - lower_pulse_b
          #else:
          #  audio_block1[offset + fine_tune_pulse_start_index_lower+(i * pulse_length):offset + fine_tune_pulse_start_index_lower + ((i+1) * pulse_length)] = np.zeros_like(lower_pulse_a)

          if i in interpolated_higher:
            #if i % 3 == 0:
            if i // max_useable_higher == 0:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_a + higher_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + (higher_pulse_a + higher_pulse_b)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all - (higher_pulse_a + higher_pulse_b)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = (higher_pulse_all + (higher_pulse_a + higher_pulse_b)) - higher_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = max_pulse_higher
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_a - higher_pulse_b - higher_pulse_c 
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + higher_pulse_a - higher_pulse_b - higher_pulse_c 
            #elif (i+2) % 3 == 0:
            elif i // max_useable_higher == 1:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_b + higher_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + (higher_pulse_b + higher_pulse_c)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all - (higher_pulse_b + higher_pulse_c)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = (higher_pulse_all + (higher_pulse_b + higher_pulse_c)) - higher_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = max_pulse_higher
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_b - higher_pulse_a - higher_pulse_c 
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + higher_pulse_b - higher_pulse_a - higher_pulse_c 
            #elif (i+1) % 3 == 0:
            elif i // max_useable_higher == 2:
              if combine_type == ocn.INTRA_COMBINE_TYPE1:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_c
              elif combine_type == ocn.INTRA_COMBINE_TYPE2:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_c + higher_pulse_a
              elif combine_type == ocn.INTRA_COMBINE_TYPE3:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + (higher_pulse_c + higher_pulse_a)
              elif combine_type == ocn.INTRA_COMBINE_TYPE4:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all - (higher_pulse_c + higher_pulse_a)
              elif combine_type == ocn.INTRA_COMBINE_TYPE5:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = (higher_pulse_all + (higher_pulse_c + higher_pulse_a)) - higher_pulse_b
              elif combine_type == ocn.INTRA_COMBINE_TYPE6:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = max_pulse_higher
              elif combine_type == ocn.INTRA_COMBINE_TYPE7:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher1
              elif combine_type == ocn.INTRA_COMBINE_TYPE8:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher2
              elif combine_type == ocn.INTRA_COMBINE_TYPE9:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = product_pulse_higher3
              elif combine_type == ocn.INTRA_COMBINE_TYPE10:
                audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_c - higher_pulse_a - higher_pulse_b 
              #audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = higher_pulse_all + higher_pulse_c - higher_pulse_a - higher_pulse_b  
          #else:
          #  audio_block2[offset + fine_tune_pulse_start_index_higher+(i * pulse_length):offset + fine_tune_pulse_start_index_higher + ((i+1) * pulse_length)] = np.zeros_like(higher_pulse_a)

      self.debug.info_message("len(audio_block): " + str(len(audio_block1)))
      last_valid = 0
      #for remainder_data in range((num_full_blocks * self.osmod.pulses_per_block * pulse_length) + fine_tune_pulse_start_index_higher, len(audio_block1)-pulse_length, pulse_length): 
      #  self.debug.info_message("remainder_data: " + str(remainder_data))
      #  audio_block1[remainder_data:remainder_data + pulse_length] = lower_pulse_a
      #  audio_block2[remainder_data:remainder_data + pulse_length] = higher_pulse_a
      #  last_valid = remainder_data + pulse_length

    except:
      self.debug.error_message("Exception in receive_pre_filters_average_data_intra_triple: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return [audio_block1, audio_block2], pulse_start_index 


  def centerPulsePeak(self, low_high, signal, pulse_index, pulse_type):
    #self.debug.info_message("centerPulsePeak")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      """ test offset is used to obtain 2 signals of length pulse_length: 1 at pulse -1/4 and 2 at pulse +1/4"""
      #test_offset = int(pulse_length/4)
      #test_offset = int(pulse_length/6)
      test_offset = int(pulse_length/3)

      pulse_test_1 = signal[pulse_index - test_offset:pulse_index + pulse_length - test_offset]
      pulse_test_2 = signal[pulse_index + test_offset:pulse_index + pulse_length + test_offset]

      #sigma_value = 5
      sigma_value = 7
      analytic_signal_1 = gaussian_filter(pulse_test_1, sigma=sigma_value)
      analytic_signal_2 = gaussian_filter(pulse_test_2, sigma=sigma_value)
      #analytic_signal_1 = hilbert(pulse_test_1)
      #analytic_signal_2 = hilbert(pulse_test_2)

      envelope_1 = np.abs(analytic_signal_1)
      envelope_2 = np.abs(analytic_signal_2)

      #self.debug.info_message("envelope_1: " + str(envelope_1))
      #self.debug.info_message("envelope_2: " + str(envelope_2))

      #smoothed_envelope = gaussian_filter(envelope, sigma=10)
      #peak_max = np.max(smoothed_envelope)
      #peak_index = np.where(smoothed_envelope == peak_max)[0]

      peak_max_1 = np.max(envelope_1)
      peak_max_2 = np.max(envelope_2)
      peak_min_1 = np.min(envelope_1)
      peak_min_2 = np.min(envelope_2)
      #self.debug.info_message("peak_max_1: " + str(peak_max_1))
      #self.debug.info_message("peak_max_2: " + str(peak_max_2))
      #self.debug.info_message("peak_min_1: " + str(peak_min_1))
      #self.debug.info_message("peak_min_2: " + str(peak_min_2))
      peak_index_max_1 = np.where(envelope_1 == peak_max_1)[0]
      peak_index_max_2 = np.where(envelope_2 == peak_max_2)[0]
      peak_index_min_1 = np.where(envelope_1 == peak_min_1)[0]
      peak_index_min_2 = np.where(envelope_2 == peak_min_2)[0]

      #self.debug.info_message("peak_index_max_1: " + str(peak_index_max_1))
      #self.debug.info_message("peak_index_max_2: " + str(peak_index_max_2))
      #self.debug.info_message("peak_index_min_1: " + str(peak_index_min_1))
      #self.debug.info_message("peak_index_min_2: " + str(peak_index_min_2))


      maximum_index = pulse_length - 1
      minimum_index = 0
      if peak_index_max_1 != maximum_index and peak_index_max_1 != minimum_index:
        corrected_peak_index_max_1 = (peak_index_max_1 + test_offset) % pulse_length
        derived_index_min_1 = (peak_index_max_1 + int(pulse_length / 2) + test_offset) % pulse_length
        #self.debug.info_message("corrected_peak_index_max_1: " + str(corrected_peak_index_max_1))
        #self.debug.info_message("derived_index_min_1: " + str(derived_index_min_1))
        self.pulse_history[low_high].append(derived_index_min_1[0])
        self.pulse_abc[pulse_type].append(derived_index_min_1[0])
      elif peak_index_min_1 != minimum_index and peak_index_min_1 != maximum_index:
        derived_index_max_1 = (peak_index_min_1 + int(pulse_length / 2) + test_offset) % pulse_length
        corrected_peak_index_min_1 = (peak_index_min_1 + test_offset) % pulse_length
        #self.debug.info_message("derived_index_max_1: " + str(derived_index_max_1))
        #self.debug.info_message("corrected_peak_index_min_1: " + str(corrected_peak_index_min_1))
        self.pulse_history[low_high].append(corrected_peak_index_min_1[0])
        self.pulse_abc[pulse_type].append(corrected_peak_index_min_1[0])
      elif peak_index_max_2 != maximum_index and peak_index_max_2 != minimum_index:
        corrected_peak_index_max_2 = (peak_index_max_2 - test_offset + pulse_length) % pulse_length
        derived_index_min_2 = (peak_index_max_2 + int(pulse_length / 2) - test_offset + pulse_length) % pulse_length
        #self.debug.info_message("corrected_peak_index_max_2: " + str(corrected_peak_index_max_2))
        #self.debug.info_message("derived_index_min_2: " + str(derived_index_min_2))
        self.pulse_history[low_high].append(derived_index_min_2[0])
        self.pulse_abc[pulse_type].append(derived_index_min_2[0])
      elif peak_index_min_2 != minimum_index and peak_index_min_2 != maximum_index:
        derived_index_max_2 = (peak_index_min_2 + int(pulse_length / 2) - test_offset + pulse_length) % pulse_length
        corrected_peak_index_min_2 = (peak_index_min_2 - test_offset + pulse_length) % pulse_length
        #self.debug.info_message("derived_index_max_2: " + str(derived_index_max_2))
        #self.debug.info_message("corrected_peak_index_min_2: " + str(corrected_peak_index_min_2))
        self.pulse_history[low_high].append(corrected_peak_index_min_2[0])
        self.pulse_abc[pulse_type].append(corrected_peak_index_min_2[0])
      else:
        self.debug.info_message("no derivation obtained")
        self.debug.info_message("peak_index_max_1: " + str(peak_index_max_1))
        self.debug.info_message("peak_index_max_2: " + str(peak_index_max_2))
        self.debug.info_message("peak_index_min_1: " + str(peak_index_min_1))
        self.debug.info_message("peak_index_min_2: " + str(peak_index_min_2))

      #self.debug.info_message("self.pulse_history[low_high]: " + str(self.pulse_history[low_high]))

    except:
      self.debug.error_message("Exception in centerPulsePeak: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

