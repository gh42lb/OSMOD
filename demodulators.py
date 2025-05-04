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

from osmod_c_interface import ptoc_float_array, ptoc_double_array, ptoc_float, ctop_int

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

    if self.plotfirstonly==False:
      self.plotfirstonly = True
      self.osmod.form_gui.plotQueue.put((signal_a, 'canvas_waveform'))
      self.osmod.form_gui.plotQueue.put((signal_b, 'canvas_second_waveform'))


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

    decoded_bitstring_1 = []
    decoded_bitstring_2 = []

    try:
      for i in range(0, len(decoded_values1_real)):
        lookup_string1 = str(decoded_values1_real[i]) + ':' + str(decoded_values1_imag[i])
        lookup_string2 = str(decoded_values2_real[i]) + ':' + str(decoded_values2_imag[i])
        if lookup_string1 == '0:0' or lookup_string2 == '0:0':
          self.osmod.has_invalid_decodes = True
          self.debug.error_message("invalid decode: " + lookup_string1)
          self.osmod.form_gui.window['ml_txrx_recvtext'].print('*', end="", text_color='red', background_color = 'white')
        else:
          self.debug.info_message("looking up: " + lookup_string1)
          self.debug.info_message("bit value: " + str(symbol_to_bits[lookup_string1]))

          if self.osmod.process_debug == True:
            self.osmod.form_gui.window['ml_txrx_recvtext'].print('[' + str(symbol_to_bits[lookup_string1]), end="", text_color='green', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(symbol_to_bits[lookup_string2]) + ']', end="", text_color='green', background_color = 'white')

          decoded_bitstring_1.append(str(symbol_to_bits[lookup_string1]))
          decoded_bitstring_2.append(str(symbol_to_bits[lookup_string2]))

      for i in range(0, len(decoded_bitstring_1)):
        lookup_string = str(decoded_bitstring_1[i]) + str(decoded_bitstring_2[i])
        binary = int(lookup_string, 2)
        char = self.b64_charfromindex_list[binary]
        self.debug.info_message("found char: " + str(char))
        self.osmod.form_gui.window['ml_txrx_recvtext'].print(str(char), end="", text_color='black', background_color = 'white')

      return decoded_bitstring_1, decoded_bitstring_2
    except:
      sys.stdout.write("Exception in displayTextResults: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def displayChartResults(self, signal_a, signal_b):
    self.osmod.form_gui.plotQueue.put((signal_a, 'canvas_waveform'))
    self.osmod.form_gui.plotQueue.put((signal_b, 'canvas_second_waveform'))


  def detectSampleOffset(self, signal):
    self.debug.info_message("detectSampleOffset" )
    try:
      max_list = []
      min_list = []
      all_list = []
      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        test_min = np.min(test_peak)
        max_indices = np.where(test_peak == test_max)
        min_indices = np.where(test_peak == test_min)
        self.debug.info_message("max indices: " + str(max_indices[0]))
        self.debug.info_message("min indices: " + str(min_indices[0]))

        for item in list(max_indices[0]):
          max_list.append(item)
          all_list.append(item)
        for item in list(min_indices[0]):
          min_list.append(item)
          all_list.append(item)

      self.debug.info_message("max_list: " + str(list(max_list)))
      self.debug.info_message("min_list: " + str(list(min_list)))
      self.debug.info_message("all_list: " + str(list(all_list)))
      if self.osmod.detector_function == 'median':
        self.debug.info_message("calculating median" )
        median_index_max = int(np.median(np.array(max_list)))
        median_index_min = int(np.median(np.array(min_list)))
        median_index_all = int(np.median(np.array(all_list)))
      elif self.osmod.detector_function == 'mode':
        self.debug.info_message("calculating mode" )
        median_index_max = int(stats.mode(max_list).mode[0])
        median_index_min = int(stats.mode(min_list).mode[0])
        median_index_all = int(stats.mode(all_list).mode[0])
      self.debug.info_message("mean max index: " + str(median_index_max))
      self.debug.info_message("mean min index: " + str(median_index_min))
      self.debug.info_message("mean all index: " + str(median_index_all))

      pulse_size = (self.osmod.symbol_block_size*2)/self.osmod.pulses_per_block
      half_pulse_size = pulse_size / 2
      sample_start = int((median_index_max + half_pulse_size) % pulse_size)

      self.debug.info_message("sample_start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in detectSampleOffset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return sample_start


  def extractPhaseValuesWithOffsetDouble(self, signal1, signal2, median_block_offset, frequency, num_waves, where):
    self.debug.info_message("extractPhaseValuesWithOffset" )
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

        self.debug.info_message("dtype: "   + str(signal.dtype))
        self.debug.info_message("value 1: " + str(signal[0]))
        self.debug.info_message("value 2: " + str(signal[1]))
        self.debug.info_message("value 3: " + str(signal[2]))
        self.debug.info_message("value 4: " + str(signal[3]))
        self.debug.info_message("value 5: " + str(signal[4]))

        self.osmod.compiled_lib.costas_loop_8psk.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_float, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.osmod.compiled_lib.costas_loop_8psk(c_signal, c_recovered_signal1a, c_recovered_signal1b, c_frequency, c_t, signal.size)

        return [recovered_signal1a, recovered_signal1b], phase_error_history

      else:
        return self.recoverBasebandSignalPythonCode(frequency, signal)

    except:
      self.debug.error_message("Exception in recoverBasebandSignalOptimized: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


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

  """ contiguous by distance for and wrap around"""
  def interpolate_contiguous_items(self, list_items):
    self.debug.info_message("interpolate_contiguous_items")
    try:

      saved_list_items = list_items
      have_suspect_items = True
      while have_suspect_items:
        have_suspect_items = False
        median_input_list = int(np.median(np.array(list_items)))
        self.debug.info_message("median_input_list: " + str(median_input_list))
        for i in list_items:
          self.debug.info_message("distance: " + str(abs(median_input_list - i)))
          if abs(median_input_list - i) > int((self.osmod.pulses_per_block / self.osmod.num_carriers) / 2):
            self.debug.info_message("suspect item: " + str(i))
            list_items.remove(i)
            have_suspect_items = True

      median_input_list_truth = median_input_list

      """ now repeat process using final truth value as median """
      list_items = saved_list_items
      for i in list_items:
        self.debug.info_message("distance from truth: " + str(abs(median_input_list_truth - i)))
        if abs(median_input_list_truth - i) > int((self.osmod.pulses_per_block / self.osmod.num_carriers) / 2):
          self.debug.info_message("suspect item from truth: " + str(i))
          list_items.remove(i)


      half = int(self.osmod.pulses_per_block / 2)

      start_value = list_items[0]
      min_value = 0
      max_value = 0
      walk = 0
      for i, j in zip(list_items, list_items[1:]):
        if abs(i - j) < half:
          walk = walk - (i - j)
        elif abs(j - i) < half:
          walk = walk + (j - i)
        elif abs(j + self.osmod.pulses_per_block - i) < half:
          walk = walk + (j + self.osmod.pulses_per_block - i)
        elif abs(i + self.osmod.pulses_per_block - j) < half:
          walk = walk - (i + self.osmod.pulses_per_block - j)

        if walk > max_value:
          max_value = walk
        if walk < min_value:
          min_value = walk

        self.debug.info_message("walk: " + str(walk))

      self.debug.info_message("min: " + str(min_value))
      self.debug.info_message("max: " + str(max_value))

      interpolated_list = []
      for x in range(start_value + min_value, start_value + max_value + 1):
        interpolated_list.append(x % self.osmod.pulses_per_block)

      self.debug.info_message("Original list: " + str(list_items))
      self.debug.info_message("Interpolated list: " + str(interpolated_list))

      return interpolated_list

    except:
      sys.stdout.write("Exception in interpolate_contiguous_items: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

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


  
  def testCodeGetFrequencySectionStart(self, signal):
    self.debug.info_message("testCodeGetFrequencySectionStart" )
    try:
      all_list = []
      all_list_ab = []
      all_dict_where = {}

      pulse_width = self.osmod.symbol_block_size / self.osmod.pulses_per_block

      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        #self.debug.info_message("getting test peak")
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        test_min = np.min(test_peak)
        max_indices = np.where((test_peak*(100/test_max)) > self.osmod.parameters[5])
        min_indices = np.where((test_peak*(100/test_min)) > self.osmod.parameters[5])
        for x in range(0, len(max_indices[0]) ):
          all_list.append(max_indices[0][x] % self.osmod.symbol_block_size)
          sequence_value = int((max_indices[0][x] % self.osmod.symbol_block_size) // (self.osmod.symbol_block_size/self.osmod.pulses_per_block))
          all_list_ab.append(sequence_value)
          """ where == which pulse numerically from the start"""
          all_dict_where[sequence_value] = max_indices[0][x] % (self.osmod.symbol_block_size // self.osmod.pulses_per_block)
        for x in range(0, len(min_indices[0]) ):
          all_list.append(min_indices[0][x] % self.osmod.symbol_block_size)
          sequence_value = int((min_indices[0][x] % self.osmod.symbol_block_size) // (self.osmod.symbol_block_size/self.osmod.pulses_per_block))
          all_list_ab.append(sequence_value)
          """ where == which pulse numerically from the start"""
          all_dict_where[sequence_value] = min_indices[0][x] % (self.osmod.symbol_block_size // self.osmod.pulses_per_block)
      #self.debug.info_message("Frequency Section all indices: " + str(all_list))
      #self.debug.info_message("Frequency Section all ab indices: " + str(all_list_ab))

      half = int(self.osmod.pulses_per_block / 2)

      max_occurrences = []
      for count in range(0,half):
        max_occurrences_item = self.count_max_occurrences(all_list_ab, max_occurrences)
        if max_occurrences_item != []:
          max_occurrences.append(max_occurrences_item[0])

      median_index_all = int(np.median(np.array(all_list)))
      self.debug.info_message("Frequency Section Filter mean all index: " + str(median_index_all))

      sample_start = median_index_all
      self.debug.info_message("Frequency Setion start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in testCodeGetFrequencySectionStart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return max_occurrences, all_dict_where


  def receive_pre_filters_filter_wave(self, pulse_start_index, audio_block, frequency):

    """ calculate receive side RRC filter"""
    pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

    """ apply receive side RRC filter"""
    for block_count in range(0, int(len(audio_block) // self.osmod.symbol_block_size)): 
      offset = (block_count * self.osmod.pulses_per_block) * pulse_length
      #self.debug.info_message("block_count: " + str(block_count))
      for pulse_count in range(0, self.osmod.pulses_per_block): 
        if (offset + pulse_start_index + ( (pulse_count+1) * pulse_length)) < int(len(audio_block)):
          audio_block[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] = audio_block[offset + pulse_start_index+(pulse_count * pulse_length):offset + pulse_start_index + ((pulse_count+1) * pulse_length)] * self.osmod.filtRRC_coef_main

    self.osmod.getDurationAndReset('apply RRC to wave')

    """ fft bandpass filter"""
    fft_filtered_lower  = self.bandpass_filter_fft(audio_block, frequency[0] + self.osmod.fft_filter[0], frequency[0] + self.osmod.fft_filter[1])
    fft_filtered_higher = self.bandpass_filter_fft(audio_block, frequency[1] + self.osmod.fft_filter[2], frequency[1] + self.osmod.fft_filter[3] )

    self.osmod.getDurationAndReset('bandpass_filter_fft in receive_pre_filters_filter_wave')

    return (fft_filtered_lower.real + fft_filtered_lower.imag ) / 2, (fft_filtered_higher.real + fft_filtered_higher.imag) / 2

  def removeConflictingItemsTwoList(self, max_occurrences_lists):
    self.debug.info_message("removeConflictingItemsTwoList")
    try:

      max_occurrences_lower = max_occurrences_lists[0]
      max_occurrences_higher = max_occurrences_lists[1]

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

    except:
      self.debug.error_message("Exception in removeConflictingItemsTwoList: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return [max_occurrences_lower, max_occurrences_higher]

  def receive_pre_filters_interpolate(self, pulse_start_index, audio_block, frequency):
    self.debug.info_message("receive_pre_filters_interpolate")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

      fft_filtered_lower  = self.bandpass_filter_fft(audio_block, frequency[0] + self.osmod.fft_interpolate[0], frequency[0] + self.osmod.fft_interpolate[1])
      fft_filtered_higher = self.bandpass_filter_fft(audio_block, frequency[1] + self.osmod.fft_interpolate[2], frequency[1] + self.osmod.fft_interpolate[3])

      self.osmod.getDurationAndReset('bandpass_filter_fft')

      max_occurrences_lower, where_lower    = self.testCodeGetFrequencySectionStart(fft_filtered_lower)
      max_occurrences_higher, where_higher  = self.testCodeGetFrequencySectionStart(fft_filtered_higher)
      self.debug.info_message("max_occurrences_lower: " + str(max_occurrences_lower))
      self.debug.info_message("max_occurrences_higher: " + str(max_occurrences_higher))

      self.osmod.getDurationAndReset('testCodeGetFrequencySectionStart')

      max_occurrences_lists = self.removeConflictingItemsTwoList([max_occurrences_lower, max_occurrences_higher])
      max_occurrences_lower  = max_occurrences_lists[0]
      max_occurrences_higher = max_occurrences_lists[1]

      self.osmod.getDurationAndReset('removeConflictingItemsTwoList')

      self.debug.info_message("max_occurrences_lower 2: " + str(max_occurrences_lower))
      self.debug.info_message("max_occurrences_higher 2: " + str(max_occurrences_higher))

      interpolated_lower  = self.interpolate_contiguous_items(max_occurrences_lower)
      interpolated_higher = self.interpolate_contiguous_items(max_occurrences_higher)

      self.osmod.getDurationAndReset('interpolate_contiguous_items')

      self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      half = int(self.osmod.pulses_per_block/2)

      """ if either of the inetrpolated lists is complete, fill out the other interpolated list if incomplete"""
      if len(interpolated_lower) == half and len(interpolated_higher) < half:
        for i in range(0,self.osmod.pulses_per_block):
          if i not in interpolated_lower and i not in interpolated_higher:
            interpolated_higher.append(i)
      elif len(interpolated_higher) == half and len(interpolated_lower) < half:
        for i in range(0,self.osmod.pulses_per_block):
          if i not in interpolated_lower and i not in interpolated_higher:
            interpolated_lower.append(i)

      self.debug.info_message("interpolated_lower 3: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher 3: " + str(interpolated_higher))

      """ if either of the inetrpolated lists is greater than the other list, fill out the other interpolated list"""
      interpolated_lower  = self.sort_interpolated(interpolated_lower)
      interpolated_higher = self.sort_interpolated(interpolated_higher)
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

      self.debug.info_message("interpolated_lower 4: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher 4: " + str(interpolated_higher))

      if len(interpolated_lower) == half:
        first_value = self.get_first_interpolated(interpolated_lower)
        self.block_start_candidates.append(first_value)

      self.osmod.getDurationAndReset('interpolate_contiguous_items 2')


      self.debug.info_message("modified interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("modified interpolated_higher: " + str(interpolated_higher))

    except:
      self.debug.error_message("Exception in receive_pre_filters_interpolate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return self.sort_interpolated(interpolated_lower), self.sort_interpolated(interpolated_higher)

  """ min can be a higher numeric value but is defined as the first in the series"""
  def get_first_interpolated(self, interpolated_lower):
    self.debug.info_message("get_first_interpolated")
    try:
      """ does the data wrap around? """
      """ no wrap"""
      if self.osmod.pulses_per_block - 1 not in interpolated_lower:
        first_interpolated = min(interpolated_lower)
        self.debug.info_message("first_interpolated: " + str(first_interpolated))
        return (first_interpolated)

      """ the data does wrap around"""
      half = int(self.osmod.pulses_per_block/2)
      normalized_list = []
      for item in interpolated_lower:
        if item < half:
          normalized_list.append(item + self.osmod.pulses_per_block)
        else:
          normalized_list.append(item)

      first_interpolated = min(normalized_list) % self.osmod.pulses_per_block
      self.debug.info_message("first_interpolated: " + str(first_interpolated))
      return (first_interpolated)
    except:
      self.debug.error_message("Exception in get_first_interpolated: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def sort_interpolated(self, interpolated_lower):
    self.debug.info_message("sort_interpolated")
    try:
      half = int(self.osmod.pulses_per_block/2)
      normalized_list = []
      restored_sorted_list = []
      for item in interpolated_lower:
        if item < half:
          normalized_list.append(item + self.osmod.pulses_per_block)
        else:
          normalized_list.append(item)

      self.debug.info_message("normalized_list: " + str(normalized_list))
      normalized_list.sort()
      self.debug.info_message("normalized_list sorted: " + str(normalized_list))

      """ does the data wrap around? """
      if self.osmod.pulses_per_block - 1 in normalized_list:
        for item in normalized_list:
          if item < self.osmod.pulses_per_block:
            restored_sorted_list.append(item)
        for item in normalized_list:
          if item >= self.osmod.pulses_per_block:
            restored_sorted_list.append(item % self.osmod.pulses_per_block)
        self.debug.info_message("restored_sorted_list: " + str(restored_sorted_list))

        self.osmod.getDurationAndReset('sort_interpolated')

        return restored_sorted_list
      else:
        for item in normalized_list:
          if item >= self.osmod.pulses_per_block:
            restored_sorted_list.append(item % self.osmod.pulses_per_block)
        for item in normalized_list:
          if item < self.osmod.pulses_per_block:
            restored_sorted_list.append(item)
        self.debug.info_message("restored_sorted_list: " + str(restored_sorted_list))

        self.osmod.getDurationAndReset('sort_interpolated')

        return restored_sorted_list

    except:
      self.debug.error_message("Exception in sort_interpolated: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def receive_pre_filters_average_data(self, pulse_start_index, audio_block1, audio_block2, frequency, interpolated_lower, interpolated_higher):
    self.debug.info_message("receive_pre_filters_average_data")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

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
          if i in interpolated_higher:
            audio_block2[offset + pulse_start_index+(i * pulse_length):offset + pulse_start_index + ((i+1) * pulse_length)] = higher_pulse

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




