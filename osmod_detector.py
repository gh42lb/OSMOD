#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import sounddevice as sd
import numpy as np
import threading
import wave
import sys
import math

from collections import deque
from numpy import pi
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from scipy.io.wavfile import write, read
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import savgol_filter, hilbert

from modulators import ModulatorPSK 
from demodulators import DemodulatorPSK 
from queue import Queue
from scipy.interpolate import splrep, splev, PchipInterpolator

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

class OsmodDetector(object):

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_INFO)
    self.debug.info_message("__init__")
    self.osmod = osmod
    self.dict_saved_data = {}
    self.rotation_angles = [0] * 2
    self.series_angles   = [0] * 2
    self.pulse_train_length = 0
    self.disposition = 0

  def getStrongestFrequencyOverRange(self, audio_block):
    #self.debug.info_message("getStrongestFrequencyOverRange")
    try:
      freq_range = (1290,1310)
      fft_filtered, masked_fft  = self.osmod.demodulation_object.bandpass_filter_fft(audio_block, freq_range[0], freq_range[1])
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      #pulse_start_index = int(self.pulseDetector(fft_filtered) + pulse_length/2)
      #self.debug.info_message("pulse_start_index 2 : " + str(pulse_start_index))

      strong_freqs = 1400

    except:
      self.debug.error_message("Exception in getStrongestFrequency: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    #finally:
    #  self.debug.info_message("Completed getStrongestFrequency: ")

    return strong_freqs

  def getSavedDataNames(self):
    saved_data_names = []

    for key in self.dict_saved_data:
      saved_data_names.append(key)

    return saved_data_names

  """
  def pulseDetector(self, signal):
    #self.debug.info_message("pulseDetector" )
    try:
      all_list = []

      pulse_width = self.osmod.symbol_block_size / self.osmod.pulses_per_block

      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        test_min = np.min(test_peak)
        max_indices = np.where((test_peak*(100/test_max)) > self.osmod.parameters[5])
        min_indices = np.where((test_peak*(100/test_min)) > self.osmod.parameters[5])
        #self.debug.info_message("RRC max indices: " + str(max_indices[0]))
        #self.debug.info_message("RRC min indices: " + str(min_indices[0]))
        for x in range(0, len(max_indices[0]) ):
          all_list.append(int(max_indices[0][x] % pulse_width))
        for x in range(0, len(min_indices[0]) ):
          all_list.append(int(min_indices[0][x] % pulse_width))
      self.debug.info_message("RRC all indices: " + str(all_list))

      median_index_all = int(np.median(np.array(all_list)))
      self.debug.info_message("RRC Filter mean all index: " + str(median_index_all))

      #counts = np.bincount(all_list)

      #self.debug.info_message("counts : " + str(counts))

      return median_index_all  
      #sample_start = median_index_all
      #self.debug.info_message("sample_start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in findPulseStartIndex: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  """

  def pulseDetector(self, signal, frequency):
    self.debug.info_message("pulseDetector" )
    try:
      """ create an RRC pulse sample """
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      self.debug.info_message("pulse_length: " + str(pulse_length))

      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate
      term5 = 2 * np.pi * time
      term6 = term5 * frequency[0]
      #term7 = term5 * frequency[1]
      term8 = term6 
      #term9 = term7 
      symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
      #symbol_wave2 = self.amplitude * np.cos(term9) + self.amplitude * np.sin(term9)
      rrc_pulse = symbol_wave1[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main

      template = rrc_pulse
      correlation = correlate(signal, template, mode='same')
      peak_threshold = 0.7 * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]

      #self.debug.info_message("pulse detection peak indices:- " + str(peak_indices % pulse_length) )
      self.debug.info_message("pulse detection median:- " + str(int(np.median(peak_indices % pulse_length))) )
      
      calculated_pulse_start = int( (np.median(peak_indices % pulse_length) + (pulse_length/2) ) % pulse_length)
      self.debug.info_message("calculated_pulse_start:- " + str(calculated_pulse_start) )

      test_1 = int( (np.median(peak_indices % (pulse_length)) ) % (pulse_length))
      self.debug.info_message("test_1:- " + str(test_1) )
      test_3 = int( (np.median(peak_indices % (pulse_length*3)) ) % (pulse_length*3))
      self.debug.info_message("test_3:- " + str(test_3) )
      test_4 = int( (np.median(peak_indices % (pulse_length*3)) + (pulse_length/2) ) % (pulse_length*3))
      self.debug.info_message("test_4:- " + str(test_4) )

      return int(calculated_pulse_start), test_1, test_3

    except:
      sys.stdout.write("Exception in pulseDetector: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")



  def detectStandingWavePulseNew(self, signal_array, frequency, pulse_start_index, low_hi_index, detection_type):
    self.debug.info_message("detectStandingWavePulseNew" )

    try:
      if isinstance(signal_array[low_hi_index], np.ndarray) and np.issubdtype(signal_array[low_hi_index].dtype, np.complex128):
        audio_array = (signal_array[low_hi_index].real + signal_array[low_hi_index].imag ) / 2
      else:
        audio_array = signal_array[low_hi_index]

      no_pulse = False
      samples_per_wavelength = self.osmod.sample_rate / frequency[low_hi_index]
      pulse_length           = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      half                   = int(pulse_length / 2)
      location_accuracy      = 6
      self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }
      productCount = 0
      start_pulse = 0
      start_block = 0
      start = 0
      diff_array = np.array([0])
      smoothed_diff_array = np.array([0])
      offset = 0
      index = 0
      aligh_type = ocn.ALIGN_RETAIN_LOCATION
      start_block = 0
      start_pulse = 0

      def acquire_pulse_train_offsets(index, offset, audio_block_index, accuracy):
        nonlocal productCount
        pulse_a = audio_array[offset + (index * pulse_length):offset + ((index+1) * pulse_length)]
        productCount = productCount + 1
        return appendPulseAlignStruct(audio_block_index, pulse_a, offset + (index * pulse_length), accuracy)

      def locateSignalPeaks(template, signal, factor):
        correlation = correlate(signal, template, mode='same')
        peak_threshold = factor * np.max(correlation)
        peak_indices = np.where(correlation > peak_threshold)[0]
        return peak_indices

      def createTemplate():
        self.debug.info_message("createTemplate()")
        num_samples = self.osmod.symbol_block_size
        time = np.arange(num_samples) / self.osmod.sample_rate
        term5 = 2 * np.pi * time
        term8 = term5 * frequency[low_hi_index] 
        symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
        return symbol_wave1[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main

      def createWave():
        self.debug.info_message("createWave()")
        num_samples = self.osmod.symbol_block_size
        time = np.arange(num_samples) / self.osmod.sample_rate
        term5 = 2 * np.pi * time
        term8 = term5 * frequency[low_hi_index] 
        symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
        return symbol_wave1[0:pulse_length] 

      def appendPulseAlignStruct(audio_block_index, pulse, offset_notused, accuracy):

        sigma_signal = 7
        peaks1 = locateSignalPeaks(template_envelope, gaussian_filter(pulse, sigma=sigma_signal), 0.99)
        if len(peaks1) > 0:
          final_value_envelope = peaks1[0]
          no_pulse = False
        else:
          no_pulse = True
          final_value_envelope = 0

        rel_accuracy = 0.0
        abs_accuracy = location_accuracy

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
        return no_pulse

      def findPulseStart():
        self.debug.info_message("findPulseStart()")
        nonlocal offset
        nonlocal index
        nonlocal start_block
        nonlocal start_pulse

        num_full_blocks = int((len(audio_array) ) // self.osmod.symbol_block_size)
        self.non_pulse = []

        """ locate start of pulses """
        found_start = False
        block_count = 0
        while found_start == False and block_count < num_full_blocks:
          offset = (block_count * self.osmod.pulses_per_block) * pulse_length
          for index in range(0,self.osmod.pulses_per_block): 
            is_non_pulse = acquire_pulse_train_offsets(index, offset, 0, 25)
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


      """
      def findPulses():
        self.debug.info_message("findPulses()")
        nonlocal offset
        nonlocal index

        num_full_blocks = int((len(audio_array) ) // self.osmod.symbol_block_size)
        pulses = []

        block_count = 0
        while block_count < num_full_blocks:
          offset = (block_count * self.osmod.pulses_per_block) * pulse_length
          for index in range(0,self.osmod.pulses_per_block): 
            is_non_pulse = acquire_pulse_train_offsets(index, offset, 0, 25)
            if not is_non_pulse:
              pulses.append(index)

          block_count = block_count + 1

        self.debug.info_message("pulses: " + str(pulses))
      """

      def findPeaksOverSample(granularity):
        self.debug.info_message("findPeaksOverSample()")
        nonlocal offset
        nonlocal index

        self.debug.info_message("granularity: " + str(granularity))

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
            is_non_pulse = acquire_pulse_train_offsets(index, offset, 0, granularity)
            if is_non_pulse:
              self.non_pulse.append(index)
            else:
              num_pulses = num_pulses + 1

          if aligh_type == ocn.ALIGN_RETAIN_LOCATION:
            self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets
          elif aligh_type == ocn.ALIGN_MOVE_TO_MID:
            self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets_mid

        self.debug.info_message("pulse_train_alignment_struct: " + str(self.pulse_train_alignment_struct))

        median_index = self.osmod.demodulation_object.getMode(self.pulse_train_alignment_struct['pulses'])
        self.debug.info_message("median_index: " + str(median_index))

        return self.pulse_train_alignment_struct['location_points']

      def calcDiffArray(test_signal):
        nonlocal diff_array
        self.debug.info_message("calcDiffArray()")
        diff_array = np.array([0] * (len(test_signal) - pulse_length), dtype = np.float32) #np.array(len(test_signal) - pulse_length)
        for i in range(0, len(test_signal) - pulse_length):
          diff_array[i] = test_signal[i+77] - test_signal[i]

      def calcSmoothedDiffArray():
        nonlocal smoothed_diff_array
        self.debug.info_message("calcSmoothedDiffArray()")
        sigma_diff = 7
        smoothed_diff_array = gaussian_filter(diff_array, sigma=sigma_diff)

      def identifyStrongestPeaksLocationPointsTriplet(test_signal, location_points):
        self.debug.info_message("identifyStrongestPeaksLocationPointsTriplet()")
        sum_points_1 = []
        sum_points_2 = []
        sum_points_3 = []

        for i in range(0, len(location_points)): 
          location = location_points[i]
          sum_at_location_1 = np.sum(test_signal[np.arange(len(test_signal)) % (3*pulse_length) == location])
          sum_at_location_2 = np.sum(test_signal[np.arange(len(test_signal)) % (3*pulse_length) == location + pulse_length])
          sum_at_location_3 = np.sum(test_signal[np.arange(len(test_signal)) % (3*pulse_length) == location + (2 * pulse_length)])
          sum_points_1.append(sum_at_location_1)
          sum_points_2.append(sum_at_location_2)
          sum_points_3.append(sum_at_location_3)

        self.debug.info_message("sum_points_1: " + str(sum_points_1))
        self.debug.info_message("sum_points_2: " + str(sum_points_2))
        self.debug.info_message("sum_points_3: " + str(sum_points_3))

      def identifyStrongestPeaksLocationPoints(test_signal, location_points):
        self.debug.info_message("identifyStrongestPeaksLocationPoints()")
        sum_points = []
        max_index = 0
        max_sum = 0
        for i in range(0, len(location_points)): 
          location = location_points[i]
          sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % pulse_length == location])
          sum_points.append(sum_at_location)
          if sum_at_location > max_sum:
            max_sum = sum_at_location
            max_index = location

        #self.debug.info_message("sum_points: " + str(sum_points))
        self.debug.info_message("max_index: " + str(max_index))
        return sum_points, max_index

      """
      def identifyPeaksPerPulse(signal, template):
        self.debug.info_message("identifyPeaksPerPulse()")

        samples_per_wavelength = self.osmod.sample_rate / frequency[low_hi_index] 

        samples_per_wavelength_lo = self.osmod.sample_rate / frequency[0] 
        samples_per_wavelength_hi = self.osmod.sample_rate / frequency[1] 
        self.debug.info_message("samples_per_wavelength_lo: " + str(samples_per_wavelength_lo))
        self.debug.info_message("samples_per_wavelength_hi: " + str(samples_per_wavelength_hi))

        pulse_offsets = []
        num_full_blocks = int((len(signal)) // self.osmod.symbol_block_size)
        for block_count in range(0, num_full_blocks): 
          offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) 
          block_signal = signal[offset:offset + (self.osmod.pulses_per_block * pulse_length)]

          for index in range(0, self.osmod.pulses_per_block):
            offset = index * pulse_length
            pulse_signal = block_signal[offset:offset + pulse_length]
            signal_peaks = locateSignalPeaks(template, pulse_signal, 0.7)
            peaks_modulo = signal_peaks % samples_per_wavelength
            peaks_mean   = np.mean(peaks_modulo)
            self.debug.info_message("signal_peaks: " + str(signal_peaks))
            #self.debug.info_message("peaks_modulo: " + str(peaks_modulo))
            #self.debug.info_message("peaks_mean: " + str(peaks_mean))
      """


      def identifyStrongestPeaksPerSubBlock(signal):
        self.debug.info_message("identifyStrongestPeaksPerSubBlock()")

        average_queue = deque()
        sub_divisions = 6

        jumps = []
        num_full_blocks = int((len(signal)) // self.osmod.symbol_block_size)
        for block_count in range(0, num_full_blocks): 
          self.debug.info_message("New sub_block")
          sub_block_offsets = []
          sub_block_length = int((self.osmod.pulses_per_block * pulse_length) / sub_divisions)
          for sub_block in range(0,sub_divisions):
            for sequence in range(0,sub_divisions):
              offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) + (sub_block * sub_block_length) + (sequence * pulse_length)
              sub_block_signal = signal[offset:offset + sub_block_length]
              sum_points = []
              for location in range(0, pulse_length): 
                sum_at_location = np.sum(sub_block_signal[np.arange(len(sub_block_signal)) % pulse_length == location])
                sum_points.append(sum_at_location)

              index_max = np.where(sum_points == np.max(sum_points))[0]
              amount = index_max[0] - half
              if abs(amount) < 5:
                sub_block_offsets.append(amount)

          self.debug.info_message("sub_block_offsets: " + str(sub_block_offsets))
          self.debug.info_message("mean sub_block_offsets: " + str(np.mean(sub_block_offsets)))



      def identifyStrongestPeaksPerBlock(signal):
        self.debug.info_message("identifyStrongestPeaksPerBlock()")

        average_queue = deque()

        block_offsets = []
        jumps = []
        num_full_blocks = int((len(signal)) // self.osmod.symbol_block_size)
        for block_count in range(0, num_full_blocks): 
          offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) 
          block_signal = signal[offset:offset + (self.osmod.pulses_per_block * pulse_length)]
          sum_points = []
          for location in range(0, pulse_length): 
            sum_at_location = np.sum(block_signal[np.arange(len(block_signal)) % pulse_length == location])
            sum_points.append(sum_at_location)

          index_max = np.where(sum_points == np.max(sum_points))[0]
          block_offsets.append(index_max[0] - half)

          if block_count > 0:
            queue_len = len(average_queue)
            if queue_len > 3:
              average_queue.popleft()
            total = 0
            for count in range (0, len(average_queue)):
              total = total + average_queue[count]
            average = total / len(average_queue)
            #self.debug.info_message("queue_len: " + str(queue_len))
            #self.debug.info_message("average: " + str(average))
            if abs(index_max[0] - average) > 3:
              jumps.append(block_count)
              queue_len = len(average_queue)
              for count in range (0, queue_len):
                average_queue.popleft()

          average_queue.append(index_max[0])


          #identifyStrongestPeaksPerPulse(block_signal, index_max[0])

        self.debug.info_message("block_offsets: " + str(block_offsets))
        self.debug.info_message("jumps: " + str(jumps))

        return block_offsets

      def identifyStrongestPeaksPerBlockFraction(signal, divisions, search_range):
        self.debug.info_message("identifyStrongestPeaksPerBlockFraction()")

        search_lo = 0
        search_hi = pulse_length
        block_offsets = []
        num_full_blocks = int((len(signal)) // self.osmod.symbol_block_size)
        for block_count in range(0, num_full_blocks): 
          offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) 
          for division_count in range(0, divisions): 
            division_offset = int(((division_count / divisions) * self.osmod.pulses_per_block) * pulse_length) 
            block_signal = signal[offset + division_offset:offset + division_offset + int((self.osmod.pulses_per_block * pulse_length)/divisions)]
            sum_points = []
            for location in range(search_lo, search_hi): 
              sum_at_location = np.sum(block_signal[np.arange(len(block_signal)) % pulse_length == location])
              #sum_at_location = np.sum(block_signal[(np.arange(len(block_signal)) % pulse_length)/24 == location/24])
              sum_points.append(sum_at_location)

            index_max = (np.where(sum_points == np.max(sum_points))[0]) + search_lo
            block_offsets.append(index_max[0] - half)
          if search_range != 0:
            if block_count == 3:
              median_search = np.median(block_offsets) + half
              search_lo = int(median_search - (search_range / 2))
              search_hi = int(median_search + (search_range / 2))
            elif block_count > 3:
              median_search = int(np.sum(block_offsets[-5:]) / 5) + half
              search_lo = int(median_search - (search_range / 2))
              search_hi = int(median_search + (search_range / 2))
 

        self.debug.info_message("block_offsets fraction: " + str(block_offsets))

        return block_offsets

      """ method obtains four values per block for each block along the pulse train at locations: blocklen * 0.25, blocklen * 0.5, blocklen * 0.75 and blocklen * 1 """
      def getDopplerShiftFourths(signal, search_range):
        self.debug.info_message("getDopplerShiftFourths()")

        block_offsets_sigma_template = 3
        #block_offsets_sigma_template = 1.27

        override_block_offsets_sigma = self.osmod.form_gui.window['cb_overridedopplerfourthssigma'].get()
        if override_block_offsets_sigma:
          block_offsets_sigma_template = float(self.osmod.form_gui.window['in_dopplerfourthssigma'].get())

        quarter_offset = int((self.osmod.pulses_per_block / 4) * pulse_length)
        test_signal = gaussian_filter(np.abs(signal[pulse_start_index:]), sigma=block_offsets_sigma_template)

        """ split block in halves and obtain value at 0.25 and 0.75"""
        block_offsets_1 = identifyStrongestPeaksPerBlockFraction(test_signal, 2, search_range)
        """ split block in halves and obtain value at 0.5 and 1 """
        block_offsets_2 = identifyStrongestPeaksPerBlockFraction(test_signal[quarter_offset:], 2, search_range)
        
        doppler_shift = []
        for count in range(0, min(len(block_offsets_1), len(block_offsets_2)), 2 ):
          doppler_shift.append(block_offsets_1[count])
          doppler_shift.append(block_offsets_2[count])
          doppler_shift.append(block_offsets_1[count+1])
          doppler_shift.append(block_offsets_2[count+1])

        self.debug.info_message("doppler_shift: " + str(doppler_shift))

        return doppler_shift



      def findWhichSeries(data_param, mid_param, id_num, deviation_amount):
        nonlocal which_series
        nonlocal median_gradient

        data_len = len(data_param)
        rel_accuracy = 0.0
        abs_accuracy = deviation_amount
        last_in_series_1_x = mid_param
        last_in_series_1_y = data_param[mid_param]
        which_series[mid_param] = id_num
        for i in range(mid_param+1, data_len):
          y_diff = data_param[i] - data_param[last_in_series_1_x]
          x_diff = i-mid_param
          gradient = np.mean(median_gradient[i-5:i])
          #self.debug.info_message("mean gradient: " + str(gradient))
          #self.debug.info_message("x_diff: " + str(x_diff))
          #self.debug.info_message("y_diff: " + str(y_diff))
          #self.debug.info_message("y_diff / x_diff: " + str(y_diff / x_diff))
          if  math.isclose( y_diff / x_diff, gradient, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
            #self.debug.info_message("Adding point to series 1")
            which_series[i] = id_num
            last_in_series_1_x = i
            last_in_series_1_y = data_param[i]
        last_in_series_1_x = mid_param
        last_in_series_1_y = data_param[mid_param]
        #for i in range(mid_param-1, 0, -1):
        for i in range(mid_param-1, -1, -1):
          y_diff = data_param[i] - data_param[last_in_series_1_x]
          x_diff = i-mid_param
          gradient = np.mean(median_gradient[i:i+5])
          if  math.isclose( y_diff / x_diff, gradient, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
            which_series[i] = id_num
            last_in_series_1_x = i
            last_in_series_1_y = data_param[i]


      def recalcDopplerPoint(start, end, inc, offset_length, signal, part_length, last_known_good_x, max_index, deviation):
        self.debug.info_message("recalcDopplerPoint()")

        nonlocal median_gradient
        nonlocal which_series
        nonlocal doppler_shift
        try:
          #for part_count in range(0, num_full_parts * num_divisions): 
          for part_count in range(start, end, inc): 
            #self.debug.info_message("which_series[part_count]: " + str(which_series[part_count]))
            #if which_series[part_count] != max_index:
            if True:
              self.debug.info_message("match")
              offset = part_count * offset_length
              part_signal = signal[offset:offset + part_length]
              sum_points = []
              gradient = np.mean(median_gradient[part_count:part_count+5])
              expected_value = (doppler_shift[last_known_good_x] + half) - (gradient * ( last_known_good_x - part_count))
              self.debug.info_message("expected_value: " + str(expected_value))
              search_lo = int(expected_value - deviation)
              search_hi = int(expected_value + deviation)
              self.debug.info_message("search_lo: " + str(search_lo))
              self.debug.info_message("search_hi: " + str(search_hi))
              for location in range(search_lo, search_hi): 
                self.debug.info_message("finding sum at location")
                sum_at_location = np.sum(part_signal[np.arange(len(part_signal)) % pulse_length == location])
                #sum_at_location = np.sum(part_signal[(np.arange(len(part_signal)) % pulse_length)/16 == location/16])
                sum_points.append(sum_at_location)
                #doppler_shift[part_count] = sum_at_location
              self.debug.info_message("sum_points: " + str(sum_points))

              index_max = (np.where(sum_points == np.max(sum_points))[0]) + search_lo
              #part_offsets.append(index_max[0] - half)
              doppler_shift[part_count] = index_max[0] - half
              self.debug.info_message("recalculating point: doppler_shift[" + str(part_count) + "] = " + str(index_max[0] - half))
              last_known_good_x = part_count
              which_series[part_count] = max_index

        except:
          self.debug.error_message("Exception in recalcDopplerPoint: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


      def identifyDominantSeries(data, signal, num_full_parts, num_divisions, num_pulses):
        self.debug.info_message("identifyDominantSeries()")

        nonlocal median_gradient
        nonlocal which_series
        nonlocal doppler_shift

        try:
          deviation_amount = 1.0
          #deviation_amount = 0.9

          part_length   = num_pulses * pulse_length
          offset_length = int(part_length / num_divisions)

          data_len = len(data)
          mid = int(data_len / 2)

          median_gradient = [0] * data_len
          which_series = [0] * data_len

          gradients = [0] * 10
          """ find initial gradients """
          for i in range(mid, data_len):
            for g in range(0,10):
              gradients[g] = data[i-g] - data[i-g-1]
            median_gradient[i] = np.median(gradients)
          #for i in range(mid-1, 0, -1):
          for i in range(mid-1, -1, -1):
            for g in range(0,10):
              gradients[g] = data[i+g+1] - data[i+g]
            median_gradient[i] = np.median(gradients)

          findWhichSeries(data, mid, 1, deviation_amount)

          for j in range(mid, data_len):
            if which_series[j] == 0:
              which_series[j] = 2
              break
          findWhichSeries(data, j, 2, deviation_amount)

          for j in range(mid, data_len):
            if which_series[j] == 0:
              which_series[j] = 3
              break
          findWhichSeries(data, j, 3, deviation_amount)

          self.debug.info_message("bincount(which_series): " + str(np.bincount(which_series)))
          max_index = int(np.argmax(np.bincount(which_series)[1:]))+1
          self.debug.info_message("max_index: " + str(max_index))

          self.dict_saved_data['doppler before convert'] = doppler_shift.copy()

          last_known_good_x = mid
          for j in range(mid+1, data_len):
            if which_series[j] == max_index:
              last_known_good_x = j
              break


          deviation = 3

          self.debug.info_message("recalcDopplerPoint first call")

          #recalcDopplerPoint(mid, 0, -1, offset_length, signal, part_length, last_known_good_x, max_index)
          recalcDopplerPoint(mid, -1, -1, offset_length, signal, part_length, last_known_good_x, max_index, deviation)

          self.dict_saved_data['doppler mid convert'] = doppler_shift.copy()


          last_known_good_x = mid
          #for j in range(mid-1, 0, -1):
          for j in range(mid-1, -1, -1):
            if which_series[j] == max_index:
              last_known_good_x = j
              break

          self.debug.info_message("recalcDopplerPoint second call")

          recalcDopplerPoint(mid, len(doppler_shift)-6, 1, offset_length, signal, part_length, last_known_good_x, max_index, deviation)

          self.dict_saved_data['doppler after convert'] = doppler_shift.copy()


          deviation = 2


          self.debug.info_message("recalcDopplerPoint third call")

          recalcDopplerPoint(0, len(doppler_shift)-6, 1, offset_length, signal, part_length, 0, max_index, deviation)

          self.dict_saved_data['doppler after convert 2 '] = doppler_shift.copy()



          #self.debug.info_message("recalcDopplerPoint fourth call")

          #recalcDopplerPoint(len(doppler_shift)-6, -1, -1, offset_length, signal, part_length, len(doppler_shift)-1, max_index, deviation)

          #self.dict_saved_data['doppler after convert 3 '] = doppler_shift.copy()


        except:
          self.debug.error_message("Exception in identifyDominantSeries: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


        self.debug.info_message("median_gradient: " + str(median_gradient))
        self.debug.info_message("which_series: " + str(which_series))

        return [median_gradient]

      def getDopplerShiftPulsesN(signal, search_range, num_pulses):
        self.debug.info_message("getDopplerShiftPulsesN()")
        nonlocal doppler_shift

        block_offsets_sigma_template = 3
        override_block_offsets_sigma = self.osmod.form_gui.window['cb_overridedopplerfourthssigma'].get()
        if override_block_offsets_sigma:
          block_offsets_sigma_template = float(self.osmod.form_gui.window['in_dopplerfourthssigma'].get())

        part_length   = num_pulses * pulse_length

        self.debug.info_message("part_length: " + str(part_length))
        self.debug.info_message("num_pulses: " + str(num_pulses))
        self.debug.info_message("pulse_length: " + str(pulse_length))

        num_divisions = 2
        offset_length = int(part_length / num_divisions)
        signal = gaussian_filter(np.abs(signal[pulse_start_index:]), sigma=block_offsets_sigma_template)

        self.debug.info_message("offset_length: " + str(offset_length))

        search_lo = 0
        search_hi = pulse_length
        part_offsets = []
        num_full_parts = int((len(signal)) // part_length)

        self.debug.info_message("num_full_parts: " + str(num_full_parts))

        doppler_shift = []
        for part_count in range(0, num_full_parts * num_divisions): 
          offset = part_count * offset_length
          part_signal = signal[offset:offset + part_length]
          sum_points = []
          for location in range(search_lo, search_hi): 
            #sum_at_location = np.sum(part_signal[np.arange(len(part_signal)) % pulse_length == location])
            sum_at_location = np.sum(part_signal[(np.arange(len(part_signal)) % pulse_length)/16 == location/16])
            sum_points.append(sum_at_location)

          index_max = (np.where(sum_points == np.max(sum_points))[0]) + search_lo
          part_offsets.append(index_max[0] - half)
          doppler_shift.append(index_max[0] - half)
          if search_range != 0:
            if part_count == 3:
              median_search = np.median(part_offsets) + half
              search_lo = int(median_search - (search_range / 2))
              search_hi = int(median_search + (search_range / 2))
            elif part_count > 3:
              median_search = int(np.sum(part_offsets[-5:]) / 5) + half
              search_lo = int(median_search - (search_range / 2))
              search_hi = int(median_search + (search_range / 2))



        dominant_series = identifyDominantSeries(doppler_shift, signal, num_full_parts, num_divisions, num_pulses)
        self.dict_saved_data['dominant series: gradient'] = dominant_series[0]


        self.debug.info_message("doppler_shift: " + str(doppler_shift))

        return doppler_shift





      def filterData(x_param, y_param, outliers_param):
        cleaned_data_x = []
        cleaned_data_y = []
        outliers_count = 0
        outliers_len = len(outliers_param)
        for i in range(0, len(x_param)):
          if outliers_count < outliers_len and x_param[i] == outliers_param[outliers_count]:
            outliers_count = outliers_count + 1
          else:
            cleaned_data_x.append(x_param[i])
            cleaned_data_y.append(y_param[i])

        return cleaned_data_x, cleaned_data_y

      def findOutliersMedianZ(data_param):
        self.debug.info_message("findOutliersMedianZ()")

        median_data = []
        median = np.median(data_param)
        median_deviation = np.median(np.abs(data_param - median))
        
        outlier_indices = np.array([])
        if median_deviation != 0:
          self.debug.info_message("median_deviation != 0")
          median_data = 0.6745 * (data_param - median) / median_deviation
          outlier_indices = np.where(np.abs(median_data) > 1.5)[0]
        else:
          self.debug.info_message("median_deviation == 0")
          median_data = 0.6745 * (data_param - median) / 1
          outlier_indices = np.where(np.abs(median_data) > 1.2)[0]

        return [median_data, outlier_indices]


      def interpolatePulsesCheby(data):
        self.debug.info_message("interpolatePulsesCheby()")

        x = np.linspace(0, len(data)-1, len(data))
        x_smooth = np.linspace(-1, len(data), (len(data) + 1) * 100)
        cheby = self.osmod.modulation_object.chebyshevCurveInterpolation(x, x_smooth, data, 10)
        return cheby

        #interpolated_signal[min(x):max(x)] = cheby
        #return interpolated_signal



      def interpolatePulsesSpline(data, pulses_per_offset):
        self.debug.info_message("interpolatePulsesSpline()")
        self.debug.info_message("pulses_per_offset: " + str(pulses_per_offset))

        try:

          spline_smoothing = 130

          override_spline_smoothing = self.osmod.form_gui.window['cb_overridesplinesmoothing'].get()
          if override_spline_smoothing:
            spline_smoothing = float(self.osmod.form_gui.window['in_splinesmoothing'].get())

          outliers = findOutliersMedianZ(data)
          self.debug.info_message("median data: " + str(outliers[0]))
          self.debug.info_message("outliers: " + str(outliers[1]))

          #pulses_per_offset = int(self.osmod.pulses_per_block / 4)
          #spline_smoothing = float(self.osmod.form_gui.window['in_analysissplinesmoothvalue'].get())
          #spline_smoothing = 40
          #spline_smoothing = 200
 
          #x = np.linspace(0, len(data), len(data))

          x = np.linspace(0, len(data)-1, len(data))
          self.debug.info_message("x values: " + str(x))

          #x_test = np.linspace(0, len(data)-1, (len(data)*2)-1)
          x_test = np.linspace(-1/2, len(data)-1 + (1/2), ((len(data))*2) + 1)
          self.debug.info_message("test x values: " + str(x_test))


          filtered_x, filtered_y = filterData(x, data, outliers[1])
          self.debug.info_message("filtered_x: " + str(filtered_x))
          self.debug.info_message("filtered_y: " + str(filtered_y))

          #x_smooth = np.linspace(-1, len(data), (len(data) + 1) * pulses_per_offset)
          #x_smooth = np.linspace(0, len(data)-1, (len(data) * pulses_per_offset) - 1)
          x_smooth = np.linspace(-1/2, len(data) - (1/2), (len(data) * pulses_per_offset) + 1 )

          self.debug.info_message("smoothed x values: " + str(x_smooth))

          doppler_curvefit_checked = self.osmod.form_gui.window['cb_overridedopplercurvefit'].get()
          if doppler_curvefit_checked:
            interpolation_type = self.osmod.form_gui.window['combo_dopplercurvefittype'].get() 
            spline_smoothing = int(self.osmod.form_gui.window['in_dopplersmoothing'].get())

            if interpolation_type == 'B-Spline':
              y_smooth = self.osmod.modulation_object.BetaSplineCurveInterpolation(x, x_smooth, data, spline_smoothing)
            elif interpolation_type == 'Cubic-Spline':
              y_smooth = self.osmod.modulation_object.CubicSplineCurveInterpolation(x, x_smooth, data, spline_smoothing)
            elif interpolation_type == 'Pchip':
              y_smooth = self.osmod.modulation_object.PchipCurveInterpolation(x, x_smooth, data, spline_smoothing)
            elif interpolation_type == 'Chebyshev':
              y_smooth = self.osmod.modulation_object.chebyshevCurveInterpolation(x, x_smooth, data, spline_smoothing)
          else:
            if self.osmod.doppler_pulse_interpolation == 'B-Spline':
              tck = splrep(filtered_x, filtered_y, s=spline_smoothing)
              y_smooth = splev(x_smooth, tck)
            elif self.osmod.doppler_pulse_interpolation == 'Pchip':
              spline_smoothing = 50
              y_smooth = self.osmod.modulation_object.PchipCurveInterpolation(x, x_smooth, data, spline_smoothing)
            elif self.osmod.doppler_pulse_interpolation == 'Chebyshev':
              y_smooth = self.osmod.modulation_object.chebyshevCurveInterpolation(x, x_smooth, data, 12)

          self.debug.info_message("y_smooth: " + str(y_smooth))

          return y_smooth
          #self.drawPulseTrainCharts(y_smooth, chart_type, window, form_gui, canvas_name, chart_name, plot_color, False, fixed_scale, scaling_params)

          #self.calcDeltas(data)

        except:
          self.debug.error_message("Exception in interpolatePulsesSpline: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


      #def identifyStrongestPeaks(test_signal):
      def identifyStrongestPeaks(test_signal, resolution_increment, consolidation_amount, specify_range, range_lohi):
        self.debug.info_message("identifyStrongestPeaks()")
        sum_points = []
        #for location in range(0, pulse_length): 
        if specify_range == False:
          range_lo = 0
          range_hi = pulse_length
        else:
          range_lo = range_lohi[0]
          range_hi = range_lohi[1]

        #for location in range(0, pulse_length, resolution_increment): 
        for location in range(range_lo, range_hi, resolution_increment): 
          #sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % pulse_length == location])
          sum_at_location = np.sum(test_signal[(np.arange(len(test_signal)) % pulse_length)/consolidation_amount == location/consolidation_amount])
          sum_points.append(sum_at_location)

        #self.debug.info_message("sum_points: " + str(sum_points))
        self.debug.info_message("max sum_points: " + str(np.max(sum_points)))
        self.debug.info_message("min sum_points: " + str(np.min(sum_points)))

        index_max = np.where(sum_points == np.max(sum_points))[0]
        index_min = np.where(sum_points == np.min(sum_points))[0]
        self.debug.info_message("index_max: " + str(index_max))
        self.debug.info_message("index_min: " + str(index_min))
        self.debug.info_message("diff: " + str(index_max - index_min))

        return index_max[0], index_min[0]

      def calcReversals(samples_array):
        self.debug.info_message("calcReversals()")
        reversal_array_up = []
        reversal_array_down = []
        last_diff = samples_array[0]
        if samples_array[1] > last_diff:
          trend = ocn.DIFF_TREND_UP
        else:
          trend = ocn.DIFF_TREND_DOWN
        for i in range(1, len(samples_array) ):
          change = samples_array[i] - last_diff
          last_diff = samples_array[i]
          if change < 0 and trend == ocn.DIFF_TREND_UP:
            reversal_array_down.append(i % (3*pulse_length))
            trend = ocn.DIFF_TREND_DOWN
          elif change > 0 and trend == ocn.DIFF_TREND_DOWN:
            reversal_array_up.append(i % (3*pulse_length))
            trend = ocn.DIFF_TREND_UP

        self.debug.info_message("reversal_array_up: " + str(reversal_array_up))
        self.debug.info_message("reversal_array_down: " + str(reversal_array_down))

      """
      def identifyPulseTrain2(signal):
        self.debug.info_message("identifyPulseTrain()")

        ratio = np.max(signal) / np.max(template_envelope)
        max_signal = np.max(signal)

        num_full_blocks = int((len(signal)) // self.osmod.symbol_block_size)
        for block_count in range(0, num_full_blocks): 
          offset = ((block_count * self.osmod.pulses_per_block) * pulse_length) 

          for index in range(0,self.osmod.pulses_per_block): 
            test_pulse = signal[offset + (index * pulse_length):offset + ((index+1) * pulse_length)]
            #correlation_coef = np.corrcoef(test_pulse / ratio, template_envelope)[0,1]
            correlation = ((test_pulse[half]/max_signal) - (test_pulse[0]/max_signal)) * 10000
            self.debug.info_message("index : " + str(index) )
            self.debug.info_message("correlation : " + str(correlation) )
      """

      def identifyPulseTrain3(signal):
        self.debug.info_message("identifyPulseTrain3()")

        list_factor = self.osmod.interpolator.test_list_factor

        sum_points = []
        for pulse_num in range(0, self.osmod.pulses_per_block): 
          location = (pulse_num * pulse_length)+half
          sum_at_location = np.sum(signal[(np.arange(len(signal)) % (pulse_length * self.osmod.pulses_per_block))+half == location])
          #sum_at_location = np.sum(signal[int(((np.arange(len(signal)) % (pulse_length * self.osmod.pulses_per_block))+half)/10) == int(location/10)])
          #sum_at_location = np.sum(test_signal[(np.arange(len(test_signal)) % pulse_length)/consolidation_amount == location/consolidation_amount])

          sum_points.append(sum_at_location)

        #at_max = 0.94 * np.max(sum_points)
        at_max = list_factor * np.max(sum_points)

        index_max = np.where(sum_points >= at_max)[0]
        self.debug.info_message("index_max: " + str(index_max))
        #self.debug.info_message("sum_points: " + str(sum_points) )

        return_list = index_max
        self.debug.info_message("return_list: " + str(return_list) )
        #index_max = np.where(sum_points == np.max(sum_points))[0]
        #block_offsets.append(index_max[0] - half)

        return return_list.tolist()

      def identifyPulseTrain(signal):
        self.debug.info_message("identifyPulseTrain()")

        #correlation_factor = 2
        correlation_factor = 2.2
        #correlation_factor = 2.7
        ratio = np.max(signal) / np.max(template_envelope)

        correlation = correlate(signal/ratio, template_envelope, mode='same')
        peak_threshold = correlation_factor * np.mean(correlation)
        peak_indices = np.where(correlation > peak_threshold)[0]

        #self.debug.info_message("correlation : " + str(correlation) )
        #self.debug.info_message("peak_indices : " + str(peak_indices) )
        #self.debug.info_message("len(peak_indices) : " + str(len(peak_indices)) )
        #self.debug.info_message("correlation peak_indices: " + str(correlation[peak_indices]) )

        test = np.where(np.abs(peak_indices % pulse_length) < 3 ) [0]
        #self.debug.info_message("test: " + str(test) )
        self.debug.info_message("len(test): " + str(len(test)) )
        all_non_interpolated = (peak_indices[test] // pulse_length) % self.osmod.pulses_per_block 
        self.debug.info_message("all_non_interpolated: " + str(all_non_interpolated) )
        bincount = np.bincount(all_non_interpolated)
        self.debug.info_message("bincount: " + str(bincount) )
        persistent = []
        for i in range(0, len(bincount)):
          if bincount[i] > 3:
            persistent.append(i)
        return persistent


      sigma_template = 7
      #wave              = createWave()
      rrc_pulse         = createTemplate()
      template_wave     = rrc_pulse
      template_envelope = gaussian_filter(rrc_pulse, sigma=sigma_template)

      if detection_type == ocn.LOCATE_PULSE_START_INDEX:
        self.debug.info_message("LOCATE_PULSE_START_INDEX")
        findPulseStart()

        #pulse_start_sigma_template = 7
        pulse_start_sigma_template = self.osmod.pulse_start_sigma

        override_pulse_start_sigma = self.osmod.form_gui.window['cb_overridepulsestartsigma'].get()
        if override_pulse_start_sigma:
          pulse_start_sigma_template = float(self.osmod.form_gui.window['in_pulsestartsigma'].get())

        test_signal = gaussian_filter(np.abs(audio_array[start:]), sigma=pulse_start_sigma_template)

        if self.osmod.sample_rate == 8000:
          start_index_2, _ = identifyStrongestPeaks(test_signal, 1, 1, False, (0,0))
        elif self.osmod.sample_rate == 48000:
          #start_index_2, _ = identifyStrongestPeaks(test_signal, 8, 1, False, (0,0))

          self.debug.info_message("LOCATE_PULSE_START_INDEX step 1")
          first_inc = 10
          #first_inc = 20
          #first_inc = 30
          start_index_first_pass, _ = identifyStrongestPeaks(test_signal, first_inc, 1, False, (0,0))
          span = first_inc
          self.debug.info_message("LOCATE_PULSE_START_INDEX step 2")
          second_inc = 3
          #second_inc = 6
          #second_inc = 8
          start_index_second_pass, _ = identifyStrongestPeaks(test_signal, second_inc, 1, True, (start_index_first_pass - span, start_index_first_pass + span))
          span = second_inc
          self.debug.info_message("LOCATE_PULSE_START_INDEX step 3")
          start_index_2, _ = identifyStrongestPeaks(test_signal, 1, 1, True, (start_index_second_pass - span, start_index_second_pass + span))

        signal_start = (start_pulse * pulse_length) + ((start_index_2 + half) % pulse_length)
        self.debug.info_message("signal start at: " + str(signal_start))
        return [signal_start]

      elif detection_type == ocn.LOCATE_PULSE_TRAIN:
        self.debug.info_message("LOCATE_PULSE_TRAIN")

        #sigma_template = 7
        #template_envelope = gaussian_filter(rrc_pulse, sigma=sigma_template)

        #pulse_train_sigma_template = 1.6
        pulse_train_sigma_template = 5.0

        override_pulse_train_sigma = self.osmod.form_gui.window['cb_overridepulsetrainsigma'].get()
        if override_pulse_train_sigma:
          pulse_train_sigma_template = float(self.osmod.form_gui.window['in_pulsetrainsigma'].get())

        test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=pulse_train_sigma_template)

        pulse_train = identifyPulseTrain3(test_signal)
        #pulse_train = identifyPulseTrain(test_signal)

        self.debug.info_message("pulse_train: " + str(pulse_train) )
        return [0, pulse_train]

      elif detection_type == ocn.CALC_BLOCK_OFFSETS:
        self.debug.info_message("CALC_BLOCK_OFFSETS")

        smoothing = ocn.SIGNAL_ENVELOPE_GAUSSIAN
        #smoothing = ocn.SIGNAL_ENVELOPE_SAVITZKY_GOLAY
        #smoothing = ocn.SIGNAL_ENVELOPE_HILBERT_GAUSS
        #smoothing = ocn.SIGNAL_ENVELOPE_HILBERT_SG

        block_offsets_sigma_template = 1.25
        #block_offsets_sigma_template = 1.27
        override_block_offsets_sigma = self.osmod.form_gui.window['cb_overrideblockoffsetssigma'].get()
        if override_block_offsets_sigma:
          block_offsets_sigma_template = float(self.osmod.form_gui.window['in_blockoffsetssigma'].get())

        if smoothing == ocn.SIGNAL_ENVELOPE_GAUSSIAN:
          test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=block_offsets_sigma_template)
        elif smoothing == ocn.SIGNAL_ENVELOPE_SAVITZKY_GOLAY:
          window_length = 3 #must be odd number
          polyorder = 1
          test_signal = savgol_filter(np.abs(audio_array[pulse_start_index:]), window_length, polyorder)
        elif smoothing == ocn.SIGNAL_ENVELOPE_HILBERT:
          test_signal = hilbert( np.abs(audio_array[pulse_start_index:]) )
        elif smoothing == ocn.SIGNAL_ENVELOPE_HILBERT_GAUSS:
          analytic_signal = hilbert( np.abs(audio_array[pulse_start_index:]) )
          sigma_template = 2
          test_signal = gaussian_filter(analytic_signal, sigma=sigma_template)
        elif smoothing == ocn.SIGNAL_ENVELOPE_HILBERT_SG:
          analytic_signal = hilbert( np.abs(audio_array[pulse_start_index:]) )
          window_length = 3 #must be odd number
          polyorder = 1
          test_signal = savgol_filter(analytic_signal, window_length, polyorder)


        if self.osmod.sample_rate == 8000:
          block_offsets = identifyStrongestPeaksPerBlock(test_signal)
        elif self.osmod.sample_rate == 48000:
          num_full_blocks = int((len(test_signal)) // self.osmod.symbol_block_size)
          block_offsets = [0] * num_full_blocks

        self.dict_saved_data['block offsets'] = block_offsets

        return [0, [], block_offsets]

      elif detection_type == ocn.CALC_BLOCK_OFFSETS_FRACTION:
        self.debug.info_message("CALC_BLOCK_OFFSETS")

        smoothing = ocn.SIGNAL_ENVELOPE_GAUSSIAN

        #block_offsets_sigma_template = 1.25
        #block_offsets_sigma_template = 3.5
        block_offsets_sigma_template = 3
        override_block_offsets_sigma = self.osmod.form_gui.window['cb_overrideblockoffsetssigma'].get()
        if override_block_offsets_sigma:
          block_offsets_sigma_template = float(self.osmod.form_gui.window['in_blockoffsetssigma'].get())

        if smoothing == ocn.SIGNAL_ENVELOPE_GAUSSIAN:
          test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=block_offsets_sigma_template)

        self.dict_saved_data['smoothed signal'] = test_signal

        block_offsets = identifyStrongestPeaksPerBlock(test_signal)
        self.dict_saved_data['block offsets'] = block_offsets

        block_offsets = identifyStrongestPeaksPerBlockFraction(test_signal, 2, 0)
        #block_offsets = identifyStrongestPeaksPerBlockFraction(test_signal, 4, 0)
        self.dict_saved_data['block offsets half'] = block_offsets

        #doppler_fourths = getDopplerShiftFourths(audio_array)
        #self.dict_saved_data['block offsets fourth'] = doppler_fourths

        #pulses = interpolatePulsesSpline(doppler_fourths)
        #self.dict_saved_data['interpolated pulse offsets'] = pulses


        return [0, [], block_offsets]


      elif detection_type == ocn.LOCATE_PULSES_PEAK:
        self.debug.info_message("LOCATE_PULSES_PEAK")

        sigma_template = 7
        test_signal = gaussian_filter(np.abs(audio_array), sigma=sigma_template)
        peak, _ = identifyStrongestPeaks(test_signal, 1, 1, False, (0,0))
        return [peak]

      elif detection_type == ocn.CALC_PULSE_OFFSETS:
        self.debug.info_message("CALC_PULSE_OFFSETS")

        median_gradient = []
        which_series = []
        doppler_shift = []

        #doppler_fourths = getDopplerShiftFourths(audio_array, int(pulse_length / 8))
        #self.dict_saved_data['block offsets fourth'] = doppler_fourths


        #doppler_fourths_b = getDopplerShiftFourths(audio_array, int(pulse_length / 24))
        #doppler_fourths_b = getDopplerShiftPulsesN(audio_array, int(pulse_length / 8), int(self.osmod.pulses_per_block / 2))

        """ length of scan (relative to block length) used to identify location of max pulse """
        block_scan_fraction = 1/4
        block_scan_range = int(pulse_length / 16)

        if self.osmod.sample_rate == 8000:
          doppler_fourths = getDopplerShiftFourths(audio_array, int(pulse_length / 8))
          self.dict_saved_data['block offsets fourth'] = doppler_fourths
          #doppler_fourths_b = getDopplerShiftPulsesN(audio_array, block_scan_range, int(self.osmod.pulses_per_block * block_scan_fraction))
          #self.dict_saved_data['block offsets fourth b'] = doppler_fourths_b
          pulses_per_offset = int(self.osmod.pulses_per_block / 4)
        elif self.osmod.sample_rate == 48000:
          #number_of_pulses = int(self.osmod.pulses_per_block * block_scan_fraction)
          number_of_pulses = int((self.osmod.pulses_per_block * block_scan_fraction) // 6) * 6
          doppler_fourths = getDopplerShiftPulsesN(audio_array, block_scan_range, number_of_pulses)
          self.dict_saved_data['block offsets fourth b'] = doppler_fourths
          #pulses_per_offset = int(self.osmod.pulses_per_block / 8)
          pulses_per_offset = int(number_of_pulses / 2)


        #dominant_series = identifyDominantSeries(doppler_fourths_b)
        #self.dict_saved_data['dominant series 1'] = dominant_series[0]



        #pulses = interpolatePulsesCheby(doppler_fourths)
        #self.dict_saved_data['interpolated pulse offsets'] = pulses


        pulses = interpolatePulsesSpline(doppler_fourths, pulses_per_offset)
        self.dict_saved_data['interpolated pulse offsets'] = pulses


        return [pulse_start_index, [], [], pulses]


      elif detection_type == ocn.FIND_TRIPLET_MAX_POINT:
        self.debug.info_message("FIND_TRIPLET_MAX_POINT")

        sigma_template = 4
        test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=sigma_template)

        location_points = findPeaksOverSample(3)

        identifyStrongestPeaksLocationPointsTriplet(test_signal, location_points)


      elif detection_type == ocn.PULSE_DETECT_INFO:
        self.debug.info_message("PULSE_DETECT_INFO")

        #identifyPeaksPerPulse(audio_array, template_wave)
        #identifyPeaksPerPulse(audio_array, wave)

        sigma_template = 4
        test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=sigma_template)
        identifyStrongestPeaksPerSubBlock(test_signal)

        location_points = findPeaksOverSample(3)

        magnitude_points, start_index_1 = identifyStrongestPeaksLocationPoints(test_signal, location_points)
        identifyStrongestPeaksLocationPointsTriplet(test_signal, location_points)

        sigma_template = 7
        test_signal = gaussian_filter(np.abs(audio_array), sigma=sigma_template)

        calcDiffArray(test_signal)
        calcSmoothedDiffArray()
        _, start_index_3 = identifyStrongestPeaks(diff_array, 1, 1, False, (0,0))
        _, start_index_4 = identifyStrongestPeaks(smoothed_diff_array, 1, 1, False, (0,0))

        calcReversals(smoothed_diff_array)

        self.debug.info_message("start_index_1: " + str(start_index_1))
        #self.debug.info_message("start_index_2: " + str(start_index_2))
        self.debug.info_message("start_index_3: " + str(start_index_3))
        self.debug.info_message("start_index_4: " + str(start_index_4))

        self.debug.info_message("test code 1")
        test  = np.where((diff_array % pulse_length)  >= 0.99 * np.max(diff_array) ) [0]
        test2 = np.where((diff_array % pulse_length)  <= 0.99 * np.min(diff_array) ) [0]
        self.debug.info_message("test: " + str(test))
        self.debug.info_message("test2: " + str(test2))
        for i in range(0, 100):
          self.debug.info_message(" " + str(test[i] % pulse_length))
 
        self.debug.info_message("test code 2")
        test  = np.where((smoothed_diff_array % pulse_length)  >= 0.99 * np.max(smoothed_diff_array) ) [0]
        test2 = np.where((smoothed_diff_array % pulse_length)  <= 0.99 * np.min(smoothed_diff_array) ) [0]
        self.debug.info_message("test: " + str(test))
        self.debug.info_message("test2: " + str(test2))
        for i in range(0, 100):
          self.debug.info_message(" " + str(test[i] % pulse_length))


    except:
      sys.stdout.write("Exception in detectStandingWavePulseNew: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def detectStandingWavePulse(self, signal, frequency, pulse_start_index, test_1, test_3):
    self.debug.info_message("detectStandingWavePulse" )

    try:

      if isinstance(signal, np.ndarray) and np.issubdtype(signal.dtype, np.complex128):
        self.debug.info_message("signal is complex128 in detectStandingWavePulse" )
        signal = (signal.real + signal.imag ) / 2

      #threshold_1 = 0.7
      threshold_1 = 0.7

      """ create an RRC pulse sample """
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      self.debug.info_message("pulse_length: " + str(pulse_length))
      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate
      term5 = 2 * np.pi * time

      freq_index = 0
      term6 = term5 * frequency[freq_index]
      term8 = term6 
      symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
      rrc_pulse = symbol_wave1[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main
      template = rrc_pulse
      correlation = correlate(signal, template, mode='same')

      """ find location of normal pulses """
      peak_threshold = threshold_1 * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]
      #self.debug.info_message("normal pulse detection peak indices 1:- " + str(peak_indices % pulse_length) )
      pulse_start_pulselength = int( (self.osmod.demodulation_object.getMode(peak_indices % pulse_length) ) % pulse_length)
      self.debug.info_message("pulse_start_pulselength:- " + str(pulse_start_pulselength) )

      """ find location of standing wave pulses"""
      peak_threshold = threshold_1 * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]
      peak_threshold_standingwave = 0.97 * np.max(correlation[peak_indices])
      peaks_standingwave, _ = find_peaks(correlation, height = peak_threshold_standingwave)
      #self.debug.info_message("standingwave pulse detection peak indices 3:- " + str(peaks_standingwave % (pulse_length * 3)) )
      pulse_start_threepulselength = int( (self.osmod.demodulation_object.getMode(peaks_standingwave % (pulse_length * 3)) ) % (pulse_length * 3))
      self.debug.info_message("pulse_start_threepulselength:- " + str(pulse_start_threepulselength) )

      """ find location of standing wave pulses"""
      peak_threshold = threshold_1 * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]
      peak_threshold_standingwave = 0.97 * np.max(correlation[peak_indices])
      peaks_standingwave, _ = find_peaks(correlation, height = peak_threshold_standingwave)
      #self.debug.info_message("standingwave pulse detection peak indices 1.5:- " + str(peaks_standingwave % int(pulse_length * 1.5)) )
      pulse_start_onepointfivepulselength = int( (self.osmod.demodulation_object.getMode(peaks_standingwave % int(pulse_length * 1.5)) ) % int(pulse_length * 1.5))
      self.debug.info_message("pulse_start_onepointfivepulselength:- " + str(pulse_start_onepointfivepulselength) )

      rel_accuracy = 0.0
      abs_accuracy = 7
      shift_required = 0
      test = pulse_start_onepointfivepulselength-pulse_start_pulselength
      self.debug.info_message("test:- " + str(test) )

      adjusted_start = pulse_start_index
      test_metric_1 = pulse_start_threepulselength        - pulse_start_onepointfivepulselength
      test_metric_2 = pulse_start_threepulselength        - pulse_start_pulselength
      test_metric_3 = pulse_start_onepointfivepulselength - pulse_start_pulselength
      self.debug.info_message("test_metric_1:- " + str(test_metric_1) )
      self.debug.info_message("test_metric_2:- " + str(test_metric_2) )
      self.debug.info_message("test_metric_3:- " + str(test_metric_3) )
      self.debug.info_message("test_1:- " + str(test_1) )
      self.debug.info_message("test_3:- " + str(test_3) )
      self.debug.info_message("(pulse_start_threepulselength - pulse_start_index) metric: " + str(pulse_start_threepulselength - pulse_start_index) )

      #return pulse_start_index   # 1 of 3
      #return (test_3 + int(pulse_length/2)) % (3*pulse_length)   # 1 of 3
      #return (test_3 + int(pulse_length/2) + pulse_length) % (3*pulse_length)    # 2 of 3    0 of 3
      #return (test_3 + int(pulse_length/2) + (2*pulse_length)) % (3*pulse_length)   # 2 of 3   1 of 3

      # 2 of 3 and 3 of 3....
      if math.isclose(test_metric_1, 150, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        if math.isclose(test_metric_2 - test_metric_3, 150, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
          self.debug.info_message("pulse seq 1")
          return pulse_start_index # this test works well
          #return pulse_start_index + pulse_length
      if math.isclose(test_metric_1, 0, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        if math.isclose(test_metric_2 - test_metric_3, 0, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
          self.debug.info_message("pulse seq 2")
          if test_metric_2 < pulse_length:
            self.debug.info_message("suggestion 1: +0 or +200 ")
          elif test_metric_2 >= pulse_length:
            self.debug.info_message("suggestion 2: +100 ")

          return pulse_start_index + (2*pulse_length) # this test works well
      if math.isclose(test_metric_2, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        if math.isclose(test_metric_1 + test_metric_3, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
          self.debug.info_message("pulse seq 3")
          return pulse_start_index # ?????
      if math.isclose(test_metric_1, 100, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        if math.isclose(test_metric_2 - test_metric_3, 100, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
          self.debug.info_message("pulse seq 4")
          return pulse_start_index  + (2*pulse_length) # ????

      """
      adjusted_start = pulse_start_pulselength + pulse_length   #int(pulse_length/2)
      test_metric_1 = pulse_start_onepointfivepulselength - pulse_start_threepulselength
      if math.isclose(test_metric_1, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        adjusted_start = pulse_start_onepointfivepulselength + pulse_length
      elif math.isclose(test_metric_1, 0, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        #adjusted_start = pulse_start_onepointfivepulselength + (2*pulse_length)
        #adjusted_start = pulse_start_onepointfivepulselength + int(pulse_length/2)
        adjusted_start = pulse_start_onepointfivepulselength + pulse_start_pulselength + int(pulse_length/2) + pulse_length

      test_metric_2 = pulse_start_pulselength - pulse_start_onepointfivepulselength
      if math.isclose(test_metric_2, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        adjusted_start = pulse_start_pulselength + int(pulse_length/2) + pulse_length

      test_metric_3 = pulse_start_threepulselength - pulse_start_pulselength
      if math.isclose(test_metric_3, 0, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        adjusted_start = pulse_start_pulselength - int(pulse_length/2)
      elif math.isclose(test_metric_3, 150, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        adjusted_start = pulse_start_threepulselength + pulse_length
      elif math.isclose(test_metric_3, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        adjusted_start = pulse_start_threepulselength + (2*pulse_length)

      test_metric_4 = pulse_start_threepulselength - pulse_start_onepointfivepulselength
      if math.isclose(test_metric_4, 150, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
        #adjusted_start = pulse_start_onepointfivepulselength + pulse_length
        adjusted_start = pulse_start_pulselength + int(pulse_length/2)
      """

      adjusted_start = adjusted_start % (3*pulse_length)

      #if math.isclose(test, 0, rel_tol = rel_accuracy, abs_tol = abs_accuracy) or math.isclose(test, 150, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
      #  shift_required = 0
      #if math.isclose(test, 100, rel_tol = rel_accuracy, abs_tol = abs_accuracy) or math.isclose(test, 250, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
      #  shift_required = -100
      #if math.isclose(test, 200, rel_tol = rel_accuracy, abs_tol = abs_accuracy) or math.isclose(test, 50, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
      #  shift_required = -200
      #self.debug.info_message("shift_required:- " + str(shift_required) )
      

      if self.osmod.i3_pulse_align == ocn.I3_STANDINGWAVE_PULSE_1_OF_3:
        #adjusted_start = (int(pulse_start_threepulselength - (pulse_length/2)) + (pulse_length * 6)) % (pulse_length * 3)
        #adjusted_start = (pulse_start_pulselength + shift_required + int(pulse_length/2) + (pulse_length * 3)) % (pulse_length * 3)
        #adjusted_start = (pulse_start_threepulselength - int(pulse_length/2) + (pulse_length * 3)) % (pulse_length * 3)
        self.debug.info_message("adjusted_start:- " + str(adjusted_start) )
        return adjusted_start

    except:
      sys.stdout.write("Exception in detectStandingWavePulse: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")




  def detectIdentifyPulseTrain(self, fft_filtered, frequency, pulse_start_index):
    self.debug.info_message("detectIdentifyPulseTrain" )

    try:
      """ create an RRC pulse sample """
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      self.debug.info_message("pulse_length: " + str(pulse_length))

      modified_pulse_start_index = (pulse_start_index + (pulse_length/2)) % pulse_length
      self.debug.info_message("modified_pulse_start_index: " + str(modified_pulse_start_index))

      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate
      term5 = 2 * np.pi * time

      term8 = term5 * frequency[0]
      symbol_wave1 = self.osmod.modulation_object.amplitude * np.cos(term8) + self.osmod.modulation_object.amplitude * np.sin(term8)
      rrc_pulse_lower = symbol_wave1[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main

      term9 = term5 * frequency[1]
      symbol_wave2 = self.osmod.modulation_object.amplitude * np.cos(term9) + self.osmod.modulation_object.amplitude * np.sin(term9)
      rrc_pulse_higher = symbol_wave2[0:len(self.osmod.filtRRC_coef_main)] * self.osmod.filtRRC_coef_main

      #correlation_factor = 0.74
      correlation_factor = 0.7

      template = rrc_pulse_lower
      correlation = correlate(fft_filtered[0], template, mode='same')
      peak_threshold = correlation_factor * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]
      self.debug.info_message("peak_indices lower: " + str(peak_indices) )
      self.debug.info_message("len(peak_indices) lower: " + str(len(peak_indices)) )
      self.debug.info_message("correlation peak_indices lower: " + str(correlation[peak_indices]) )

      test = np.where(np.abs((peak_indices % pulse_length) - modified_pulse_start_index) < 7 ) [0]
      self.debug.info_message("len(test) lower: " + str(len(test)) )
      all_non_interpolated_lower = (peak_indices[test] // pulse_length) % self.osmod.pulses_per_block #64
      unique_non_interpolated_lower = np.unique((peak_indices[test] // pulse_length) % self.osmod.pulses_per_block) #64)
      self.debug.info_message("unique_non_interpolated_lower: " + str(unique_non_interpolated_lower) )
      bincount = np.bincount(all_non_interpolated_lower)
      self.debug.info_message("bincount lower: " + str(bincount) )
      persistent_lower = []
      for i in range(0, len(bincount)):
        if bincount[i] > 1:
          persistent_lower.append(i)
      self.debug.info_message("persistent_lower: " + str(persistent_lower) )


      template = rrc_pulse_higher
      correlation = correlate(fft_filtered[1], template, mode='same')
      peak_threshold = correlation_factor * np.max(correlation)
      peak_indices = np.where(correlation > peak_threshold)[0]
      self.debug.info_message("peak_indices higher: " + str(peak_indices) )
      self.debug.info_message("len(peak_indices) higher: " + str(len(peak_indices)) )
      self.debug.info_message("correlation peak_indices higher: " + str(correlation[peak_indices]) )

      test = np.where(np.abs((peak_indices % pulse_length) - modified_pulse_start_index) < 7 ) [0]
      self.debug.info_message("len(test) higher: " + str(len(test)) )
      all_non_interpolated_higher = (peak_indices[test] // pulse_length) % self.osmod.pulses_per_block  #64
      unique_non_interpolated_higher = np.unique((peak_indices[test] // pulse_length) % self.osmod.pulses_per_block)  #64)
      self.debug.info_message("unique_non_interpolated_higher: " + str(unique_non_interpolated_higher) )
      bincount = np.bincount(all_non_interpolated_higher)
      self.debug.info_message("bincount higher: " + str(bincount) )
      persistent_higher = []
      for i in range(0, len(bincount)):
        if bincount[i] > 1:
          persistent_higher.append(i)
      self.debug.info_message("persistent_higher: " + str(persistent_higher) )

      return persistent_lower, persistent_higher

    except:
      sys.stdout.write("Exception in detectIdentifyPulseTrain: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")




  def processAudioArray(self, audio_array, pulse_start_index, start_block, shift_amount, block_offsets):
    self.debug.info_message("processAudioArray")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      self.debug.info_message("block_offsets: " + str(block_offsets))

      """
      if shift_amount < 0:
        padding = np.zeros((abs(shift_amount) * pulse_length), dtype = audio_array.dtype)
        audio_array = np.append(padding, audio_array)
        pulse_start_index = (pulse_start_index + (abs(shift_amount) * pulse_length) )
      elif shift_amount > 0:
        audio_array = audio_array[(abs(shift_amount) * pulse_length):]
        pulse_start_index = (pulse_start_index - (abs(shift_amount) * pulse_length) + (3 * pulse_length * self.osmod.pulses_per_block)) % (3*pulse_length)
      """

      start = pulse_start_index + (start_block * self.osmod.pulses_per_block * pulse_length)
      num_full_blocks = int((len(audio_array) - start ) // self.osmod.symbol_block_size)
      self.debug.info_message("num_full_blocks: " + str(num_full_blocks))
      new_audio_array = np.zeros(num_full_blocks * self.osmod.pulses_per_block * pulse_length, dtype = audio_array.dtype)
      audio_array_length = len(audio_array)
      for block_count in range(0, num_full_blocks): 
        for index in range(0,self.osmod.pulses_per_block): 
          offset_source = ((block_count * self.osmod.pulses_per_block) * pulse_length) + start + (index * pulse_length) + block_offsets[block_count]

          if offset_source < 0:
            adjustment = abs(offset_source)
          else:
            adjustment = 0

          offset_dest   = ((block_count * self.osmod.pulses_per_block) * pulse_length) + (index * pulse_length)
          if offset_source + pulse_length < audio_array_length:
            new_audio_array[offset_dest + adjustment:offset_dest + pulse_length] = audio_array[offset_source + adjustment:offset_source + pulse_length]

      return 0, new_audio_array

    except:
      self.debug.error_message("Exception in processAudioArray: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      self.debug.info_message("audio_array_length: " + str(audio_array_length))
      self.debug.info_message("index: " + str(index))
      self.debug.info_message("offset_source: " + str(offset_source))
      self.debug.info_message("offset_dest: " + str(offset_dest))
      self.debug.info_message("offset_source + pulse_length: " + str(offset_source + pulse_length))


  def processAudioArrayPulses(self, audio_array, pulse_start_index, start_block, shift_amount, pulse_offsets):
    self.debug.info_message("processAudioArrayPulses")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      self.debug.info_message("pulse_offsets: " + str(pulse_offsets))

      """
      if shift_amount < 0:
        padding = np.zeros((abs(shift_amount) * pulse_length), dtype = audio_array.dtype)
        audio_array = np.append(padding, audio_array)
        pulse_start_index = (pulse_start_index + (abs(shift_amount) * pulse_length) )
      elif shift_amount > 0:
        audio_array = audio_array[(abs(shift_amount) * pulse_length):]
        pulse_start_index = (pulse_start_index - (abs(shift_amount) * pulse_length) + (3 * pulse_length * self.osmod.pulses_per_block)) % (3*pulse_length)
      """

      start = pulse_start_index + (start_block * self.osmod.pulses_per_block * pulse_length)

      #num_pulses = int((len(audio_array) - start ) // self.osmod.symbol_block_size) * self.osmod.pulses_per_block
      num_pulses = len(pulse_offsets)
      self.debug.info_message("num_pulses: " + str(num_pulses))
      new_audio_array = np.zeros(num_pulses * pulse_length, dtype = audio_array.dtype)
      audio_array_length = len(audio_array)
      for index in range(0, num_pulses): 
        offset_source = (index * pulse_length) + start + int(pulse_offsets[index])

        if offset_source < 0:
          adjustment = abs(offset_source)
        else:
          adjustment = 0

        offset_dest = index * pulse_length
        if offset_source + pulse_length < audio_array_length and offset_source + pulse_length > 0:
          #new_audio_array[offset_dest:offset_dest + pulse_length] = audio_array[offset_source:offset_source + pulse_length]
          new_audio_array[offset_dest + adjustment:offset_dest + pulse_length] = audio_array[offset_source + adjustment:offset_source + pulse_length]

      return 0, new_audio_array

    except:
      self.debug.error_message("Exception in processAudioArrayPulses: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      self.debug.info_message("audio_array_length: " + str(audio_array_length))
      self.debug.info_message("index: " + str(index))
      self.debug.info_message("offset_source: " + str(offset_source))
      self.debug.info_message("offset_dest: " + str(offset_dest))
      self.debug.info_message("offset_source + pulse_length: " + str(offset_source + pulse_length))



  def processShiftAmount(self, audio_array, pulse_start_index, start_block, shift_amount):
    self.debug.info_message("processShiftAmount")
    try:
      pulse_length = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      if shift_amount > 0:
        padding = np.zeros((abs(shift_amount) * pulse_length) - pulse_start_index, dtype = audio_array.dtype)
        #pulse_start_index = (pulse_start_index + (abs(shift_amount) * pulse_length) )
        #return pulse_start_index, np.append(padding, audio_array)
        return 0, np.append(padding, audio_array)
      elif shift_amount < 0:
        #pulse_start_index = (pulse_start_index - (abs(shift_amount) * pulse_length) + (3 * pulse_length * self.osmod.pulses_per_block)) % (3*pulse_length)
        #return pulse_start_index, audio_array[(abs(shift_amount) * pulse_length):]
        return 0, audio_array[(abs(shift_amount) * pulse_length) + pulse_start_index:]
      else:
        #return pulse_start_index, audio_array
        return 0, audio_array[pulse_start_index:]

    except:
      self.debug.error_message("Exception in processShiftAmount: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calcResidualPhaseAngles(self, audio_array, frequency, pulse_offsets, interpolated_lower, interpolated_higher):
    self.debug.info_message("calcResiduals")
    try:
      samples_per_wavelength_lower  = self.osmod.sample_rate / frequency[0]
      samples_per_wavelength_higher = self.osmod.sample_rate / frequency[1]

      num_full_blocks = int(len(audio_array) // self.osmod.symbol_block_size)

      block_start_residual_lower  = [0] * num_full_blocks
      block_start_residual_higher = [0] * num_full_blocks

      block_end_residual_lower    = [0] * num_full_blocks
      block_end_residual_higher   = [0] * num_full_blocks

      block_mid_residual_lower    = [0] * num_full_blocks
      block_mid_residual_higher   = [0] * num_full_blocks

      total_block_residual_lower  = [0] * num_full_blocks
      total_block_residual_higher = [0] * num_full_blocks

      absolute_adjust_lower       = [0] * num_full_blocks
      absolute_adjust_higher      = [0] * num_full_blocks

      total_block_drift_lower     = [0] * num_full_blocks
      total_block_drift_higher    = [0] * num_full_blocks

      total_block_drift_lower2    = [0] * num_full_blocks
      total_block_drift_higher2   = [0] * num_full_blocks

      first_block_mid_residual_lower  = 0.0
      first_block_mid_residual_higher = 0.0

      pulse_train_length_triplet = 3 * (len(interpolated_lower) // 3)

      compressioncalc_override_checked = self.osmod.form_gui.window['cb_overridedopplercompressioncalc'].get()
      compressioncalc_absolute         = self.osmod.form_gui.window['cb_overridedopplercompressionabsolute'].get()
      compressioncalc_factor           = float(self.osmod.form_gui.window['in_dopplercompressionfactor'].get())


      #calibration_drift_per_pulse_lower  = ((np.array(pulse_offsets) / samples_per_wavelength_lower)  * 2 * np.pi) % (np.pi/4)
      calibration_drift_per_pulse_lower  = ((np.array(pulse_offsets) / samples_per_wavelength_lower)  * 2 * np.pi)
      #calibration_drift_per_pulse_lower  = (((np.array(pulse_offsets) / samples_per_wavelength_lower)  * 2 * np.pi) / (pulse_train_length_triplet /3)) % (np.pi/4)
      self.dict_saved_data['calibration_drift_per_pulse_lower']  = calibration_drift_per_pulse_lower
      #calibration_drift_per_pulse_higher = ((np.array(pulse_offsets) / samples_per_wavelength_higher) * 2 * np.pi) % (np.pi/4)
      calibration_drift_per_pulse_higher = ((np.array(pulse_offsets) / samples_per_wavelength_higher) * 2 * np.pi)
      #calibration_drift_per_pulse_higher = (((np.array(pulse_offsets) / samples_per_wavelength_higher) * 2 * np.pi) / (pulse_train_length_triplet /3)) % (np.pi/4)
      self.dict_saved_data['calibration_drift_per_pulse_higher'] = calibration_drift_per_pulse_higher


      for block_count in range(0, num_full_blocks): 
        offset = block_count * self.osmod.pulses_per_block

        """ calculate value at start point """
        start_lower  = (pulse_offsets[offset] / samples_per_wavelength_lower)  * 2 * np.pi
        block_start_residual_lower[block_count] = start_lower

        start_higher = (pulse_offsets[offset] / samples_per_wavelength_higher) * 2 * np.pi
        block_start_residual_higher[block_count]  = start_higher

        """ calculate value at end point """
        end_lower    = ((pulse_offsets[pulse_train_length_triplet -1 + offset] ) / samples_per_wavelength_lower)  * 2 * np.pi
        block_end_residual_lower[block_count]  = end_lower

        end_higher   = ((pulse_offsets[pulse_train_length_triplet -1 + offset] ) / samples_per_wavelength_higher)  * 2 * np.pi
        block_end_residual_higher[block_count]  = end_higher

        #2 is 3rd section s instead of k on -ve gradient
        #6 is overcompensating 3rd section i insted of k on -ve gradient
        # 4 has come perfect decodes!!!!!!!!! when pulse train length triplet is 12 !!!!!  aaaaaaaa44y4y44444kjkkjkjjw0   should be aaaaaaaaaayyyyyyyyyykkkkkkkk
        # with auto rotation angles of 7 and 0
        #adjustment_phase_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count]) * 4
        #adjustment_phase_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) * 4

        # 1 also works with a straight gradient
        #adjustment_phase_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count]) * 1
        #adjustment_phase_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) * 1


        #phase_adjust_type = PHASE_ADJUST_ABSOLUTE
        #phase_adjust_type = ocn.PHASE_ADJUST_RELATIVE

        #compressioncalc_override_checked = self.osmod.form_gui.window['cb_overridedopplercompressioncalc'].get()
        #compressioncalc_absolute = self.osmod.form_gui.window['cb_overridedopplercompressionabsolute'].get()
        #compressioncalc_factor   = float(self.osmod.form_gui.window['in_dopplercompressionfactor'].get())

        #for pulse_count in range(0, len(interpolated_lower)):
        #  total_block_drift_lower[block_count]  = total_block_drift_lower[block_count]  + calibration_drift_per_pulse_lower[offset + pulse_count]
          #total_block_drift_lower[block_count]  = total_block_drift_lower[block_count]  + (calibration_drift_per_pulse_lower[offset + pulse_count]  / (pulse_train_length_triplet /3)
          #adjustment_phase_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet /3)
        #for pulse_count in range(0, len(interpolated_higher)):
        #  total_block_drift_higher[block_count] = total_block_drift_higher[block_count] + calibration_drift_per_pulse_higher[offset + pulse_count]
        total_block_drift_lower[block_count]  = calibration_drift_per_pulse_lower[offset + int(len(interpolated_lower)/2)]
        #total_block_drift_lower[block_count]  = total_block_drift_lower[block_count] % (np.pi/4)
        total_block_drift_lower[block_count]  = total_block_drift_lower[block_count] % (np.pi * 2)
        total_block_drift_lower2[block_count] = calibration_drift_per_pulse_lower[offset + int(len(interpolated_lower)/2)] % (np.pi * 2 * 8)

        total_block_drift_higher[block_count] = calibration_drift_per_pulse_higher[offset + int(len(interpolated_lower)/2)]
        #total_block_drift_higher[block_count] = total_block_drift_higher[block_count] % (np.pi/4)
        total_block_drift_higher[block_count] =  total_block_drift_higher[block_count] % (np.pi * 2)
        total_block_drift_higher2[block_count] = calibration_drift_per_pulse_higher[offset + int(len(interpolated_lower)/2)] % (np.pi * 2 * 8)


        if compressioncalc_override_checked:
          if compressioncalc_absolute:
            """ calculate compression / expansion """
            phase_compression_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet /3)
            phase_compression_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) / (pulse_train_length_triplet /3)
            #phase_compression_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet / compressioncalc_factor)
            #phase_compression_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) / (pulse_train_length_triplet / compressioncalc_factor)

            """ calculate drift of mid point relative to first block """
            absolute_adjust_lower[block_count]  = (block_end_residual_lower[block_count]  + block_start_residual_lower[block_count])  / 2
            absolute_adjust_higher[block_count] = (block_end_residual_higher[block_count] + block_start_residual_higher[block_count]) / 2
            diff_lower  = absolute_adjust_lower[block_count]  - absolute_adjust_lower[0]
            diff_higher = absolute_adjust_higher[block_count] - absolute_adjust_higher[0]

            """ add relative drift and compression """
            #adjustment_phase_lower  = phase_compression_lower  + (diff_lower / compressioncalc_factor)
            #adjustment_phase_higher = phase_compression_higher + (diff_higher / compressioncalc_factor)
            adjustment_phase_lower  = phase_compression_lower  + (diff_lower / pulse_train_length_triplet)
            adjustment_phase_higher = phase_compression_higher + (diff_higher / pulse_train_length_triplet)
          else:
            adjustment_phase_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet / compressioncalc_factor)
            adjustment_phase_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) / (pulse_train_length_triplet / compressioncalc_factor)
        else:
        #if phase_adjust_type == ocn.PHASE_ADJUST_RELATIVE:
          #YES!!!!THIS WORKS FOR CURVED DOPPLER SHIFT....aaibaaaiyyyy445yy41kkkkkkku     length 12
          adjustment_phase_lower  = (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet /3)
          adjustment_phase_higher = (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) / (pulse_train_length_triplet /3)
        #elif phase_adjust_type == ocn.PHASE_ADJUST_ABSOLUTE:
        #  adjustment_phase_lower  = block_start_residual_lower[block_count] + (block_end_residual_lower[block_count]  - block_start_residual_lower[block_count])  / (pulse_train_length_triplet /3)
        #  adjustment_phase_higher = block_start_residual_higher[block_count] + (block_end_residual_higher[block_count] - block_start_residual_higher[block_count]) / (pulse_train_length_triplet /3)


        self.debug.info_message("adjustment_phase_lower: " + str(adjustment_phase_lower))
        self.debug.info_message("adjustment_phase_higher: " + str(adjustment_phase_higher))

        total_block_residual_lower[block_count]  = self.osmod.modulation_object.normalizeAngle(adjustment_phase_lower)
        total_block_residual_higher[block_count] = self.osmod.modulation_object.normalizeAngle(adjustment_phase_higher)


      self.dict_saved_data['total_block_drift_lower']  = total_block_drift_lower
      self.dict_saved_data['total_block_drift_higher'] = total_block_drift_higher
      self.dict_saved_data['total_block_drift_lower2']  = total_block_drift_lower2
      self.dict_saved_data['total_block_drift_higher2'] = total_block_drift_higher2


      self.debug.info_message("pulse_train_length_triplet: " + str(pulse_train_length_triplet))
      self.debug.info_message("samples_per_wavelength_lower: " + str(samples_per_wavelength_lower))
      self.debug.info_message("samples_per_wavelength_higher: " + str(samples_per_wavelength_higher))

      #if compressioncalc_override_checked and compressioncalc_absolute:
      #  return [total_block_residual_lower + absolute_adjust_lower, total_block_residual_higher + absolute_adjust_higher]
      #else:
      return [total_block_residual_lower, total_block_residual_higher]

    except:
      self.debug.error_message("Exception in calcResiduals: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def calcPulseTrainSectionAngles(self, frequency, interpolated_lower, interpolated_higher):
    self.debug.info_message("calcPulseTrainSectionAngles")
    try:
      samples_per_wavelength_lower  = self.osmod.sample_rate / frequency[0]
      samples_per_wavelength_higher = self.osmod.sample_rate / frequency[1]
      pulse_length           = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      pulse_train_length = int(min(len(interpolated_lower), len(interpolated_higher)) // 3) * 3
      #self.pulse_train_length = pulse_train_length
      self.pulse_train_length = int(min(len(interpolated_lower), len(interpolated_higher)))
      self.debug.info_message("pulse_train_length: " + str(pulse_train_length))

      pulse_train_angles_lower  = [0] * self.osmod.pulses_per_block
      pulse_train_angles_higher = [0] * self.osmod.pulses_per_block

      pattern_offset_lower  = [0, self.osmod.i3_offsets[0], self.osmod.i3_offsets[1]]
      pattern_offset_higher = [0, self.osmod.i3_offsets[2], self.osmod.i3_offsets[3]]

      self.debug.info_message("calcPulseTrainSectionAngles LOC1")

      """ determine phase at each pulse midpoint along an unshifted pulse train starting at origin."""
      #for pulse_train_item in range (0, pulse_train_length):
      for pulse_train_item in range (0, int(self.osmod.pulses_per_block / 2)):
        self.debug.info_message("pulse_train_item: " + str(pulse_train_item))
        #pulse_train_angles_lower[pulse_train_item]  = self.osmod.modulation_object.normalizeAngle(pulse_length / samples_per_wavelength_lower  + pattern_offset_lower[pulse_train_item % 3])
        pulse_train_angles_lower[pulse_train_item]  = self.osmod.modulation_object.normalizeAngle( ( ((pulse_train_item * pulse_length) + (pulse_length / 2)) / samples_per_wavelength_lower) + pattern_offset_lower[pulse_train_item % 3])
        #pulse_train_angles_higher[pulse_train_item] = self.osmod.modulation_object.normalizeAngle(pulse_length / samples_per_wavelength_higher + pattern_offset_higher[pulse_train_item % 3])
        pulse_train_angles_higher[pulse_train_item] = self.osmod.modulation_object.normalizeAngle( ( ((pulse_train_item * pulse_length) + (pulse_length / 2)) / samples_per_wavelength_higher) + pattern_offset_higher[pulse_train_item % 3])
        pulse_train_item = pulse_train_item + 1

      number_of_possible_series = int(self.osmod.pulses_per_block / 2) - pulse_train_length + 1

      series_angles_lower  = [0] * number_of_possible_series
      series_angles_higher = [0] * number_of_possible_series

      self.debug.info_message("calcPulseTrainSectionAngles LOC2")

      for series in range(0, number_of_possible_series):
        #for pulse_count in range(0, pulse_train_length):
        #  self.debug.info_message("pulse_count + series: " + str(pulse_count + series))
        #  series_angles_lower[series]  = series_angles_lower[series] + pulse_train_angles_lower[pulse_count + series]
        #  series_angles_higher[series] = series_angles_higher[series] + pulse_train_angles_higher[pulse_count + series]
        phase_lower = (((pulse_length * series) + (pulse_length * pulse_train_length)) / samples_per_wavelength_lower) - ((pulse_length * series) / samples_per_wavelength_lower) 
        series_angles_lower[series]  = self.osmod.modulation_object.normalizeAngle(((pulse_length * series) / samples_per_wavelength_lower) + (phase_lower/2) + pattern_offset_lower[series % 3])
        phase_higher = (((pulse_length * series) + (pulse_length * pulse_train_length)) / samples_per_wavelength_higher) - ((pulse_length * series) / samples_per_wavelength_higher) 
        series_angles_higher[series] = self.osmod.modulation_object.normalizeAngle(((pulse_length * series) / samples_per_wavelength_higher) + (phase_higher/2) + pattern_offset_higher[series % 3])

        #series_angles_lower[series]  = self.osmod.modulation_object.normalizeAngle(series_angles_lower[series])
        #series_angles_higher[series] = self.osmod.modulation_object.normalizeAngle(series_angles_higher[series])

      self.debug.info_message("series_angles_lower: " + str(series_angles_lower))
      self.debug.info_message("series_angles_higher: " + str(series_angles_higher))

      self.series_angles[0] = series_angles_lower
      self.series_angles[1] = series_angles_higher

    except:
      self.debug.error_message("Exception in calcPulseTrainSectionAngles: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def writePulseTrainRotationDetails(self):
    self.debug.info_message("writePulseTrainRotationDetails")
    try:
      self.debug.info_message("writing file")
      self.osmod.analysis.writeDataToFile2([[self.pulse_train_length, self.rotation_angles[0], self.rotation_angles[1], self.series_angles[0], self.series_angles[1]]], "rotation_file.csv", 'w')
    except:
      self.debug.error_message("Exception in writePulseTrainRotationDetails: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def findDisposition(self,  interpolated_lower, interpolated_higher):
    self.debug.info_message("findDisposition")
    try:
      match_type = ocn.DISPOSITION_NO_MATCH
      best_ambiguous_match = 0

      rotation_lo = self.rotation_angles[0]
      rotation_hi = self.rotation_angles[1]
 
      rotation_dict = self.osmod.rotation_tables
      self.disposition = -1
      if rotation_dict != None:
        self.debug.info_message("located tables for mode")
        self.debug.info_message("rotation_dict: " + str(rotation_dict))
        #pulse_train_length = min(int((len(interpolated_lower) // 3 ) * 3), int((len(interpolated_higher) // 3 ) * 3))
        pulse_train_length = min(len(interpolated_lower), len(interpolated_higher))
        self.debug.info_message("pulse_train_length: " + str(pulse_train_length))
        active_table = rotation_dict[str(pulse_train_length)]
        self.debug.info_message("active_table: " + str(active_table))
        self.debug.info_message("rotation_lo: " + str(rotation_lo))
        self.debug.info_message("rotation_hi: " + str(rotation_hi))

        adjust_lo = 0
        adjust_hi = 0
        if rotation_lo > 4:
          adjust_lo = - 2
        elif rotation_lo < 2:
          adjust_lo = 2
        if rotation_hi > 4:
          adjust_hi = - 2
        elif rotation_hi < 2:
          adjust_hi = 2

        rotation_lo  = self.osmod.modulation_object.normalizeAngle(rotation_lo + adjust_lo)
        rotation_hi  = self.osmod.modulation_object.normalizeAngle(rotation_hi + adjust_hi)
        self.debug.info_message("adjusted rotation_lo: " + str(rotation_lo))
        self.debug.info_message("adjusted rotation_hi: " + str(rotation_hi))

        """ -1 is not found, -2 is more than 1 found, >=0 is valid value for disposition."""
        rel_accuracy = 0.0

        #abs_accuracy = 1e-1
        #accuracy_increment = 1e-1

        abs_accuracy = self.osmod.disposition_increment
        accuracy_increment = self.osmod.disposition_increment

        found_match = False
        for _ in range (0,15):
          if found_match == True:
            break

          abs_accuracy = abs_accuracy + accuracy_increment

          for test_disposition in range(0, len(active_table)):
            self.debug.info_message("active_table[test_disposition]: " + str(active_table[test_disposition]))
            self.debug.info_message("test_disp_lo: " + str(active_table[test_disposition][0]))
            self.debug.info_message("test_disp_hi: " + str(active_table[test_disposition][1]))
            test_disp_lo  = self.osmod.modulation_object.normalizeAngle(active_table[test_disposition][0] + adjust_lo)
            test_disp_hi  = self.osmod.modulation_object.normalizeAngle(active_table[test_disposition][1] + adjust_hi)
            self.debug.info_message("adjusted test_disp_lo: " + str(test_disp_lo))
            self.debug.info_message("adjusted test_disp_hi: " + str(test_disp_hi))

            if math.isclose(rotation_lo, test_disp_lo, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
              if math.isclose(rotation_hi, test_disp_hi, rel_tol = rel_accuracy, abs_tol = abs_accuracy):
                if found_match == False:
                  self.debug.info_message("found match")
                  self.debug.info_message("test_disposition: " + str(test_disposition))
                  self.debug.info_message("abs_accuracy: " + str(abs_accuracy))
                  self.disposition = test_disposition
                  found_match = True
                  match_type = ocn.DISPOSITION_MATCH_SINGLE
                  best_ambiguous_match = test_disposition
                else:
                  self.debug.info_message("ERROR duplicate match found")
                  self.disposition = -2
                  match_type = ocn.DISPOSITION_MATCH_AMBIGUOUS

      return self.disposition
      #return self.disposition, match_type, best_ambiguous_match


    except:
      self.debug.error_message("Exception in findDisposition: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def createInterpolated(self):
    self.debug.info_message("createInterpolated")
    try:
      half = int(self.osmod.pulses_per_block / 2)
      interpolated_lower = []
      interpolated_higher = []
      for interp in range(0, half):
        interpolated_lower.append(interp)
        interpolated_higher.append(interp + half)

      self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      return [interpolated_lower, interpolated_higher]

    except:
      self.debug.error_message("Exception in createInterpolated: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


