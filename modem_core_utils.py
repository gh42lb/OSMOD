#!/usr/bin/env python

import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
import scipy as sp
import gc
import FreeSimpleGUI as sg
import random
import ctypes
import platform

from numpy import pi
from scipy.signal import butter, filtfilt, firwin, sosfiltfilt, hilbert
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io.wavfile import write, read
from datetime import datetime, timedelta
from scipy.fft import fft
from numpy.fft import ifft
from scipy.signal import periodogram, zoom_fft, lfilter
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from numpy.polynomial import Chebyshev as T
from scipy import stats
from scipy.interpolate import CubicSpline, splrep, splev, PchipInterpolator, UnivariateSpline
from osmod_c_interface import ptoc_float_array, ptoc_double_array, ptoc_float, ctop_int, ptoc_int_array, ptoc_numpy_int_array, ptoc_double, ptoc_double_pointer_array

from scipy import signal as scipy_signal
from datetime import datetime, timedelta

from crc import Calculator, Configuration

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



class ModemCoreUtils(object):

  osmod = None
  dict_binpair_to_quad = {'00':0, '01':1, '10':2, '11':3}
  """ optimized for 64 bit encodings """

  """ character p 010,000 is best padding character"""

  """ temporary base 64 char format """
  #encoding_b64    = 'abcdefghijklmnopqrstuvwxyz 0123456789~!@#$%^&*()_+`-={}|[]\\:\";\'<'
  #encoding_b64    = ' abcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+`-={}|[]\\:\";\'<'
  #encoding_b64    = ' abcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+`-={}.[]\\:\";\'<'
  encoding_b64     = ' abcdefghijklmno|pqrstuvwxyz0123456789[](){}+-:;=?@^,\'./~_%!#$&*'


  """ optimized for regular character set """
  encoding_normal  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+`-={}|[]\\:\";\'<>?,./\n '

  amplitude = 2.0

  chart_data_dict = {}



  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD)
    self.debug.info_message("__init__")
    self.osmod = osmod

    """ initialise data structures """
    self.b64_charfromindex_list = []
    self.b64_indexfromchar_dict = {}
    self.normal_charfromindex_list = []
    self.normal_indexfromchar_dict = {}

    """ set some defaults """
    self.sample_rate = 44100 # Hz
    self.symbol_rate = 100 # symbols / second
    self.amplitude = 0.5

    for char in self.encoding_b64:
      self.b64_charfromindex_list.append(char)
      self.debug.verbose_message("b64_charfromindex_list appending: " + str(char))
    for index in range(len(self.b64_charfromindex_list)):
      char = self.b64_charfromindex_list[index]
      self.b64_indexfromchar_dict[char] = index
      self.debug.verbose_message("b64_indexfromchar_dict: [" + str(char) + ']=' + str(index))

    for char in self.encoding_normal:
      self.normal_charfromindex_list.append(char)
      self.debug.verbose_message("normal_charfromindex_list appending: " + str(char))
    for index in range(len(self.normal_charfromindex_list)):
      char = self.normal_charfromindex_list[index]
      self.normal_indexfromchar_dict[char] = index
      self.debug.verbose_message("normal_indexfromchar_dict: [" + str(char) + ']=' + str(index))


  """ file based methods"""

  def readFileWav(self, filename):
    try:
      self.debug.info_message("reading data")
      samp_rate, audio_data = read(filename)
    except:
      self.debug.error_message("Exception in modDemod: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    signal = np.frombuffer(audio_data, dtype=np.float32)

    return signal.astype(np.float64)
    """ long double breaks on some platforms as precision varies from one platform to the next"""
    #return signal.astype(np.longdouble)
    #return np.frombuffer(audio_data, dtype=np.float32)
    #return np.frombuffer(audio_data, dtype=np.float64)


  def writeFileWavSR(self, filename, multi_block, sample_rate):
    self.debug.info_message("writeFileWavSR")
    return self.writeFileWavCommon(filename, multi_block, sample_rate)

  def writeFileWav(self, filename, multi_block):
    self.debug.info_message("writeFileWav")
    return self.writeFileWavCommon(filename, multi_block, self.osmod.sample_rate)

  def writeFileWavCommon(self, filename, multi_block, sample_rate):
    self.debug.info_message("writeFileWavCommon")
    self.debug.info_message("multi_block data type: " + str(multi_block.dtype))
    try:
      self.debug.info_message("test1")
      test1 = np.max(np.abs(multi_block))
      self.debug.info_message("test2")
      test2 = multi_block * (2**15 - 1)

      #multi_block = multi_block * (2**15 - 1) / np.max(np.abs(multi_block))
      #multi_block = multi_block * (2**6 - 1) / np.max(np.abs(multi_block))
      multi_block = multi_block * (2**3 - 1) / np.max(np.abs(multi_block))

      self.debug.info_message("writing audio file")
      multi_block = multi_block.astype(np.float32)
      #multi_block = multi_block.astype(np.float64)
      #write(filename, self.osmod.sample_rate, multi_block)
      write(filename, sample_rate, multi_block)
    except:
      self.debug.error_message("Exception in writeFileWav: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def writeFileWav2(self, filename, multi_block):
    self.debug.info_message("writeFileWav2")
    self.debug.info_message("multi_block data type: " + str(multi_block.dtype))
    try:
      #self.debug.info_message("test1")
      #test1 = np.max(np.abs(multi_block))
      #self.debug.info_message("test2")
      #test2 = multi_block * (2**15 - 1)

      #multi_block = multi_block * (2**15 - 1) / np.max(np.abs(multi_block))

      self.debug.info_message("writing audio file")
      multi_block = multi_block.astype(np.float32)
      #multi_block = multi_block.astype(np.float64)
      write(filename, 48000, multi_block)
    except:
      self.debug.error_message("Exception in writeFileWav: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ average an array of angles """
  def averageAngles(self, angle_array):
    #self.debug.info_message("averageAngles")
    try:
      """ convert to cartesian """
      sin_sum = np.sum(np.sin(angle_array))
      cos_sum = np.sum(np.cos(angle_array))

      return np.arctan2(sin_sum, cos_sum)

    except:
      self.debug.error_message("Exception in averageAngles: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def averageAngles2(self, angle_array, std_threshold = 1.5):
    self.debug.info_message("averageAngles2")
    return self.averageAngles3(angle_array, 0.1, std_threshold)

  """ average an array of angles after removing outliers in the data"""
  def averageAngles3(self, angle_array, num_items_ratio, std_threshold = 1.5):
    #self.debug.info_message("averageAngles3")
    try:
      std_threshold_inc = std_threshold
      num_items = len(angle_array) * num_items_ratio
      mean_angle = stats.circmean(angle_array)
      #self.debug.info_message("mean_angle: " + str(mean_angle))

      if len(angle_array) <= num_items:
        return mean_angle

      std_dev = stats.circstd(angle_array)
      #self.debug.info_message("std_dev: " + str(std_dev))
      filtered_angles = [angle for angle in angle_array if abs(angle - mean_angle) <= std_threshold * std_dev]
      while len(filtered_angles) < num_items:
        std_threshold = std_threshold + std_threshold_inc
        std_dev = stats.circstd(angle_array)
        filtered_angles = [angle for angle in angle_array if abs(angle - mean_angle) <= std_threshold * std_dev]

      #self.debug.info_message("std_threshold: " + str(std_threshold))
      mean_filtered_angle = stats.circmean(filtered_angles)
      #self.debug.info_message("mean_filtered_angle: " + str(mean_filtered_angle))
      return mean_filtered_angle
    except:
      self.debug.error_message("Exception in averageAngles2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  """ filter data and get max. can be used with wave data"""
  def filteredSum(self, array, num_items_ratio, std_threshold = 1.5):
    #self.debug.info_message("filteredSum")
    try:
      std_threshold_inc = std_threshold
      num_items = len(array) * num_items_ratio
      #self.debug.info_message("num_items: " + str(num_items))
      #self.debug.info_message("len(array): " + str(len(array)))
      if len(array) <= num_items:
        return np.sum(array) / len(array)

      mean = np.mean(array)
      #self.debug.info_message("mean: " + str(mean))

      std_dev = np.std(array)
      #self.debug.info_message("std_dev: " + str(std_dev))
      filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]
      while len(filtered_items) < num_items:
        std_threshold = std_threshold + std_threshold_inc
        #self.debug.info_message("filtered_items: " + str(filtered_items))
        #self.debug.info_message("len(filtered_items): " + str(len(filtered_items)))
        std_dev = np.std(filtered_items)
        mean = (np.mean(filtered_items) + mean) / 2
        filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]

      #self.debug.info_message("std_threshold: " + str(std_threshold))
      sum_filtered = np.sum(filtered_items) / len(filtered_items)
      #self.debug.info_message("sum_filtered: " + str(sum_filtered))
      return sum_filtered
    except:
      self.debug.error_message("Exception in filteredSum: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ filter data and get max. can be used with wave data"""
  def filteredMax(self, array, num_items_ratio, std_threshold = 1.5):
    self.debug.info_message("filteredMax")
    try:
      num_items = len(array) * num_items_ratio

      if len(array) <= num_items:
        return np.max(array)

      mean = np.mean(array)
      self.debug.info_message("mean: " + str(mean))

      std_dev = np.std(array)
      self.debug.info_message("std_dev: " + str(std_dev))
      filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]
      while len(filtered_items) < num_items:
        std_threshold = std_threshold + 2
        std_dev = np.std(filtered_items)
        filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]

      self.debug.info_message("std_threshold: " + str(std_threshold))
      max_filtered = np.max(filtered_items)
      self.debug.info_message("max_filtered: " + str(max_filtered))
      return max_filtered
    except:
      self.debug.error_message("Exception in filteredMax: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ filter data and get min. can be used with wave data"""
  def filteredMin(self, array, num_items_ratio, std_threshold = 1.5):
    self.debug.info_message("filteredMin")
    try:
      num_items = len(array) * num_items_ratio
      if len(array) <= num_items:
        return np.min(array)

      mean = np.mean(array)
      self.debug.info_message("mean: " + str(mean))

      std_dev = np.std(array)
      self.debug.info_message("std_dev: " + str(std_dev))
      filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]
      while len(filtered_items) < num_items:
        std_threshold = std_threshold + 2
        std_dev = np.std(filtered_items)
        filtered_items = [item for item in array if abs(item - mean) <= std_threshold * std_dev]

      self.debug.info_message("std_threshold: " + str(std_threshold))
      min_filtered = np.min(filtered_items)
      self.debug.info_message("min_filtered: " + str(min_filtered))
      return min_filtered
    except:
      self.debug.error_message("Exception in filteredMin: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def getSmallestAngle(self, angle):
    #self.debug.info_message("getSmallestAngle")
    try:
      smallest_angle = min(abs(angle % (2*np.pi)), (2*np.pi) - abs(angle % (2*np.pi))) 
      return smallest_angle
    except:
      self.debug.error_message("Exception in getSmallestAngle: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  """ normalize the angle to zero based in radians i.e. 0 to 2*pi rad"""
  def normalizeAngle(self, angle):
    #self.debug.info_message("normalizeAngle")
    try:
      normalized_angle = (angle + (2*np.pi)) % (2*np.pi)
      return normalized_angle
    except:
      self.debug.error_message("Exception in normalizeAngle: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def normalizeAngleArray(self, angle_array):
    self.debug.info_message("normalizeAngleArray")
    try:
      normalized_angle_array = (angle_array + (2*np.pi)) % (2*np.pi)
      return normalized_angle_array
    except:
      self.debug.error_message("Exception in normalizeAngleArray: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  """ noise methods"""

  def addTimingNoise(self, signal):
    timing_noise_std = 0.1
    timing_noise = np.random.normal(0, timing_noise_std, len(signal))
    return (signal + timing_noise)


  def addPhaseNoise2(self, signal):
    fft_x = np.fft.fft(signal)
    phase_noise = np.random.normal(0, 0.1, len(fft_x))
    noisy_fft_x = fft_x * np.exp(1j * phase_noise)
    return np.fft.ifft(noisy_fft_x).real

  def addWhiteNoise(self, signal, noise_factor):
    noise = np.random.normal(0, signal.std(), size=signal.shape)
    return (signal + (noise_factor * noise))

  def addAWGN(self, signal, noise_factor, signal_frequency):
    self.debug.info_message("addAWGN")

    try:
      noise = np.random.normal(0, signal.std(), len(signal))
      return (signal + (noise_factor * noise))

    except:
      self.debug.error_message("Exception in addAWGN: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ Define filters """
  def apply_filterSR(self, signal, params, center_frequency, sample_rate):
    return self.apply_filter_common(signal, params, center_frequency, sample_rate)

  def apply_filter(self, signal, params, center_frequency):
    return self.apply_filter_common(signal, params, center_frequency, self.osmod.sample_rate)

  # 'tx_filter' : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
  def apply_filter_common(self, signal, params, center_frequency, sample_rate):
    try:
      self.debug.info_message("apply_filter_common")
      filter_type = params[0]
      filter_pass_type = params[1]
      filter_width = params[2]
      repeats = params[3]
      filter_order = params[4]

      if filter_type == ocn.FILTER_NONE:
        return signal
      elif filter_type == ocn.FILTER_BUTTERWORTH:
        if filter_pass_type == ocn.FILTER_BAND_PASS:

          override_txrx_filter_width = self.osmod.form_gui.window['cb_override_txrx_filter_width'].get()
          if override_txrx_filter_width:
            filter_width  = int(self.osmod.form_gui.window['in_txrx_filter_width_2'].get())

          self.debug.info_message("filter_width: " + str(filter_width))

          """ filter the output signal """
          for _ in range(repeats):
            sig1 = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(signal, center_frequency + filter_width/2, filter_order, sample_rate)
            signal = sig1
          for _ in range(repeats):
            sig2 = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(signal, center_frequency - filter_width/2, filter_order, sample_rate)
            signal = sig2

        elif filter_pass_type == ocn.FILTER_NOTCH:
          override_txrx_filter_width = self.osmod.form_gui.window['cb_override_txrx_filter_width'].get()
          if override_txrx_filter_width:
            filter_width  = int(self.osmod.form_gui.window['in_txrx_filter_width'].get())
          for _ in range(repeats):
            signal = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(signal, center_frequency - filter_width/2, filter_order, sample_rate)
            signal = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(signal, center_frequency + filter_width/2, filter_order, sample_rate)

        elif filter_pass_type == ocn.FILTER_NOTCH_2:
          for _ in range(repeats):
            sig2 = self.osmod.modulation_object.filter_sharp_cutoff_notch(signal, center_frequency - filter_width/2, center_frequency + filter_width/2, filter_order, sample_rate)
            signal = sig2

        elif filter_pass_type == ocn.FILTER_BAND_PASS_X2:
          """ filter the output signal """
          filter_width_offset = params[2]
          filter_width  = filter_width_offset[0]
          filter_offset = filter_width_offset[1] # offset from center frequncy +- for the two notches

          override_txrx_filter_width = self.osmod.form_gui.window['cb_override_txrx_filter_width'].get()
          if override_txrx_filter_width:
            filter_width  = int(self.osmod.form_gui.window['in_txrx_filter_width'].get())

          center_frequency_a = center_frequency - filter_offset
          center_frequency_b = center_frequency + filter_offset
          signal_a = signal.copy()
          signal_b = signal.copy()

          for _ in range(repeats):
            signal_a = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(signal_a, center_frequency_a + filter_width/2, filter_order, sample_rate)
            signal_a = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(signal_a, center_frequency_a - filter_width/2, filter_order, sample_rate)
            signal_b = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(signal_b, center_frequency_b + filter_width/2, filter_order, sample_rate)
            signal_b = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(signal_b, center_frequency_b - filter_width/2, filter_order, sample_rate)

          return (signal_a + signal_b) / 2

        return signal

    except:
      self.debug.error_message("Exception in apply_filter: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def filter_sharp_cutoff_low_pass(self, signal, cutoff_freq, filter_order, sample_rate):
  #def filter_sharp_cutoff_low_pass(self, signal, cutoff_freq):
    #return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'low',50)
    return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'low',filter_order, sample_rate)

  def filter_sharp_cutoff_high_pass(self, signal, cutoff_freq, filter_order, sample_rate):
  #def filter_sharp_cutoff_high_pass(self, signal, cutoff_freq):
    #return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'highpass',50)
    return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'highpass',filter_order, sample_rate)

  """ This method works well """
  def filter_sharp_cutoff_common(self, signal, cutoff_freq, filter_type, order, sample_rate):
    try:
      #nyquist_frequency = 0.5 * self.osmod.sample_rate
      nyquist_frequency = 0.5 * sample_rate
      normalized_cutoff = cutoff_freq / nyquist_frequency
      sos = butter(order, normalized_cutoff, btype=filter_type, analog=False, output='sos')
      return_value = sosfiltfilt(sos, signal)

    except:
      self.debug.error_message("Exception in filter_low_pass_2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return return_value


  def filter_sharp_cutoff_notch(self, signal, low_cutoff_freq, high_cutoff_freq, filter_order, sample_rate):
    try:
      nyquist_frequency = 0.5 * sample_rate
      normalized_low_cutoff  = low_cutoff_freq / nyquist_frequency
      normalized_high_cutoff = high_cutoff_freq / nyquist_frequency

      sos = butter(N=filter_order, Wn=[normalized_low_cutoff, normalized_high_cutoff], btype="bandstop", fs=sample_rate, output='sos')
      return_value = sosfiltfilt(sos, signal)

    except:
      self.debug.error_message("Exception in filter_low_pass_2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return return_value



  def filter_low_pass(self, signal, cutoff_freq):
    return self.filter_common(signal, cutoff_freq, 'low')

  def filter_high_pass(self, signal, cutoff_freq):
    return self.filter_common(signal, cutoff_freq, 'highpass')

  """ This method works well """
  def filter_common(self, signal, cutoff_freq, filter_type):
    #self.debug.info_message("filter_low_pass_2")
    try:
      nyquist_frequency = 0.5 * self.osmod.sample_rate
      normalized_cutoff = cutoff_freq / nyquist_frequency

      sos = butter(2, normalized_cutoff, btype=filter_type, analog=False, output='sos')

      return_value = sosfiltfilt(sos, signal)

    except:
      self.debug.error_message("Exception in filter_low_pass_2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    #finally:
    #  self.debug.info_message("Completed filter_low_pass_2: ")

    return return_value

  def matched_filter(self, signal, phase_shape):
    return np.convolve(signal, np.conjugate(pulse_shape[::-1]), mode='same')


  """ this works fine"""
  def filterFIRbandpass(self, signal, Fs, low, high):

    self.debug.info_message("filterFIRbandpass: ")
    try:
      nyquist_frequency = 0.5 * self.osmod.sample_rate
      normalized_cutoff_low  = low / nyquist_frequency
      normalized_cutoff_high = high / nyquist_frequency

      N = len(signal)
      t = np.linspace(-N/2, N/2, N, endpoint=False) * (1/Fs)
      """ bandpass flitering """
      bandpass_cutoff = [0.2, 0.7] # normalized!
      filter_taps = firwin(N, bandpass_cutoff, pass_zero = False, window='hamming')
      """ end """

      return_value = np.convolve(signal, filter_taps, mode='same')

    except:
      self.debug.error_message("Exception in filterFIRbandpass: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed filterFIRbandpass: ")

    return return_value

  """ this works fine"""
  def filterFIRlowpass(self, signal, Fs):

    self.debug.info_message("filterFIRlowpass: ")
    try:
      N = len(signal)

      t = np.linspace(-N/2, N/2, N, endpoint=False) * (1/Fs)
      """ lowpass flitering """
      lowpass_cutoff = 0.3 # normalized
      filter_taps = firwin(N, lowpass_cutoff, pass_zero = True)
      """ end """

      return_value = np.convolve(signal, filter_taps, mode='same')

    except:
      self.debug.error_message("Exception in filterFIRlowpass: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed filterFIRlowpass: ")

    return return_value

    
  def filterButterworth(self, data):  
    b, a = butter(4, 100/500, btype='lowpass', analog = False)
    return filtfilt(b, a, data)

      

  """ Coefficient based wave shaping methods"""

  """ this method works fine"""
  def filterSpanRRC(self, signal_length, alpha, T, Fs):  
    self.debug.info_message("filterRRC: ")
    return_value = None
    symbol_span = 1
    total_symbol_span = 3
    try:
      N = signal_length * total_symbol_span

      t = np.linspace(-N/2, N/2, N, endpoint=False) * (1/Fs) * symbol_span * (12*total_symbol_span*self.osmod.symbol_block_size / N)
      #h = np.zeros(N, dtype=np.float32)
      h = np.zeros(N, dtype=np.float64)
      for i in range(N):
        if t[i] == 0.0:
          h[i] = (1.0 - alpha) + ((4 * alpha) / np.pi)
        elif abs(t[i]) == T / (4 * alpha):
          h[i] = (alpha / np.sqrt(2)) * ((1 + (2/np.pi)) * np.sin(np.pi / (4 * alpha)) + (1 - (2/np.pi)) * np.cos(np.pi / (4*alpha)))
        else:
          h[i] = (np.sin(np.pi * t[i] * (1 - alpha) / T) + 4 * alpha * (t[i] / T) * np.cos(np.pi * t[i] * (1+alpha) / T)) / (np.pi * t[i] * (1 - (4 * alpha * t[i] / T) **2 ) /T)

      split_values = np.split(h, total_symbol_span)

    except:
      self.debug.error_message("Exception in filterRRC: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed filterRRC: ")

    return split_values[0], split_values[1], split_values[2]




  """ Charting and plotting methods"""

  def plotWave(self, N, data):
    plt.figure(figsize=(12,4))
    time = np.linspace(-N/2, N/2, N, endpoint=False)
    plt.plot(time, data, label = 'Wave Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.legend()
    plt.show()

  def plotWaveCanvas(self, N, data, canvas):
    self.debug.info_message("plotWaveCanvas")
    self.debug.info_message("num points to plot: " + str(N))
    self.debug.info_message("data len: " + str(len(data)))
    plt.figure(figsize=(4,4))
    time = np.linspace(-N/2, N/2, N, endpoint=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    fig, ax = plt.subplots()
    ax.set_xlim(-N/2, N/2)
    ax.plot(time, data, label = 'Wave Data')
    figure_canvas = FigureCanvasTkAgg(fig, canvas)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)


  def plotConstellationAndSignal(self):
    plt.figure(figsize=(8,6))
    for bits, symbol in constellation_8apsk.items():
      plt.plot(symbol.real,symbol.imag, 'o', label = str(bits))

    plt.plot([s.real for s in modulated_signal])
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.title('8-APSK Constellation and Modulated Signal')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
    
  def stringToTriplet(self, string):
    self.debug.info_message("stringToTriplet")

    try:
      binary_array_pre_fec = []
      bit_triplets1 = []
      bit_triplets2 = []

      sent_triplets_1 = []
      sent_triplets_2 = []

      for char in string:
        self.debug.info_message("processing char: " + str(char) )
        self.osmod.form_gui.txwindowQueue.put(str(char))

        """ decimal index of character """
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
          if self.osmod.process_debug == True:
          #if self.osmod.process_debug == True and self.osmod.form_gui.window['cb_use_preset_message'].get() == True:
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row1), end="", text_color='green', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row2), end="", text_color='green', background_color = 'white')

    except:
      sys.stdout.write("Exception in stringToTriplet: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return [bit_triplets1, bit_triplets2], [sent_triplets_1, sent_triplets_2], binary_array_pre_fec


  def stringToTripletFEC(self, string):
    self.debug.info_message("stringToTripletFEC")

    try:
      bit_triplets1 = []
      bit_triplets2 = []

      sent_triplets_1 = []
      sent_triplets_2 = []

      binary_string = ''

      for char in string:
        self.debug.info_message("processing char: " + str(char) )
        self.osmod.form_gui.txwindowQueue.put(str(char))

        """ decimal index of character """
        index = self.b64_indexfromchar_dict[char]

        """ char to numpy array of binary values. numpy array to binary triplets"""
        binary = format(index, "06b")[0:6]
        self.debug.info_message("binary : " + str(binary) )

        #sub_string = np.binary_repr(index)
        #self.debug.info_message("sub_string : " + str(sub_string) )
        binary_string = binary_string + binary


      self.debug.info_message("binary_string : " + str(binary_string) )
      binary_array_pre_fec = np.fromstring(binary_string, 'u1') - ord('0')
      self.debug.info_message("binary_array_pre_fec : " + str(binary_array_pre_fec) )
      """LDPC code goes here """
      #binary_array_post_ldpc = binary_array_pre_ldpc

      if self.osmod.chunk_num == 0:
        binary_array_post_fec = self.osmod.fec.encodeFEC(binary_array_pre_fec[self.osmod.extrapolate_seqlen * 6:])
        binary_array_post_fec = np.append(binary_array_pre_fec[:self.osmod.extrapolate_seqlen * 6], binary_array_post_fec)
      else:
        binary_array_post_fec = self.osmod.fec.encodeFEC(binary_array_pre_fec)


      self.debug.info_message("binary_array_post_fec : " + str(binary_array_post_fec) )
      post_binary_string = "".join(binary_array_post_fec.astype(str))
      self.debug.info_message("post_binary_string : " + str(post_binary_string) )
      #decimal_value = int(binary_string, 2)

      padding_count = (6 - (len(post_binary_string) % 6)) % 6
      self.debug.info_message("padding_count : " + str(padding_count) )
      post_binary_string = post_binary_string + '0' * padding_count

      for six_bit_seq in range(0, len(post_binary_string), 6):
        binary = post_binary_string[six_bit_seq:six_bit_seq+6]
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
          if self.osmod.process_debug == True:
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row1), end="", text_color='green', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row2), end="", text_color='green', background_color = 'white')

    except:
      sys.stdout.write("Exception in stringToTripletFEC: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return [bit_triplets1, bit_triplets2], [sent_triplets_1, sent_triplets_2], binary_array_pre_fec




  def stringToDoubletQuad(self, string):
    quad_array   = []
    bit_doublets = []

    for char in string:
      binary = format(ord(char), "08b")
      for i in range(0, len(binary), 2):
        pair = binary[i:i + 2]
        bit_doublets.append(pair)
        quad_array.append(self.dict_binpair_to_quad[pair]) 

    return bit_doublets, quad_array



  def PchipCurveInterpolation(self, x_points, x_smooth, y_points, smoothing):
    self.debug.info_message("PchipCurveInterpolation")
    try:
      self.debug.info_message("PchipCurveInterpolation")

      smoothing_spline = UnivariateSpline(x_points, y_points, s=smoothing)
      y_smoothed = smoothing_spline(x_points)

      pchip_interp = PchipInterpolator(x_points, y_smoothed)
      y_smooth = pchip_interp(x_smooth)

      return np.clip(y_smooth, a_min = min(y_points), a_max = max(y_points))

    except:
      self.debug.error_message("Exception in PchipCurveInterpolation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def CubicSplineCurveInterpolation(self, x_points, x_smooth, y_points, smoothing):
    self.debug.info_message("CubicSplineCurveInterpolation")
    try:
      self.debug.info_message("CubicSplineCurveInterpolation")

      smoothing_spline = UnivariateSpline(x_points, y_points, s=smoothing)
      y_smoothed = smoothing_spline(x_points)

      cs = CubicSpline(x_points, y_smoothed)
      y_smooth = cs(x_smooth)

      return np.clip(y_smooth, a_min = min(y_points), a_max = max(y_points))

    except:
      self.debug.error_message("Exception in CubicSplineCurveInterpolation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def BetaSplineCurveInterpolation(self, x_points, x_smooth, y_points, smoothing):
    self.debug.info_message("BetaSplineCurveInterpolation")
    try:
      self.debug.info_message("BetaSplineCurveInterpolation")
      tck = splrep(x_points, y_points, s=smoothing)
      y_smooth = splev(x_smooth, tck)

      return np.clip(y_smooth, a_min = min(y_points), a_max = max(y_points))

    except:
      self.debug.error_message("Exception in BetaSplineCurveInterpolation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def chebyshevCurveInterpolation(self, x_points, x_smooth, y_points, smoothing):
    self.debug.info_message("chebyshevCurveInterpolation")
    try:
      self.debug.info_message("x_points: " + str(x_points))
      self.debug.info_message("y_points: " + str(y_points))
      self.debug.info_message("len(x_points): " + str(len(x_points)))
      self.debug.info_message("len(y_points): " + str(len(y_points)))
      deg = int(smoothing)   # 10
      cheby_fit = T.fit(x_points, y_points, deg)
      minx = min(x_points)
      maxx = max(x_points)

      #x_smooth = np.linspace(minx, maxx, int(maxx-minx))
      y_cheby = cheby_fit(x_smooth)
      self.debug.info_message("x_smooth: " + str(x_smooth))
      self.debug.info_message("y_cheby: " + str(y_cheby))
      return y_cheby

    except:
      self.debug.error_message("Exception in chebyshevCurveInterpolation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def interpolatePhaseDrift(self, signal, median_block_offset, frequency, where, calc_type):
    self.debug.info_message("interpolatePhaseDrift")
    try:
      interpolated_signal = np.zeros(len(signal), dtype=float)
      wrap_around_signal = np.zeros(len(signal), dtype=int)
      signal_angle = np.angle(signal)

      num_waves    = self.osmod.parameters[4]
      num_points = int(self.osmod.sample_rate / frequency)
      term1 = num_points*num_waves

      two_times_pi = np.pi * 2
      max_phase_value = two_times_pi / 8
      wrap_around = 0
      gradient = 0

      num_values   = int((len(signal_angle) - median_block_offset) // self.osmod.symbol_block_size)
      original_phase_values = [0]*num_values
      error_values = [0]*num_values
      wrap_values  = [0]*num_values
      x_values     = [0]*num_values
      self.debug.info_message("num_values: " + str(num_values))
      for x in range(median_block_offset, len(signal_angle), self.osmod.symbol_block_size):
        if x <= len(signal_angle) - self.osmod.symbol_block_size:
          middle_this  = x + where
          middle_last  = x + where - self.osmod.symbol_block_size
          #original_phase_values[x] = ((sum(signal_angle[middle_this-(term1):middle_this+(term1)-1]) / (2*term1)) + np.pi) % max_phase_value
          original_value = sum(signal_angle[middle_this-(term1):middle_this+(term1)-1]) / (2*term1)
          this_block_phase = ((sum(signal_angle[middle_this-(term1):middle_this+(term1)-1]) / (2*term1)) + np.pi) % max_phase_value
          last_block_phase = ((sum(signal_angle[middle_last-(term1):middle_last+(term1)-1]) / (2*term1)) + np.pi) % max_phase_value
          interpolated_signal[middle_this] = this_block_phase

          self.debug.info_message("x: " + str(x))
          self.debug.info_message("block: " + str(int((x-median_block_offset)//self.osmod.symbol_block_size)))

          original_phase_values[int((x-median_block_offset)//self.osmod.symbol_block_size)] = original_value
          error_values[int((x-median_block_offset)//self.osmod.symbol_block_size)] = this_block_phase
          x_values[int((x-median_block_offset)//self.osmod.symbol_block_size)] = x

          """ stays in same plane """
          min_diff_1 = (this_block_phase - last_block_phase)
          """ wrap around +1 """
          min_diff_2 = (this_block_phase + max_phase_value) - last_block_phase
          """ wrap around -1 """
          min_diff_3 = (last_block_phase + max_phase_value) - this_block_phase

          if x < median_block_offset + self.osmod.symbol_block_size:
            wrap_around_signal[middle_this] = 0
          else:      
            self.debug.info_message("this last diff 1 diff2 diff 3: " + str(this_block_phase) + ", " + str(last_block_phase) + ", " + str(min_diff_1) + ", " + str(min_diff_2) + ", " + str(min_diff_3))
            if abs(min_diff_1) < max_phase_value/2:
              self.debug.info_message("stays in same plane")
              #gradient = this_block_phase - last_block_phase
            elif abs(min_diff_2) < abs(min_diff_3) and abs(min_diff_2) < max_phase_value/2:
              self.debug.info_message("wrap around +1")
              wrap_around = wrap_around + 1
              #gradient = (this_block_phase + max_phase_value) - last_block_phase
            elif abs(min_diff_3) < abs(min_diff_2) and abs(min_diff_3) < max_phase_value/2:
              self.debug.info_message("wrap around -1")
              wrap_around = wrap_around - 1
              #gradient = (last_block_phase + max_phase_value) - this_block_phase
            else:
              self.debug.info_message("unknown")

            wrap_around_signal[middle_this] = wrap_around
            wrap_values[int((x-median_block_offset)//self.osmod.symbol_block_size)] = wrap_around


      last_value  = 0
      wrap_around = 0
      points_x = []
      points_y = []

      self.debug.info_message("creating interpolated")
      self.debug.info_message("error_values: " + str(error_values))
      self.debug.info_message("wrap_values: " + str(wrap_values))

      """ normalize. wrap values already cumulative"""
      #rolling_wrap = 0
      for i in range (0, len(error_values)):
        #rolling_wrap = rolling_wrap + wrap_values[i]
        #error_values[i] = error_values[i] + (rolling_wrap * max_phase_value)
        error_values[i] = error_values[i] + (wrap_values[i] * max_phase_value)

      self.debug.info_message("original_phase_values: " + str(original_phase_values))
      self.debug.info_message("error_values: " + str(error_values))
      self.debug.info_message("original + error: " + str(original_phase_values + error_values))
      for i in range (0, len(original_phase_values)):
        corrected_phase_value = original_phase_values[i] + error_values[i]
        adjusted_phase_value = (corrected_phase_value + two_times_pi) / two_times_pi
        adjusted_for_eighths = int((adjusted_phase_value * 8) % 8)
        self.debug.info_message("character code: " + str(adjusted_for_eighths))

      #for i in range (1, len(error_values)):
      #  gradient    = error_values[i] - error_values[i-1]
      #  anticipated = error_values[i] + gradient
      #  if error_values[i] - anticipated < max_phase_value / 2:
      #    self.debug.info_message("all is good")


      index_count = 0
      for x in range(0, len(signal_angle)):
        if (x - median_block_offset) % self.osmod.symbol_block_size == 0:
          if index_count < num_values and x_values[index_count] == x:
            self.debug.info_message("processing: " + str(x))
            self.debug.info_message("index_count: " + str(index_count))
            last_value  = error_values[index_count]
            interpolated_signal[x] = last_value
            index_count = index_count + 1
          else:
            interpolated_signal[x] = last_value
        else:
          interpolated_signal[x] = last_value

      if calc_type == ocn.PHASE_ERROR_ROUGH:
        return interpolated_signal

      if calc_type == ocn.PHASE_ERROR_SMOOTH:

        minx = min(x_values)
        maxx = max(x_values)
        x_smooth = np.linspace(minx, maxx, int(maxx-minx))
        cheby = self.chebyshevCurveInterpolation(x_values, x_smooth, error_values, 10)
        interpolated_signal[min(x_values):max(x_values)] = cheby
        return interpolated_signal

    except:
      self.debug.error_message("Exception in interpolatePhaseDrift: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getStrongestNFromFdd(self, fdd, N):
    #self.debug.info_message("getStrongestNFromFdd")

    fdd_strongest = {}
    try:
      frequencies = fdd["frequency"]
      fft_magnitudes = fdd["magnitude"]

      #self.debug.info_message("argsort: " + str(np.argsort(fft_magnitudes)[:-6:-1]))
      top_n_indices = np.argsort(fft_magnitudes)[:-6:-1]
      strongest_frequencies = frequencies[top_n_indices]
      #self.debug.info_message("BEST  === strongest_frequencies: " + str(strongest_frequencies))

      top_n_indices = np.argsort(fft_magnitudes)[-N:][::-1]
      #self.debug.info_message("top_n_indices: " + str(top_n_indices))
      strongest_frequencies = frequencies[top_n_indices]
      strongest_magnitudes  = fft_magnitudes[top_n_indices]

      fdd_strongest["frequency"] = strongest_frequencies
      fdd_strongest["magnitude"] = strongest_magnitudes

      #self.debug.info_message("strongest_frequencies: " + str(strongest_frequencies))
      #self.debug.info_message("strongest_magnitudes: " + str(strongest_magnitudes))
    except:
      self.debug.error_message("Exception in getStrongestNFromFdd: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return fdd_strongest


  def createFdd(self, data):
    self.debug.info_message("createFdd")

    fdd = {}
    try:
      fft_output = np.fft.fft(data)
      fdd["output"]   = fft_output
      fdd["data_len"] =   len(data)
    except:
      self.debug.error_message("Exception in createFdd: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return fdd


  #def processFddLoHi(self, data, lo, high):
  def processFddLoHi(self, fdd, lo, high):
    #self.debug.info_message("processFddLoHi")

    #fdd = {}
    try:
      #fft_output = np.fft.fft(data)
      #frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      fft_output  = fdd["output"]
      data_len    = fdd["data_len"]
      #frequencies = np.fft.fftfreq(data_len, 1/self.osmod.sample_rate)
      frequencies = np.fft.fftfreq(data_len, 1/self.osmod.getRxSampleRate())

      positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
      fft_magnitudes = np.abs(fft_output)[positive_frequency_indices]
      frequencies = frequencies[positive_frequency_indices]
      fdd["frequency"] = frequencies
      fdd["magnitude"] = fft_magnitudes
      fdd["output"]    = fft_output

    except:
      self.debug.error_message("Exception in processFddLoHi: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return fdd


  def getStrongestFrequencies(self, data, N, lo, high):
    self.debug.info_message("getStrongestFrequencies")
    self.debug.info_message("lo: " + str(lo))
    self.debug.info_message("high: " + str(high))
    try:
      def mysort(x):
        arg_x = np.sort(x)
        rank = [np.where(arg_x == i)[0][0] for i in x]
        return np.array(rank)

      fft_output = np.fft.fft(data)
      frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
      fft_magnitudes = np.abs(fft_output)[positive_frequency_indices]
      #fft_magnitudes = np.abs(fft_output)
      frequencies = frequencies[positive_frequency_indices]
      self.debug.info_message("frequencies: " + str(frequencies))
      self.debug.info_message("fft_magnitudes: " + str(fft_magnitudes))


      #strongest_index = np.argmax(fft_magnitudes)
      #strong_freqs = frequencies[strongest_index]
      #strong_magnitudes = fft_magnitudes[strongest_index]
      #sys.stdout.write("test strongest index: " + str(strongest_index) + "\n")
      #sys.stdout.write("test strongest frequency: " + str(strong_freqs) + "\n")
      #sys.stdout.write("test strong_magnitudes: " + str(strong_magnitudes) + "\n")
      #newarr = np.delete(fft_magnitudes, strongest_index)
      #strongest_index2 = np.argmax(newarr)
      #sys.stdout.write("test strongest index 2: " + str(strongest_index2) + "\n")



      self.debug.info_message("argsort: " + str(np.argsort(fft_magnitudes)[:-6:-1]))
      #self.debug.info_message("argsort: " + str(np.argsort(fft_magnitudes).ravel()))
      #self.debug.info_message("sort: " + str(sorted(fft_magnitudes, key=lambda x: x[0])))
      #self.debug.info_message("mysort: " + str(mysort(fft_magnitudes)[-N:]))
      top_n_indices = np.argsort(fft_magnitudes)[:-6:-1]
      strongest_frequencies = frequencies[top_n_indices]
      self.debug.info_message("BEST  === strongest_frequencies: " + str(strongest_frequencies))


      top_n_indices = np.argsort(fft_magnitudes)[-N:][::-1]
      self.debug.info_message("top_n_indices: " + str(top_n_indices))
      #top_n_indices = np.argsort(fft_magnitudes)[-N:]
      strongest_frequencies = frequencies[top_n_indices]
      strongest_magnitudes  = fft_magnitudes[top_n_indices]

      self.debug.info_message("strongest_frequencies: " + str(strongest_frequencies))
      self.debug.info_message("strongest_magnitudes: " + str(strongest_magnitudes))
    except:
      self.debug.error_message("Exception in getStrongestFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return strongest_frequencies, strongest_magnitudes


  """ This method works fine"""
  def getIsSignalPresent(self, data, watch_freq):
    #self.debug.info_message("getIsSignalPresent")
    try:
      fdd = {}
      delta = 5
      fft_output = np.fft.fft(data)
      #frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      frequencies = np.fft.fftfreq(len(data), 1/self.osmod.getRxSampleRate())
      positive_frequency_indices = np.where((frequencies > watch_freq - delta) & (frequencies < watch_freq + delta))[0]

      fft_magnitude = np.abs(fft_output)[positive_frequency_indices]
      frequencies = frequencies[positive_frequency_indices]
      strongest_index = np.argmax(fft_magnitude)

      fft_data = np.abs(fft_output)
      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      #sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      #sys.stdout.write("strongest frequency: " + str(strong_freqs) + "\n")
      #sys.stdout.write("strong_magnitudes: " + str(strong_magnitudes) + "\n")

      fdd["output"]   = fft_output
      fdd["data_len"] = len(data)

    except:
      self.debug.error_message("Exception in getIsSignalPresent: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    #finally:
    #  self.debug.info_message("Completed getIsSignalPresent: ")

    return strong_freqs, strong_magnitudes, fft_output, len(data), fdd


  def getStrongestFrequencyCZT(self, signal, lo, high, samplerate, czt_num_points):
    self.debug.info_message("getStrongestFrequencyCZT")
    try:
      #num_points=5000 #number of frequency bins
      num_points = czt_num_points #50000 #number of frequency bins

      czt_out = zoom_fft(signal, [lo, high], m=num_points, fs=samplerate)
      freqs = np.linspace(lo, high, num_points)
      strongest_index = np.argmax(np.abs(czt_out))
      strongest_freq = freqs[strongest_index]

    except:
      self.debug.error_message("Exception in getStrongestFrequencyCZT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed getStrongestFrequencyCZT: ")

    return strongest_freq



  """ This method works fine"""
  def getStrongestFrequency2(self, data, lo, high, samplerate):
    self.debug.info_message("getStrongestFrequency2")
    try:
      fft_output = np.fft.fft(data)
      frequencies = np.fft.fftfreq(len(data), 1/samplerate)
      positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
      fft_magnitude = np.abs(fft_output)[positive_frequency_indices]

      frequencies = frequencies[positive_frequency_indices]
      strongest_index = np.argmax(fft_magnitude)

      """ calculated interpolated frequency """
      #fft_data = np.abs(fft_output)
      #interpolated_peak_index = self.fft_parabolic_interpolation(fft_data, np.argmax(fft_data[:len(fft_data)//2]) )
      #interpolated_frequency = interpolated_peak_index * self.osmod.sample_rate / len(data)

      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      sys.stdout.write("strongest frequency: " + str(strong_freqs) + "\n")
      #sys.stdout.write("strongest interpolated frequency: " + str(interpolated_frequency) + "\n")
      sys.stdout.write("strong_magnitudes: " + str(strong_magnitudes) + "\n")
  
    except:
      self.debug.error_message("Exception in getStrongestFrequency2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed getStrongestFrequency2: ")

    return strong_freqs




  """ This method works fine"""
  def getStrongestFrequency(self, data, lo, high):
    self.debug.info_message("getStrongestFrequency")
    try:
      fft_output = np.fft.fft(data)
      frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
      fft_magnitude = np.abs(fft_output)[positive_frequency_indices]

      frequencies = frequencies[positive_frequency_indices]
      strongest_index = np.argmax(fft_magnitude)

      """ calculated interpolated frequency """
      #fft_data = np.abs(fft_output)
      #interpolated_peak_index = self.fft_parabolic_interpolation(fft_data, np.argmax(fft_data[:len(fft_data)//2]) )
      #interpolated_frequency = interpolated_peak_index * self.osmod.sample_rate / len(data)

      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      sys.stdout.write("strongest frequency: " + str(strong_freqs) + "\n")
      #sys.stdout.write("strongest interpolated frequency: " + str(interpolated_frequency) + "\n")
      sys.stdout.write("strong_magnitudes: " + str(strong_magnitudes) + "\n")
  
    except:
      self.debug.error_message("Exception in getStrongestFrequency: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed getStrongestFrequency: ")

    return strong_freqs


  def fft_parabolic_interpolation(self, data, peak_index):
    self.debug.info_message("fft_parabolic_interpolation")
    try:
      y1,y2,y3 = np.log(data[peak_index-1:peak_index+2])    
      interpolated_peak_index = peak_index + (y1 - y3) / (2 * (y1 - 2 * y2 + y3))
      return interpolated_peak_index

    except:
      self.debug.error_message("Exception in fft_parabolic_interpolation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calculateSNR(self, signal, signal_frequency):
    self.debug.info_message("calculateSNR")

    try:
      fft_signal = np.fft.fft(signal)
      frequencies = np.fft.fftfreq(len(fft_signal), 1/self.osmod.sample_rate)

      """ use best method to determine signal width"""
      freq_low_signal  = signal_frequency[0] + min(self.osmod.fft_filter[0], self.osmod.fft_interpolate[0])
      freq_high_signal = signal_frequency[1] + max(self.osmod.fft_filter[3], self.osmod.fft_interpolate[3])

      """
      if self.osmod.tx_filter[2] > 0:
        freq_low_signal  = self.osmod.center_frequency - (self.osmod.tx_filter[2]/2)    
        freq_high_signal = self.osmod.center_frequency + (self.osmod.tx_filter[2]/2)    

        # is this a Filtered Carrier mode?
        if self.osmod.carrier_separation > (freq_high_signal - freq_low_signal):
          freq_low_signal  = self.osmod.center_frequency - ((self.osmod.carrier_separation)/2)    
          freq_high_signal = self.osmod.center_frequency + ((self.osmod.carrier_separation)/2)      

      else:
        freq_low_signal  = signal_frequency[0] + self.osmod.fft_filter[0]
        freq_high_signal = signal_frequency[1] + self.osmod.fft_filter[3]
      """

      freq_indices = np.where((frequencies >= freq_low_signal) & (frequencies <= freq_high_signal))
      signal_psd = np.abs(fft_signal[freq_indices])**2
      self.debug.info_message("signal_psd: " + str(signal_psd) )

      freq_low_noise = 250
      freq_high_noise = 2750
      freq_indices = np.where(((frequencies > freq_low_noise) & (frequencies < freq_low_signal)) | ((frequencies > freq_high_signal) & (frequencies < freq_high_noise) ))
      #freq_indices = np.where(((frequencies > freq_low_noise) & (frequencies < freq_high_noise) ))
      noise_psd = np.abs(fft_signal[freq_indices])**2
      self.debug.info_message("noise_psd: " + str(noise_psd) )

      signal_width = freq_high_signal - freq_low_signal
      noise_width  = 2500 - signal_width
      #bandwidth_factor = 10 * np.log10(noise_width / signal_width) 
      bandwidth_factor = (noise_width / signal_width) 


      """ noise in (2500 - signal width) in Hz"""
      noise_power  = np.sum(noise_psd)

      """ SNR over 50 Hz """
      SNR_50 = 10 * np.log10((np.sum(signal_psd) - (noise_power / bandwidth_factor)) / (noise_power / bandwidth_factor))

      """ signal power with noise subtracted out...approximation"""
      signal_power = np.sum(signal_psd) -  (noise_power / bandwidth_factor)

      """ noise power over full 2500 Hz """
      noise_power  = noise_power + (noise_power / bandwidth_factor)

      """ SNR over 2500 Hz """
      SNR_2500 = 10 * np.log10(signal_power / noise_power)



      """
      signal_power_db = 10 * np.log10(signal_power)
      noise_power_db  = 10 * np.log10(noise_power)
      SNR = signal_power_db - (noise_power_db - bandwidth_factor)
      self.debug.info_message("SNR: " + str(SNR))
      """

      return SNR_2500
      #return SNR_50

    except:
      self.debug.error_message("Exception in calculateSNR: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def calculate_EbN0(self, noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency):
    ebn0_db, ebn0, SNR_equiv_db = self.calculate_EbN0_1(noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency)
    ebn0_db_alt, ebn0_alt, SNR_equiv_db_alt = self.calculate_EbN0_2(noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency)
    #ebn0_alt = self.calculate_EbN0_2(noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency)
    self.osmod.form_gui.window['text_ebn0db_value_alt'].update("Eb/N0 dB (alt): "f"{ebn0_db_alt:.3f}")
    self.osmod.form_gui.window['text_snr_value_alt'].update("SNR dB (alt): "f"{SNR_equiv_db_alt:.3f}")
    return ebn0_db, ebn0, SNR_equiv_db

  def calculate_EbN0_1(self, noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency):
    self.debug.info_message("calculateSNR_EbN0")

    gc.collect()

    try:
      fft_signal = np.fft.fft(noise_free_signal)
      frequencies = np.fft.fftfreq(len(fft_signal), 1/self.osmod.sample_rate)

      freq_low_signal  = signal_frequency[0] + min(self.osmod.fft_filter[0], self.osmod.fft_interpolate[0])
      freq_high_signal = signal_frequency[1] + max(self.osmod.fft_filter[3], self.osmod.fft_interpolate[3])

      signal_width = freq_high_signal - freq_low_signal

      freq_indices = np.where((frequencies >= freq_low_signal) & (frequencies <= freq_high_signal))
      signal_power_spectrum = np.abs(fft_signal[freq_indices])**2
      self.debug.info_message("signal_power_spectrum: " + str(signal_power_spectrum) )

      fft_noise = np.fft.fft(noisy_signal)
      frequencies = np.fft.fftfreq(len(fft_noise), 1/self.osmod.sample_rate)
      freq_low_noise = 0
      freq_high_noise = 3000
      freq_indices = np.where(((frequencies > freq_low_noise) & (frequencies <= freq_low_signal)) | ((frequencies >= freq_high_signal) & (frequencies < freq_high_noise) ))
      noise_power_spectrum = (np.abs(fft_noise[freq_indices])**2) / (len(noisy_signal) ** 2)
      average_power = np.mean(noise_power_spectrum)

      N0 = average_power / (self.osmod.sample_rate / len(noise_free_signal))
      self.debug.info_message("noise_psd: " + str(noise_power_spectrum) )

      # Parseval's normalization
      average_power = np.sum(signal_power_spectrum) / (len(noise_free_signal) ** 2)
      Eb = average_power / bit_rate 
      self.debug.info_message("Eb: " + str(Eb) )

      ebn0 = Eb / N0
      ebn0_db = 10 * np.log10(ebn0)
      """ equivalent SNR over standard 2500 Hz bandwidth"""
      SNR_equiv_db = ebn0 + 10 * np.log10(bit_rate / 2500)

      self.debug.info_message("Eb/N0: " + "{:.2f}".format(ebn0) )
      self.debug.info_message("Eb/N0 (dB): " + "{:.2f}".format(ebn0_db) + " (dB)")
      self.debug.info_message("Equivalent SNR over 2500 Hz standard (dB): " + "{:.2f}".format(SNR_equiv_db) + " (dB)")

      return float(ebn0_db), float(ebn0), float(SNR_equiv_db)
    except:
      self.debug.error_message("Exception in calculateSNR_EbN0: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))




  def calculate_EbN0_2(self, noisy_signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency):
    self.debug.info_message("calculateSNR_EbN0")

    gc.collect()

    try:
      fft_signal = np.fft.fft(noise_free_signal)
      frequencies = np.fft.fftfreq(len(fft_signal), 1/self.osmod.sample_rate)

      """ use best method to determine signal width"""
      freq_low_signal  = signal_frequency[0] + min(self.osmod.fft_filter[0], self.osmod.fft_interpolate[0])
      freq_high_signal = signal_frequency[1] + max(self.osmod.fft_filter[3], self.osmod.fft_interpolate[3])
      """
      if self.osmod.tx_filter[2] > 0:
      #if False:
        freq_low_signal  = center_frequency - (self.osmod.tx_filter[2]/2)    
        freq_high_signal = center_frequency + (self.osmod.tx_filter[2]/2)    

        # is this a Filtered Carrier mode?
        if self.osmod.carrier_separation > (freq_high_signal - freq_low_signal):
          freq_low_signal  = center_frequency - ((self.osmod.carrier_separation + 2)/2)    
          freq_high_signal = center_frequency + ((self.osmod.carrier_separation + 2)/2)      
      else:
        freq_low_signal  = signal_frequency[0] + self.osmod.fft_filter[0]
        freq_high_signal = signal_frequency[1] + self.osmod.fft_filter[3]
      """

      signal_width = freq_high_signal - freq_low_signal

      freq_indices = np.where((frequencies >= freq_low_signal) & (frequencies <= freq_high_signal))
      signal_power = np.abs(fft_signal[freq_indices])**2
      self.debug.info_message("signal_power: " + str(signal_power) )

      fft_noise = np.fft.fft(noisy_signal)
      frequencies = np.fft.fftfreq(len(fft_noise), 1/self.osmod.sample_rate)
      freq_low_noise = 0
      freq_high_noise = 3000
      freq_indices = np.where(((frequencies > freq_low_noise) & (frequencies <= freq_low_signal)) | ((frequencies >= freq_high_signal) & (frequencies < freq_high_noise) ))
      noise_power = np.abs(fft_noise[freq_indices])**2
      self.debug.info_message("noise_psd: " + str(noise_power) )

      #Eb = np.mean(signal_power) / bit_rate
      Eb = np.sum(signal_power) / bit_rate # total signal power
      #Eb = (np.sum(signal_power) / len(noise_free_signal)) / bit_rate # total signal power
      self.debug.info_message("Eb: " + str(Eb) )

      """ N0 is often derived using average (mean)"""
      #N0 = np.mean(noise_power)
      N0 = np.sum(noise_power) / (3000 - signal_width) # spectral density...power per 1 Hz
      self.debug.info_message("N0: " + str(N0) )
      """ ...but the definition states that N0 is psd in 1Hz of bandwidth..."""

      ebn0 = Eb / N0
      ebn0_db = 10 * np.log10(ebn0)
      """ equivalent SNR over standard 2500 Hz bandwidth"""
      SNR_equiv_db = ebn0 + 10 * np.log10(bit_rate / 2500)

      self.debug.info_message("Eb/N0: " + "{:.2f}".format(ebn0) )
      self.debug.info_message("Eb/N0 (dB): " + "{:.2f}".format(ebn0_db) + " (dB)")
      self.debug.info_message("Equivalent SNR over 2500 Hz standard (dB): " + "{:.2f}".format(SNR_equiv_db) + " (dB)")

      return float(ebn0_db), float(ebn0), float(SNR_equiv_db)
    except:
      self.debug.error_message("Exception in calculateSNR_EbN0: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))





  def calculateBER(self, bit_triplets):
    self.debug.info_message("calculateBER")
    try:
      self.debug.info_message("calculateBER")

    except:
      self.debug.error_message("Exception in calculateBER: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ bandpass filter fft"""
  def bandpass_filter_fft2(self, signal, freq_lo, freq_hi):
    self.debug.info_message("bandpass_filter_fft")

    try:
      fft_signal   = np.fft.fft(signal)
      frequencies  = np.fft.fftfreq(len(signal), 1/self.osmod.sample_rate)
      mask         = (np.abs(frequencies) >= freq_lo) & (np.abs(frequencies) <= freq_hi)
      fft_filtered = fft_signal * mask
      filtered_signal = np.fft.ifft(fft_filtered)

    except:
      self.debug.error_message("Exception in bandpass_filter_fft2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return filtered_signal, fft_signal[mask]

  """ bandpass filter fft"""
  def bandpass_filter_fft(self, signal, freq_lo, freq_hi):
    self.debug.info_message("bandpass_filter_fft")

    self.debug.info_message("len(signal): " + str(len(signal)))

    #if False:
    if True:
      self.debug.info_message("FFT_INPUT max: " + str(np.max(signal)))

      fft_result, sig2 =  self.bandpass_filter_fft_python(signal, freq_lo, freq_hi)

      self.debug.info_message("FFT_RESULT data type: " + str(type(fft_result)))
      self.debug.info_message("FFT_RESULT data subtype: " + str(fft_result.dtype))
      self.debug.info_message("FFT_RESULT length: " + str(len(fft_result)))
      self.debug.info_message("FFT_RESULT max: " + str(np.max(fft_result)))

      #if isinstance(fft_result, np.ndarray) and np.issubdtype(fft_result.dtype, np.complex128):
      #  self.debug.info_message("FFT RESULT IS NDARRAY OF COMPLEX128")
      return fft_result, sig2
    else:
      self.debug.info_message("FFT_INPUT max: " + str(np.max(signal)))

      fft_result, sig2 =  self.bandpass_filter_fft_c(signal, freq_lo, freq_hi)

      self.debug.info_message("FFT_RESULT data type: " + str(type(fft_result)))
      self.debug.info_message("FFT_RESULT data subtype: " + str(fft_result.dtype))
      self.debug.info_message("FFT_RESULT length: " + str(len(fft_result)))
      self.debug.info_message("FFT_RESULT max: " + str(np.max(fft_result)))

      return fft_result, sig2

  def bandpass_filter_fft_c(self, signal, freq_lo, freq_hi):
    self.debug.info_message("bandpass_filter_fft_c")

    try:
        if np.issubdtype(signal.dtype, np.complex128):
          self.debug.info_message("signal is complex128")
          complex_signal = signal
        else:
          self.debug.info_message("converting signal to complex128")
          complex_signal = signal.astype(complex)

        for i in range(0, 10):
          self.debug.info_message("signal[i] is: " + str(complex_signal[i]))
        self.debug.info_message("freq_lo is: " + str(freq_lo))
        self.debug.info_message("freq_hi is: " + str(freq_hi))

        freq_array = np.array([freq_lo, freq_hi], dtype=np.float32)

        num_carriers = 1
        num_output_items = 1
        fft_output = [0] * num_output_items
        output_signal = np.zeros_like(complex_signal)
        #output_signal[0].real = 123
        fft_output[0] = output_signal

        c_freq_lo            = ptoc_float(freq_lo)
        c_freq_hi            = ptoc_float(freq_hi)
        c_freq_array         = ptoc_float_array(freq_array)
        c_num_carriers       = num_carriers
        c_sample_rate        = self.osmod.sample_rate
        c_signal_length      = len(signal)

        c_fft_output    = (ctypes.POINTER(ctypes.c_double) * num_output_items)()
        c_fft_output[0] = ptoc_double_array(fft_output[0])

        self.osmod.compiled_lib.fft.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags = 'C'), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double) ) ]
        self.osmod.compiled_lib.fft.restype = ctypes.c_int
        success = self.osmod.compiled_lib.fft(complex_signal, c_freq_array, c_num_carriers, c_sample_rate, c_signal_length, c_fft_output)

        for i in range(0, 10):
          #output = fft_output[0].astype(complex)
          output = fft_output[0]
          self.debug.info_message("fft_output[i] is: " + str(output[i]))
        return fft_output[0], 0
    except:
      sys.stdout.write("Exception in bandpass_filter_fft_c: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


    #return signal

  """ bandpass filter fft"""
  def bandpass_filter_fft_python(self, signal, freq_lo, freq_hi):
    self.debug.info_message("bandpass_filter_fft_python")

    try:
      fft_signal  = sp.fft.fft(signal)
      frequencies = sp.fft.fftfreq(len(signal), 1/self.osmod.sample_rate)
      mask         = (np.abs(frequencies) >= freq_lo) & (np.abs(frequencies) <= freq_hi)
      fft_filtered = fft_signal * mask
      filtered_signal = sp.fft.ifft(fft_filtered)
    except:
      self.debug.error_message("Exception in bandpass_filter_fft: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    self.debug.info_message("fft signal data type: " + str(filtered_signal.dtype))

    return filtered_signal, fft_signal[mask]


  """ lowpass filter fft ????????????????????"""
  def lowpass_filter_fft(self, signal, freq_hi):
    self.debug.info_message("lowpass_filter_fft")

    try:
      fft_signal  = sp.fft.fft(signal)
      frequencies = sp.fft.fftfreq(len(signal), 1/self.osmod.sample_rate)
      mask         = (np.abs(frequencies) <= freq_hi)
      fft_filtered = fft_signal * mask
      filtered_signal = sp.fft.ifft(fft_filtered)
    except:
      self.debug.error_message("Exception in lowpass_filter_fft: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    self.debug.info_message("fft signal data type: " + str(filtered_signal.dtype))

    return filtered_signal, fft_signal[mask]


  def shiftAllFrequenciesExp(self, signal, shift_hz, sample_rate):
    self.debug.info_message("shiftAllFrequenciesExp")

    try:
      #fs = self.osmod.sample_rate
      fs = sample_rate # self.osmod.getRxSampleRate()
      num_samples = len(signal)
      t = np.arange(num_samples) / fs
      x = np.asarray(signal, dtype = np.float64)
      analytic_signal = hilbert(x)
      shift_vector = np.exp(1j * 2 * np.pi * shift_hz * t)
     
      shifted_signal = analytic_signal * shift_vector
      return shifted_signal #.astype(np.float64)

    except:
      self.debug.error_message("Exception in shiftAllFrequenciesExp: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def shiftAllFrequencies(self, signal, shift_hz, sample_rate):
    self.debug.info_message("shiftAllFrequencies")

    try:
      n = len(signal)
      #fs = self.osmod.sample_rate # * 10
      fs = sample_rate # self.osmod.getRxSampleRate()

      window = np.ones(n)
      #window = np.hanning(n)
      #window = np.hamming(n)
      #window = np.blackman(n)
      #window = np.kaiser(n, beta=5)
      #window = np.kaiser(n, beta=10)

      #window = np.hanning(n)
      self.debug.info_message("LOC 1")

      signal = signal.astype(np.float64)
      window = window.astype(np.float64)

      fft_data = np.fft.rfft(signal * window)

      self.debug.info_message("LOC 2")

      #fft_signal  = sp.fft.fft(signal)
      frequencies = np.fft.rfftfreq(len(signal), d=1/fs)

      self.debug.info_message("LOC 3")

      bins_to_shift = int(round(shift_hz * n / fs))
      #bins_to_shift = int(round(shift_hz * n / (fs * 10)))
      #bins_to_shift = int(round(bins_to_shift / 10))

      self.debug.info_message("bins_to_shift: " + str(bins_to_shift))

      shifted_fft = np.zeros_like(fft_data, dtype=np.complex128)

      if bins_to_shift > 0:
        shifted_fft[bins_to_shift:] = fft_data[:-bins_to_shift]
        shifted_fft[:bins_to_shift] = 0

      elif bins_to_shift < 0:
        shift_abs = abs(bins_to_shift)
        shifted_fft[:-shift_abs] = fft_data[shift_abs:]
        shifted_fft[-shift_abs:] = 0
      else:
        shifted_fft = fft_data

      shifted_signal = np.fft.irfft(shifted_fft)

      max_val = np.max(np.abs(shifted_signal))

      self.debug.info_message("max_val: " + str(max_val))

      if max_val > 0:
        shifted_signal = shifted_signal / max_val * 0.99

      #shifted_signal = (shifted_signal * 32767).astype(np.int16)
      shifted_signal = (shifted_signal * 32767).astype(np.float64)
      return shifted_signal

      #mask         = (np.abs(frequencies) >= freq_lo) & (np.abs(frequencies) <= freq_hi)
      #fft_filtered = fft_signal * mask
      #filtered_signal = sp.fft.ifft(fft_filtered)

      #self.debug.info_message("fft signal data type: " + str(filtered_signal.dtype))

      #return filtered_signal, fft_signal[mask]

    except:
      self.debug.error_message("Exception in shiftAllFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def resampleDopplerShiftFFT(self, signal, ratio):
    self.debug.info_message("resampleDopplerShiftFFT")
    try:
      num_new = int(round(len(signal) * ratio))

      if num_new < 1:
        return self.resampleDopplerShiftFFT_up(signal, num_new), num_new - len(signal)
      elif num_new > 1:
        return self.resampleDopplerShiftFFT_down(signal, num_new), num_new - len(signal)
      else:
        return signal, 0

    except:
      self.debug.error_message("Exception in resampleDopplerShiftFFT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def resampleDopplerShiftFFT_up(self, signal, num_new):
    self.debug.info_message("resampleDopplerShiftFFT")
    try:
      N = len(signal)
      X = np.fft.fft(signal)

      X_new = np.zeros(num_new, dtype=complex)

      half_N = (N + 1) // 2
      X_new[:half_N] = X[:half_N]
      X_new[-(N // 2):] = X[-(N // 2):]

      return np.fft.ifft(X_new).real * (num_new / N)

    except:
      self.debug.error_message("Exception in resampleDopplerShiftFFT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def resampleDopplerShiftFFT_down(self, signal, num_new):
    self.debug.info_message("resampleDopplerShiftFFT")
    try:
      N = len(signal)
      X = np.fft.fft(signal)

      half_new = (num_new + 1) // 2
      x_new = np.concatenate([X[:half_new], X[-(num_new // 2):]])

      if num_new % 2 == 0:
        X_new[half_new] = X_new[half_new] / 2

      return np.fft.ifft(X_new).real * (num_new / N)

    except:
      self.debug.error_message("Exception in resampleDopplerShiftFFT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def resampleDopplerShift(self, signal, orig_fs, new_fs):
    self.debug.info_message("resampleDopplerShift")

    try:
      #from scipy import signal
      #orig_fs = self.osmod.sample_rate
      #new_fs = 
      #num_samples_new_signal = int(len(signal) * new_fs / orig_fs)
      num_samples_new_signal = int(round(len(signal) * new_fs / orig_fs))
      resampled_signal = scipy_signal.resample(signal, num_samples_new_signal)

      self.debug.info_message("orig_signal-length: " + str(len(signal)))
      self.debug.info_message("resampled_signal-length: " + str(len(resampled_signal)))

      return resampled_signal, num_samples_new_signal - len(signal)
    except:
      self.debug.error_message("Exception in resampleDopplerShift: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def linearDopplerShiftAutoCorrect(self, signal, target_strongest_frequency, frequency_delta):
    self.debug.info_message("linearDopplerShiftAutoCorrect")
    try:
      fs = self.osmod.sample_rate
      frequency_test = self.osmod.modulation_object.getStrongestFrequency(signal, target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta)
      self.debug.info_message("frequency_test: " + str(frequency_test))

      test_signal = signal
      for _ in range(10):
        adjust_ratio = frequency_test / target_strongest_frequency
        self.debug.info_message("adjust_ratio: " + str(adjust_ratio))
        new_signal, _ = self.osmod.modulation_object.resampleDopplerShift(test_signal, fs / adjust_ratio, fs)
        new_frequency_test = self.osmod.modulation_object.getStrongestFrequency(new_signal, target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta)
        error_value = abs((new_frequency_test - frequency_test) * 1000)
        self.debug.info_message("error_value: " + str(error_value))
        if error_value < 5:
        #if error_value < 3:
          break
        else:
          frequency_test = new_frequency_test
          test_signal = new_signal
          self.debug.info_message("frequency_test: " + str(frequency_test))

      return new_signal
    except:
      self.debug.error_message("Exception in linearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def nonLinearDopplerShiftAutoCorrect(self, orig_signal, target_strongest_frequency, frequency_delta):
    self.debug.info_message("nonLinearDopplerShiftAutoCorrect")
    try:
      fs = self.osmod.sample_rate
      upconvert_factor = 10
      self.debug.info_message("upconvert_factor: " + str(upconvert_factor))

      """ upconvert the signal for processing at higher resolution"""
      if upconvert_factor != 1:
        orig_signal = orig_signal.astype(np.complex128)
        orig_signal = scipy_signal.resample(orig_signal, len(orig_signal) * upconvert_factor)
        target_strongest_frequency = target_strongest_frequency / upconvert_factor
        frequency_delta = frequency_delta / upconvert_factor

      orig_signal_len = len(orig_signal)


      for loop_count in range(0,2):

        new_signal = np.zeros((len(orig_signal)+1000,), dtype = orig_signal.dtype)
        sample_factor = 1 / (2**loop_count) 
        rolling_offset = 0

        self.debug.info_message("dtype: " + str(orig_signal.dtype))

        delta_increments = int(orig_signal_len * sample_factor)
        self.debug.info_message("delta_increments: " + str(delta_increments))

        loop_max = len(orig_signal)
        for signal_index in range(0, loop_max, delta_increments):

          signal = orig_signal[signal_index:signal_index + delta_increments]
          if len(signal) == delta_increments:
            frequency_test = self.osmod.modulation_object.getStrongestFrequency2(np.append(signal, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
            self.debug.info_message("frequency_test: " + str(frequency_test))
            error_value = abs((target_strongest_frequency - frequency_test) * 1000)
            self.debug.info_message("error_value: " + str(error_value))

            test_signal = signal
            adjust_ratio = frequency_test / target_strongest_frequency
            self.debug.info_message("adjust_ratio: " + str(adjust_ratio))
            if error_value > 0.5:
              new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShift(test_signal, fs / adjust_ratio, fs)

              #new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShift(test_signal, target_strongest_frequency, frequency_test)
            else:
              new_signal_part = test_signal
              diff = 0
            #new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShift(test_signal, frequency_test, target_strongest_frequency)

            self.debug.info_message("new_signal_part length: " + str(len(new_signal_part)))
            self.debug.info_message("diff: " + str(diff))
            if delta_increments + diff == len(new_signal_part) and signal_index + delta_increments + rolling_offset + diff < len(new_signal) :
              new_signal[signal_index + rolling_offset:signal_index + delta_increments + rolling_offset + diff] = new_signal_part
              max_signal_len = signal_index + delta_increments + rolling_offset + diff
            else:
              self.debug.info_message("broadcast lengths are different")
              self.debug.info_message("delta_increments + diff: " + str(delta_increments + diff))
              self.debug.info_message("signal_index + delta_increments + rolling_offset + diff: " + str(signal_index + delta_increments + rolling_offset + diff))
              self.debug.info_message("len(new_signal): " + str(len(new_signal)))


            rolling_offset = rolling_offset + diff

          else:
            self.debug.info_message("End of signal data.")

        new_signal = new_signal[0:max_signal_len]

        orig_signal = new_signal
        orig_signal_len = len(orig_signal)


      """ downconvert signal back to original sample rate"""
      if upconvert_factor != 1:
        new_signal = scipy_signal.resample(new_signal, int(orig_signal_len / upconvert_factor))
        new_signal = new_signal.astype(np.float64)
        #orig_signal = np.zeros((len(orig_signal),), dtype = np.float64)


      self.debug.info_message("completed nonLinearDopplerShiftAutoCorrect")

      return new_signal
    except:
      self.debug.error_message("Exception in nonLinearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def calcAlignmentMetrics(self, signal, frequencies, padding_factor, sample_rate):
    self.debug.info_message("calcAlignmentMetrics")
    try:
      self.debug.info_message("CALCULATING ALIGNMENT METRICS")

      f1 = frequencies[0]
      f2 = frequencies[1]
      df1 = self.resolveFrequencyToNDP(signal, padding_factor, f1, 0.5, 8, 0, 10, 1000, sample_rate)
      df2 = self.resolveFrequencyToNDP(signal, padding_factor, f2, 0.5, 8, 0, 10, 1000, sample_rate)

      metric_1 = abs(f1 - df1)
      metric_2 = abs(f2 - df2)
      metric_3 = abs((df2 - df1) - (f2 - f1))

      self.debug.info_message("metric_1: " + str(metric_1))
      self.debug.info_message("metric_2: " + str(metric_2))
      self.debug.info_message("metric_3: " + str(metric_3))

      self.osmod.form_gui.window['text_decode_accuracy_metric'].update("LDS: " + str(metric_3))
      self.osmod.form_gui.window['text_decode_accuracy_metric_2'].update("FS: " + str((metric_1 + metric_2 )/2.0))

      return metric_1, metric_2, metric_3

    except:
      self.debug.error_message("Exception in calcAlignmentMetrics: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
          


  def getDopplerRatioFromDoubleCarrier(self, signal, target_frequencies, padding_factor, upconvert_factor, frequency_delta, czt_num_points, num_dp, sample_rate):
    self.debug.info_message("getDopplerRatioFromDoubleCarrier")
    try:
      f1 = target_frequencies[0]
      f2 = target_frequencies[1]
      self.debug.info_message("f1: " + str(f1 * upconvert_factor))
      self.debug.info_message("f2: " + str(f2 * upconvert_factor))
      #df1 = self.resolveFrequencyToNDP(signal, padding_factor, f1, 5/upconvert_factor, 8, 0, 10)
      #df2 = self.resolveFrequencyToNDP(signal, padding_factor, f2, 5/upconvert_factor, 8, 0, 10)
      #df1 = self.resolveFrequencyToNDP(signal, padding_factor, f1, 10/upconvert_factor, 8, 0, 10)
      #df2 = self.resolveFrequencyToNDP(signal, padding_factor, f2, 10/upconvert_factor, 8, 0, 10)

      #alpha = 0.97
      alpha = self.osmod.getBiasFilterValue() # 1.2
      emphasized_signal = lfilter([1, -alpha], [1], signal)

      #observed_f1 = self.resolveFrequencyToNDP(signal, padding_factor, f1, frequency_delta, num_dp, 0, num_dp + 2, czt_num_points, sample_rate)
      #observed_f2 = self.resolveFrequencyToNDP(signal, padding_factor, f2, frequency_delta, num_dp, 0, num_dp + 2, czt_num_points, sample_rate)
      observed_f1 = self.resolveFrequencyToNDP(emphasized_signal, padding_factor, f1, frequency_delta, num_dp, 0, num_dp + 2, czt_num_points, sample_rate)
      observed_f2 = self.resolveFrequencyToNDP(emphasized_signal, padding_factor, f2, frequency_delta, num_dp, 0, num_dp + 2, czt_num_points, sample_rate)

      self.debug.info_message("observed_f1: " + str(observed_f1 * upconvert_factor))
      self.debug.info_message("observed_f2: " + str(observed_f2 * upconvert_factor))
     
      #term_1 = (f2 / f1)
      #term_2 = (observed_f2 / observed_f1)
      term_1 = (f2 - f1)
      term_2 = (observed_f2 - observed_f1)
      self.debug.info_message("term_1: " + str(term_1))
      self.debug.info_message("term_2: " + str(term_2))
      #ratio = term_1 / term_2
      #ratio = 1.0
      lds_ratio = term_2 / term_1
      #inv_ds = ((observed_f2 / f2) - (observed_f1 / f1))
      self.debug.info_message("lds_ratio: " + str(lds_ratio))
      #ds = 1 / inv_ds
      #self.debug.info_message("ds: " + str(ds))
      #self.debug.info_message("delta shift: " + str(ds))

      """ now calculate the frequency shift required on both carriers """
      """ first calculate the new doppler shifted frequencies """
      observed_f1_ds = observed_f1 / lds_ratio
      observed_f2_ds = observed_f2 / lds_ratio

      f1_shift_amount = f1 - observed_f1_ds
      f2_shift_amount = f2 - observed_f2_ds
 
      averaged_doppler_shift_amount = (f1_shift_amount + f2_shift_amount) /2
      self.osmod.form_gui.window['text_decode_lds_correction_hz'].update("LDS: " + str(averaged_doppler_shift_amount) )
      if abs(averaged_doppler_shift_amount) > 5:
        self.osmod.form_gui.window['text_decode_lds_correction_hz'].update(text_color = 'red') 
      else:
        self.osmod.form_gui.window['text_decode_lds_correction_hz'].update(text_color = 'light green') 

      frequency_correction_f1 = f1 - (observed_f1 - f1_shift_amount)
      frequency_correction_f2 = f2 - (observed_f2 - f2_shift_amount)

      averaged_frequency_shift_amount = (frequency_correction_f1 + frequency_correction_f2) /2
      self.osmod.form_gui.window['text_decode_fs_correction_hz'].update("FS: " + str(averaged_frequency_shift_amount) )
      if abs(averaged_frequency_shift_amount) > 5:
        self.osmod.form_gui.window['text_decode_fs_correction_hz'].update(text_color = 'red') 
      else:
        self.osmod.form_gui.window['text_decode_fs_correction_hz'].update(text_color = 'light green') 

      frequency_shift_value = ((f2 - observed_f2_ds ) + (f1 - observed_f1_ds)) / 2.0 
      #frequency_shift_value = ((observed_f2_ds - f2 ) + (observed_f1_ds - f1)) / 2.0 

      fs_only_value = f1 - observed_f1
      #fs_only_value = observed_f1 - f1


      return lds_ratio, frequency_shift_value, fs_only_value
    except:
      self.debug.error_message("Exception in getDopplerRatioFromDoubleCarrier: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getDopplerRatioFromSingleCarrier(self, signal, target_frequencies, padding_factor, upconvert_factor, frequency_delta, czt_num_points, num_dp, sample_rate):
    self.debug.info_message("getDopplerRatioFromSingleCarrier")
    try:
      f1 = target_frequencies[0]

      frequency_test = self.resolveFrequencyToNDP(signal, padding_factor, f1, 5, num_dp, 0, num_dp+2, czt_num_points, sample_rate)
      adjust_ratio = frequency_test / f1
      self.debug.info_message("adjust_ratio: " + str(adjust_ratio))

      return adjust_ratio, 0.0, 0.0
    except:
      self.debug.error_message("Exception in getDopplerRatioFromSingleCarrier: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def nonLinearDopplerShiftAutoCorrectAlgo3(self, ratio_method, orig_signal, target_strongest_frequencies, doppler_params, sample_rate):
    self.debug.info_message("nonLinearDopplerShiftAutoCorrectAlgo3")
    try:
      #fs = self.osmod.sample_rate
      fs = sample_rate # self.osmod.getRxSampleRate()
      error_target = doppler_params[0]  #0.8 #0.7
      padding_factor = doppler_params[1] # 4
      upconvert_factor = doppler_params[2]  #25  # this is the best but processing intensive
      frequency_delta = self.osmod.getAperture() #doppler_params[3] #0.5 
      source_offset_max = doppler_params[4] # 20
      czt_num_points = doppler_params[5] # 50000
      num_dp = doppler_params[6]


      doppler_override_checked = self.osmod.form_gui.window['cb_override_doppler_params'].get()
      if doppler_override_checked:
        upconvert_factor = int(self.osmod.form_gui.window['in_doppler_upconvert'].get())
        frequency_delta  = float(self.osmod.form_gui.window['in_doppler_delta'].get())
        num_dp           = int(self.osmod.form_gui.window['in_doppler_numdp'].get())
        czt_num_points   = int(self.osmod.form_gui.window['in_doppler_czt'].get())
        padding_factor   = int(self.osmod.form_gui.window['in_doppler_padding'].get())

      """ upconvert the signal for processing at higher resolution"""
      use_hifi_tx = self.osmod.form_gui.window['cb_enable_hifi_output_sampling'].get()
      use_hifi_rx = self.osmod.form_gui.window['cb_enable_hifi_input_sampling'].get()

      #if use_hifi_rx:
      if use_hifi_tx or use_hifi_rx:
        self.debug.info_message("processing at 48k")
        upconvert_factor = 1
        f1 = target_strongest_frequencies[0]
        f2 = target_strongest_frequencies[1]
      elif upconvert_factor != 1:
        orig_signal = orig_signal.astype(np.complex128)
        orig_signal = scipy_signal.resample(orig_signal, len(orig_signal) * upconvert_factor)
        f1 = target_strongest_frequencies[0] / upconvert_factor
        f2 = target_strongest_frequencies[1] / upconvert_factor
        frequency_delta = frequency_delta / upconvert_factor
      else:
        f1 = target_strongest_frequencies[0]
        f2 = target_strongest_frequencies[1]

      self.debug.info_message("upconvert_factor: " + str(upconvert_factor))

      orig_signal_len = len(orig_signal)
        
      new_signal = np.zeros((orig_signal_len+1000000,), dtype = orig_signal.dtype)

      self.debug.info_message("dtype: " + str(orig_signal.dtype))

      delta_increments = int(orig_signal_len)
      self.debug.info_message("delta_increments: " + str(delta_increments))

      signal_index = 0
      signal = orig_signal[signal_index:signal_index + delta_increments]

      adjust_ratio, calculated_frequency_shift_value, fs_only = ratio_method(signal, [f1, f2], padding_factor, upconvert_factor, frequency_delta, czt_num_points, num_dp, sample_rate)

      self.debug.info_message("calculated_frequency_shift_value: " + str(calculated_frequency_shift_value))


      lds_and_fs_auto_correct = self.osmod.form_gui.window['cb_enable_block_level_resample_auto_correct'].get()
      fs_ony_auto_correct = self.osmod.form_gui.window['cb_enable_auto_correct_frequency_only'].get()
      if lds_and_fs_auto_correct:
      #if True:
        new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShift(signal, fs / adjust_ratio, fs)
        #new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShiftFFT(signal, adjust_ratio)

        new_signal[0:delta_increments + diff] = new_signal_part
        max_signal_len = delta_increments + diff
        new_signal = new_signal[0:max_signal_len]

        """ adjust for frequency """
        #df1 = self.resolveFrequencyToNDP(new_signal, padding_factor, f1 / adjust_ratio, frequency_delta, num_dp, 0, num_dp+2, czt_num_points, sample_rate)
        #df2 = self.resolveFrequencyToNDP(new_signal, padding_factor, f2 / adjust_ratio, frequency_delta, num_dp, 0, num_dp+2, czt_num_points, sample_rate)
        #frequency_shift_value = ((f2 - df2 ) + (f1 - df1)) / 2.0   #f1 - actual_low_freq
        #self.debug.info_message("FFT frequency_shift_value: " + str(frequency_shift_value))
        #new_signal = self.osmod.modulation_object.shiftAllFrequenciesExp(new_signal, frequency_shift_value, sample_rate)
        new_signal = self.osmod.modulation_object.shiftAllFrequenciesExp(new_signal, calculated_frequency_shift_value, sample_rate)
      elif fs_ony_auto_correct:
        #new_signal = self.osmod.modulation_object.shiftAllFrequenciesExp(signal, fs_only, sample_rate)
        #signal_padded = np.append(signal, np.zeros((len(signal) * 2.0,), dtype = signal.dtype)).copy()
        #new_signal = self.osmod.modulation_object.shiftAllFrequencies(signal_padded, fs_only, sample_rate)

        new_signal = self.osmod.modulation_object.shiftAllFrequencies(signal, fs_only, sample_rate)


      #new_signal = self.osmod.modulation_object.shiftAllFrequencies(new_signal, frequency_shift_value, sample_rate)

      new_signal_len = len(new_signal)

      if upconvert_factor != 1:
        new_signal = scipy_signal.resample(new_signal, int(new_signal_len / upconvert_factor))
        new_signal = new_signal.astype(np.float64)

      self.debug.info_message("completed nonLinearDopplerShiftAutoCorrect")

      return new_signal


    except:
      self.debug.error_message("Exception in nonLinearDopplerShiftAutoCorrectAlgo3: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def resolveFrequencyToNDP(self, signal, padding_factor, guess, delta, dp, iter_count, max_iter, czt_num_points, sample_rate):
    self.debug.info_message("resolveFrequencyToNDP")
    try:
      max_resolution = False
      #max_resolution = True

      #accuracy_jump = 0.001
      accuracy_jump = 0.01
      #accuracy_jump = 0.1

      self.debug.info_message("guess: " + str(guess))
      self.debug.info_message("iter_count: " + str(iter_count))
      #padding_factor = 0

      if iter_count == 0:
        signal = np.append(signal, np.zeros((len(signal) * padding_factor,), dtype = signal.dtype)).copy()
      #frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(audio_array, np.zeros((len(audio_array)*1,), dtype = audio_array.dtype)).copy(), guess - delta, guess + delta, self.osmod.sample_rate, 10000)
      #frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(signal, guess - delta, guess + delta, self.osmod.sample_rate, 1000)
      frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(signal, guess - delta, guess + delta, sample_rate, czt_num_points)

      accuracy = abs(frequency_test - guess)

      if (accuracy > 1/(10**dp)  and iter_count <= max_iter) or (max_resolution == True and iter_count < 15):
        frequency_test = self.resolveFrequencyToNDP(signal, padding_factor, frequency_test, delta * accuracy_jump, dp, iter_count+1, max_iter, czt_num_points, sample_rate)
        return frequency_test
      else:
        if iter_count > max_iter:
          self.debug.info_message("FAIL. accuracy is : " + str(accuracy))
        else:
          self.debug.info_message("SUCCESS! accuracy is : " + str(accuracy))
        return frequency_test

    except:
      self.debug.error_message("Exception in resolveFrequencyToNDP: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ doppler_params = [error_target, padding_factor, upconvert_factor, frequency_delta, loop_count, source_offset_max, czt_num_points]"""
  """ doppler_params = [0.0075, 4, 25, 0.50, 20, 50000]"""
  def nonLinearDopplerShiftAutoCorrectParams(self, orig_signal, target_strongest_frequency, doppler_params):
    self.debug.info_message("nonLinearDopplerShiftAutoCorrect")
    try:
      # error target should ideally be around 0.004 to 0.007
      #error_target = 0.2 #0.8 #0.7
      #error_target = 0.02 #0.8 #0.7
      #error_target = 0.01 #0.8 #0.7
      #error_target = 0.011 #0.8 #0.7
      #error_target = 0.0075 #0.8 #0.7
      error_target = doppler_params[0]  #0.8 #0.7
      #error_target = 0.004 #0.8 #0.7
      #error_target = 0.7

      czt_num_points = doppler_params[5] # 50000

      #padding_factor = 2
      padding_factor = doppler_params[1] # 4

      fs = self.osmod.sample_rate
      #upconvert_factor = 10
      upconvert_factor = doppler_params[2] #25  # this is the best but processing intensive
      #upconvert_factor = 35  # this is the best but processing intensive

      #frequency_delta = 2 / upconvert_factor
      #frequency_delta = 1.5 / upconvert_factor
      frequency_delta = doppler_params[3] #0.5
 
      source_offset_max = doppler_params[4] # 20

      self.debug.info_message("upconvert_factor: " + str(upconvert_factor))

      """ upconvert the signal for processing at higher resolution"""
      if upconvert_factor != 1:
        orig_signal = orig_signal.astype(np.complex128)
        orig_signal = scipy_signal.resample(orig_signal, len(orig_signal) * upconvert_factor)
        target_strongest_frequency = target_strongest_frequency / upconvert_factor
        frequency_delta = frequency_delta / upconvert_factor

      orig_signal_len = len(orig_signal)

      #for loop_count in range(0,2):
      if True:
        loop_count = 0
        
        #delta_increments = 928005

        #new_signal = np.zeros((len(orig_signal)+1000,), dtype = orig_signal.dtype)
        new_signal = np.zeros((orig_signal_len+1000,), dtype = orig_signal.dtype)
        sample_factor = 1 / (2**loop_count) 
        rolling_offset = 0
        rolling_source_offset = 0


        self.debug.info_message("dtype: " + str(orig_signal.dtype))

        delta_increments = int(orig_signal_len * sample_factor)
        #delta_increments = int(((orig_signal_len * sample_factor) // 400) * 400)

        #delta_increments = 928005
        #delta_increments = 748000
        self.debug.info_message("delta_increments: " + str(delta_increments))

        loop_max = len(orig_signal)
        for signal_index in range(0, loop_max, delta_increments):

          #signal = orig_signal[signal_index - rolling_source_offset - rolling_offset:signal_index - rolling_source_offset + delta_increments - rolling_offset]
          signal = orig_signal[signal_index - rolling_source_offset:signal_index - rolling_source_offset + delta_increments]
          if len(signal) == delta_increments:
            #frequency_test = self.osmod.modulation_object.getStrongestFrequency2(np.append(signal, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
            frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(signal, np.zeros((len(orig_signal)*padding_factor,), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate, czt_num_points)
            #frequency_test = self.resolveFrequencyToNDP(signal, padding_factor, target_strongest_frequency, 5, 8, 0, 10)

            error_value = abs((target_strongest_frequency - frequency_test) * 1000)
            self.debug.info_message("error_value: " + str(error_value))
            self.debug.info_message("frequency_test: " + str(frequency_test))

            if error_value > error_target:
              test_signal = signal
              for source_offset in range(source_offset_max):
                #test_signal = orig_signal[signal_index - rolling_source_offset:signal_index - rolling_source_offset - source_offset + delta_increments]
                #frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(test_signal, np.zeros((len(orig_signal)*padding_factor,), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate, czt_num_points)

                adjust_ratio = frequency_test / target_strongest_frequency
                self.debug.info_message("adjust_ratio: " + str(adjust_ratio))
                new_signal_part, diff = self.osmod.modulation_object.resampleDopplerShift(test_signal, fs / adjust_ratio, fs)
                #new_frequency_test = self.osmod.modulation_object.getStrongestFrequency2(np.append(new_signal_part, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                #new_frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(new_signal_part, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                #new_frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(new_signal_part, target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                new_frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(new_signal_part, np.zeros((len(orig_signal)*padding_factor,), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate, czt_num_points)
                #new_frequency_test = self.resolveFrequencyToNDP(new_signal_part, padding_factor, target_strongest_frequency, 5, 8, 0, 10)
                error_value = abs((target_strongest_frequency - new_frequency_test) * 1000)
                self.debug.info_message("new error_value: " + str(error_value))
                #if error_value < 0.7:
                if error_value <= error_target:
                  diff = len(new_signal_part) - delta_increments # len(signal) #delta_increments
                  rolling_source_offset = rolling_source_offset + source_offset
                  source_offset = 0
                  #new_signal_part = signal
                  self.debug.info_message("Success 1!" )
                  break
                else:
                  signal = orig_signal[signal_index - rolling_source_offset:signal_index - rolling_source_offset - source_offset + delta_increments]
                  #frequency_test = self.osmod.modulation_object.getStrongestFrequency2(np.append(signal, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                  #frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(signal, np.zeros((len(orig_signal*10),), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                  #frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(signal, target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate)
                  frequency_test = self.osmod.modulation_object.getStrongestFrequencyCZT(np.append(signal, np.zeros((len(orig_signal)*padding_factor,), dtype = orig_signal.dtype)).copy(), target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta, self.osmod.sample_rate, czt_num_points)
                  #frequency_test = self.resolveFrequencyToNDP(signal, padding_factor, target_strongest_frequency, 5, 8, 0, 10)


                  test_signal = signal
                  #frequency_test = new_frequency_test
                  #test_signal = new_signal_part
            else:
              new_signal_part = signal
              #rolling_offset = rolling_offset + diff
              diff = 0
              self.debug.info_message("Success 2!" )

            self.debug.info_message("new_signal_part length: " + str(len(new_signal_part)))
            self.debug.info_message("diff: " + str(diff))
            self.debug.info_message("rolling_offset: " + str(rolling_offset))
            self.debug.info_message("signal_index: " + str(signal_index))
            self.debug.info_message("delta_increments: " + str(delta_increments))
            #if delta_increments + diff == len(new_signal_part) and signal_index + delta_increments + rolling_offset + diff < len(new_signal) :
            new_signal[signal_index + rolling_offset:signal_index + delta_increments + rolling_offset + diff] = new_signal_part
            max_signal_len = signal_index + delta_increments + rolling_offset + diff
            #else:
            #  self.debug.info_message("broadcast lengths are different")
            #  self.debug.info_message("delta_increments + diff: " + str(delta_increments + diff))
            #  self.debug.info_message("signal_index + delta_increments + rolling_offset + diff: " + str(signal_index + delta_increments + rolling_offset + diff))
            #  self.debug.info_message("len(new_signal): " + str(len(new_signal)))

            rolling_offset = rolling_offset + diff
            #diff = 0

          else:
            self.debug.info_message("End of signal data.")

        new_signal = new_signal[0:max_signal_len]

        orig_signal = new_signal
        #orig_signal_len = len(new_signal) #len(orig_signal)
        orig_signal_len = len(orig_signal)

      """ downconvert signal back to original sample rate"""
      if upconvert_factor != 1:
        new_signal = scipy_signal.resample(new_signal, int(orig_signal_len / upconvert_factor))
        new_signal = new_signal.astype(np.float64)

      self.debug.info_message("completed nonLinearDopplerShiftAutoCorrect")

      return new_signal
    except:
      self.debug.error_message("Exception in nonLinearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def adjustFrequencyShiftAndDopplerShiftSR(self, noise_free_signal, values, center_frequency, sample_rate):
    self.debug.info_message("adjustFrequencyShiftAndDopplerShift")
    return self.adjustFrequencyShiftAndDopplerShiftCommon(noise_free_signal, values, center_frequency, sample_rate)


  def adjustFrequencyShiftAndDopplerShift(self, noise_free_signal, values, center_frequency):
    self.debug.info_message("adjustFrequencyShiftAndDopplerShift")
    return self.adjustFrequencyShiftAndDopplerShiftCommon(noise_free_signal, values, center_frequency, self.osmod.sample_rate)


  def adjustFrequencyShiftAndDopplerShiftCommon(self, noise_free_signal, values, center_frequency, sample_rate):

    self.debug.info_message("adjustFrequencyShiftAndDopplerShift")
    try:

      """ TEST CODE shift frequencies"""
      self.debug.info_message(" TEST CODE FOR DOPPLER SHIFT AND FREQUENCY SHIFT")

      """ Apply manual frequency shift """
      enable_fine_tune_frequency = self.osmod.form_gui.window['cb_enable_fine_tune_frequency'].get()
      if enable_fine_tune_frequency:
        frequency_shift_value = values['slider_freq_fine_tune']
        noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, frequency_shift_value, sample_rate)
        #noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, -1 * frequency_shift_value)
      enable_fine_tune_resample = self.osmod.form_gui.window['cb_enable_fine_tune_resample'].get()

      """ Auto correct frequency shift """
      enable_resample_auto_correct = self.osmod.form_gui.window['cb_enable_resample_auto_correct'].get()
      enable_frequency_shift_auto_correct = self.osmod.form_gui.window['cb_enable_frequency_shift_auto_correct'].get()
      if enable_frequency_shift_auto_correct and enable_resample_auto_correct == False:
        target_freq_low = center_frequency + self.osmod.getTxResampleParams()[1]
        actual_low_freq = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, target_freq_low - 5, target_freq_low + 5)
        frequency_shift_value = target_freq_low - actual_low_freq
        noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, frequency_shift_value, sample_rate)

      """ Apply manual doppler shift """
      if enable_fine_tune_resample:
        doppler_shift_value = values['slider_resample_fine_tune']
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000, 8000 * doppler_shift_value)
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 800000, 800002). # need to be this accurate for correct decode
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000, 8100)
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8100, 8000)
        #noise_free_signal, _ = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, self.osmod.sample_rate, self.osmod.sample_rate * doppler_shift_value)
        noise_free_signal, _ = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, sample_rate, sample_rate * doppler_shift_value)


      """ auto correct non-linear doppler shift"""
      enable_nonllinear_doppler_auto_correct = self.osmod.form_gui.window['cb_enable_block_level_resample_auto_correct'].get()
      fs_ony_auto_correct = self.osmod.form_gui.window['cb_enable_auto_correct_frequency_only'].get()

      if enable_nonllinear_doppler_auto_correct or fs_ony_auto_correct:
        if self.osmod.getTxResampleParams()[0] == ocn.RESAMPLE_AVAILABLE:
          self.debug.info_message("RESAMPLE_AVAILABLE ")
          self.debug.info_message("correcting for non-linear doppler shift only")
          target_freq_low = center_frequency + self.osmod.getTxResampleParams()[1]
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  #this for 12800 mode
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectParams(noise_free_signal, target_freq_low, [0.0075, 4, 25, 0.50, 20, 50000])  #this for 12800 mode

          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectParams(noise_free_signal, target_freq_low, [0.004, 6, 35, 0.25, 2, 50000])  #this for 12800 mode

          f1 = center_frequency + self.osmod.getTxResampleParams()[1]
          f2 = center_frequency + self.osmod.getTxResampleParams()[2]
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromSingleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 10, 0.25, 2, 5000])  #this for 12800 mode
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromDoubleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 10, 6, 2, 5000, 8])  #this for 12800 mode
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromDoubleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 1, 6, 2, 1000, 10])  #this for 12800 mode
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromDoubleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 12, 0.25, 2, 4000, 10])  #this for 12800 mode
          noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromDoubleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 4, 3, 2, 4000, 4], sample_rate)  #this for 12800 mode
          #noise_free_signal = self.osmod.modulation_object.nonLinearDopplerShiftAutoCorrectAlgo3(self.getDopplerRatioFromDoubleCarrier, noise_free_signal, [f1, f2], [0.002, 1, 10, 0.25, 2, 5000])  #this for 12800 mode


          #self.calcAlignmentMetrics(noise_free_signal, [center_frequency + self.osmod.getTxResampleParams()[1], center_frequency + self.osmod.getTxResampleParams()[2]], 1, sample_rate)

      """ automatic adjust for linear doppler shift """
      enable_resample_auto_correct = self.osmod.form_gui.window['cb_enable_resample_auto_correct'].get()
      if enable_resample_auto_correct:
        if self.osmod.getTxResampleParams()[0] == ocn.RESAMPLE_AVAILABLE:
          self.debug.info_message("RESAMPLE_AVAILABLE ")
          if enable_frequency_shift_auto_correct == False:
            self.debug.info_message("correcting for doppler shift only")
            target_freq_low = center_frequency + self.osmod.getTxResampleParams()[1]
            #noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, 1382.5, 5)   #this for 6400 mode
            #noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, 1381.875, 5)  #this for 12800 mode
            noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  #this for 12800 mode
          else:
            self.debug.info_message("correcting for frequency shift and doppler shift ")

            if self.osmod.getTxResampleParams()[3] != 0:

              """ first doppler shift correct """
              target_freq_low = center_frequency + self.osmod.getTxResampleParams()[1]
              noise_free_signal_temp = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  

              """ second calc residual frequency shift component"""
              frequency_test_lower  = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal_temp, 1380, 1385)
              frequency_test_higher = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal_temp, 1414, 1424)
              difference = ((frequency_test_higher - frequency_test_lower) - (self.osmod.getTxResampleParams()[2] - self.osmod.getTxResampleParams()[1])) * 10000
              self.debug.info_message("difference: " + str(difference))

              partial_result = difference 
              self.debug.info_message("partial_result: " + str(partial_result))
              calculated_freq_offset = partial_result / self.osmod.getTxResampleParams()[3] 
              self.debug.info_message("calculated_freq_offset: " + str(calculated_freq_offset))

              """ third apply frequency shift component to original signal """
              noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, calculated_freq_offset, sample_rate)

              """ fourth reapply auto doppler shift correction """
              target_freq_low = center_frequency + self.osmod.getTxResampleParams()[1]
              noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  
            else:
              self.debug.info_message("RESAMPLE_UNAVAILABLE - getTxResampleParams()[3] == 0")

        else:
          self.debug.info_message("RESAMPLE_UNAVAILABLE ")

      return noise_free_signal

    except:
      self.debug.error_message("Exception in linearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def appendTableRow(self, message, sender_callsign):
    self.debug.info_message("appendTableRow")
    try:
      center_frequency = self.osmod.getCenterFrequency()
      timestamp = datetime.utcnow().strftime('%Y/%m/%d %H:%M')

      if self.osmod.form_gui.window['cb_use_prod_modes'].get() == True:
        mode = self.osmod.form_gui.window['combo_main_modem_prod_modes'].get()
      else:
        mode = self.osmod.form_gui.window['combo_main_modem_modes'].get()

      data_key = str(center_frequency) + "_" + str(mode)
      self.osmod.dict_rcvd[data_key] = [timestamp, message]
      self.debug.info_message("self.osmod.dict_rcvd: " + str(self.osmod.dict_rcvd))

      data_table = []
      for key, value in self.osmod.dict_rcvd.items():
        data_row = key.split('_') + value
        self.debug.info_message("data_row: " + str(data_row))
        data_table.append(data_row)

      self.debug.info_message("data_table: " + str(data_table))

      self.osmod.form_gui.window['tbl_frequency_mode_message'].update(data_table)
      self.osmod.received_data_table = data_table


      """ create callsign / location table """
      if sender_callsign != '':
        locator = ''
        data_key = sender_callsign
        self.osmod.dict_rcvd_callsign[data_key] = [locator, timestamp]
        self.debug.info_message("self.osmod.dict_rcvd_callsign: " + str(self.osmod.dict_rcvd_callsign))

        data_table = []
        for key, value in self.osmod.dict_rcvd_callsign.items():
          data_row = [str(key)] + value
          self.debug.info_message("data_row: " + str(data_row))
          data_table.append(data_row)
        self.debug.info_message("data_table: " + str(data_table))

        self.osmod.form_gui.window['tbl_callsign_locator'].update(data_table)
        self.osmod.received_data_table_callsign = data_table

      if self.osmod.form_gui.window['cb_bypass_display_during_test'].get() == False:
        self.osmod.form_gui.window['ml_txrx_sendtext'].update(value="")

      return timestamp
    except:
      self.debug.error_message("Exception in appendTableRow: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ index min is 99  and phase wave min is 200  for 8k sampled """
  """ index min is 142 and phase wave min is 1242 for 48k sampled """
  def alignTimePointT0(self, signal, sample_rate, symbol_block_size):
    self.debug.info_message("alignTimePointT0")
    try:
      #instantaneous_phase = np.unwrap(np.angle(analytic_signal))
      #instantaneous_frequencu = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * fs
      #self.debug.info_message("instantaneous_amplitude\: "+ str(instantaneous_amplitude))
      #for count in range (10000, 10000 + 1200):
      #  self.debug.info_message("count: "+ str(count) + "   ---   " + str(instantaneous_amplitude[count]))
      #test_signal = gaussian_filter(np.abs(audio_array[start:]), sigma=pulse_start_sigma_template)

      #pulse_train_sigma = 7
      #pulse_train_sigma = 0.77
      #pulse_train_sigma = 8.5
      #pulse_train_sigma = 20.39 
      #pulse_train_sigma = 11.49
      pulse_train_sigma = 23.7
      pulse_length      = int(symbol_block_size / self.osmod.pulses_per_block)

      #override_pulse_train_sigma = self.osmod.form_gui.window['cb_overridepulsetrainsigma'].get()
      #if override_pulse_train_sigma:
      #  pulse_train_sigma = float(self.osmod.form_gui.window['in_pulsetrainsigma'].get())
      

      def getStats(test_signal, local_pulse_length, exact):
        sum_points = []

        for location in range(0, local_pulse_length): 
          sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % local_pulse_length == location])
          sum_points.append(sum_at_location)

        if exact == True:
          index_min = np.where(sum_points == np.min(sum_points))
          index_max = np.where(sum_points == np.max(sum_points))
        else:
          index_min = np.where(sum_points <= 1.001 * np.min(sum_points))
          index_max = np.where(sum_points >= 0.999 * np.max(sum_points))

        #self.debug.info_message("min sum_points: " + str(np.min(sum_points)))
        #self.debug.info_message("max sum_points: " + str(np.max(sum_points)))

        self.debug.info_message("index_min: " + str(index_min[0]))
        self.debug.info_message("index max: " + str(index_max[0]))

        #self.debug.info_message("all index max: " + str(index_max))

        self.debug.info_message("mean index min: " + str(np.mean(index_min)))
        self.debug.info_message("mean index max: " + str(np.mean(index_max)))

        for count in range(0, len(index_min)):
          averages = splitRanges(index_min[count])
          self.debug.info_message("averages: " + str(averages))

        #splitRanges(index_min[0])
        #splitRanges(index_max[0])
        #return int(round(averages[0]))
        return int(round(averages))

        #self.debug.info_message("diff: " + str(index_max - index_min))

      def splitRanges(data):
        averages_array = np.array([])
        jumps = np.diff(data) > 1
        jump_indices = np.where(jumps)[0] + 1
        ranges = np.split(data, jump_indices)
        self.debug.info_message("ranges: " + str(ranges))

        max_range_len = 0
        max_range_index = -1
        for i in range(len(ranges)):
          self.debug.info_message("test range: " + str(ranges[i]))
          if len(ranges[i]) > max_range_len:
            max_range_len = len(ranges[i])
            max_range_index = i

        self.debug.info_message("max_range_len: " + str(max_range_len))
        self.debug.info_message("max_range_index: " + str(max_range_index))


        current_index = 0
        for i, r in enumerate(ranges):
          end_index = current_index + len(r) - 1
          average = np.mean(r)
          averages_array = np.append(averages_array, average)
          self.debug.info_message("average: " + str(average))
          current_index = end_index + 1
        #self.debug.info_message("averages 1: " + str(averages_array))

        return averages_array[max_range_index]
        #return averages_array

      def identifyModeFromSignal(test_signal, peak_location):
        self.debug.info_message("identifyModeFromSignal()")

        #pulse_start_index = peak_location - 50

        #sigma_template = 4
        #test_signal = gaussian_filter(np.abs(audio_array[pulse_start_index:]), sigma=sigma_template)
        #location_points = self.findPeaksOverSample(3)

        dict_modes = {}


        #length = int(((len(test_signal)/4) // 128) * 128)
        #segment_length = 128 * pulse_length * 4
        #test_signal = test_signal[0:segment_length] + test_signal[segment_length:segment_length*2] + test_signal[segment_length*2:segment_length*3]

        #"""
        self.debug.info_message("segmenting signal")
        segment_length = 128 * pulse_length * 2
        for loop_count in range(0,8):
          if loop_count == 0:
            new_signal = test_signal[0:segment_length]
          else:
            if segment_length * (loop_count+1) < len(test_signal):
              new_signal = new_signal + test_signal[segment_length * loop_count:segment_length * (loop_count+1)]
        test_signal = new_signal
        #"""

        #test_signal = test_signal[0:int(len(test_signal)/4)]


        #test_signal = test_signal[0:length]
        #test_signal = test_signal[0:int(len(test_signal)/8)]
        #test_signal = test_signal[0:int(len(test_signal)/2)]

        self.debug.info_message("low pass filter")

        test_signal = self.filter_sharp_cutoff_low_pass(test_signal, self.osmod.center_frequency, 50, self.osmod.getRxSampleRate())
        #test_signal = self.filter_sharp_cutoff_low_pass(test_signal, self.osmod.center_frequency - 25, 50, self.osmod.getRxSampleRate())
        #test_signal = self.filter_sharp_cutoff_high_pass(test_signal, self.osmod.center_frequency, 50, self.osmod.getRxSampleRate())

        points_per_unit = 4
        #points_per_unit = 3

        for j in range(2, 8):
          ppb = 2 ** j
          half_ppb = int(ppb / 2)
          self.debug.info_message("ppb: " + str(ppb))

          #modulo_amount = half_ppb * pulse_length
          modulo_amount = ppb * pulse_length

          # test for 8 pulses per block...
          for i in range(0, pulse_length * ppb, pulse_length): 
            location = peak_location + i

            total = 0

            for k in range(0, half_ppb):
            #for k in range(0, ppb):
              #max_sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % (modulo_amount) == location + (k * pulse_length)])
              max_sum_at_location = np.sum(test_signal[(np.arange(len(test_signal)) % (modulo_amount)) // points_per_unit == (location + (k * pulse_length)) // points_per_unit  ])
              #max_sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % (modulo_amount) == (location + (k * pulse_length)) % modulo_amount ] )
              #min_sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % (modulo_amount) == (location + ((k + half_ppb) * pulse_length)) % modulo_amount ])
              #min_sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % (modulo_amount) == (location + ((k + half_ppb) * pulse_length))])
              #min_sum_at_location = np.sum(test_signal[(np.arange(len(test_signal)) % (modulo_amount)) // 4 == (location + ((k + half_ppb) * pulse_length)) // 4 ])
              #self.debug.info_message("sum_at_location: " + str(sum_at_location))
              total = total + max_sum_at_location
              #total = total + (max_sum_at_location - min_sum_at_location)

            #dict_modes[str(ppb) + ":" + str(i)] = total / half_ppb
            dict_modes[str(ppb) + ":" + str(i)] = total
            self.debug.info_message("total: " + str(total))

        #self.debug.info_message("dict_modes: " + str(dict_modes))

        best_match_max = 0.0
        best_key = ""
        for key, value in dict_modes.items():
          if float(value) > best_match_max:
            best_match_max = float(value)
            best_key = key

        self.debug.info_message("best_key: " + str(best_key))

        mode_name = best_key.split(':')[0]
        best_ppb = int(mode_name)
        block_start_location = int(best_key.split(':')[1])
        self.osmod.form_gui.window['text_input_detected_mode'].update("LB28-" + str(mode_name) + "00-I3")
        self.osmod.form_gui.window['text_input_detected_block_start'].update(str(block_start_location))


        """ locate the first block """        
        # iterate first 20 characters of message
        magnitudes = []
        modulo_amount = best_ppb * pulse_length
        for j in range(0, 16):
          location = peak_location + block_start_location + (j * (pulse_length * best_ppb))
          total = 0

          for k in range(0, int(best_ppb/2)):
            #sum_at_location = np.sum(test_signal[np.arange(len(test_signal))  == location + (k * pulse_length)])
            max_magnitude_at_location = abs(test_signal[location + (k * pulse_length)])
            #min_magnitude_at_location = abs(test_signal[location + ((k + half_ppb) * pulse_length)] )
            total = total + max_magnitude_at_location
            #total = total + max_magnitude_at_location - min_magnitude_at_location
          magnitudes.append(total)

        self.debug.info_message("magnitudes: " + str(magnitudes))


      """ locate the amplified phase wave..."""
      #self.debug.info_message("PHASE WAVE GAUSSIAN... ")
      #getStats(gaussian_filter(np.abs(signal), sigma=7), pulse_length * 3)

      exact_type = True
      #exact_type = False

      #""" process hilbert """
      #self.debug.info_message("HILBERT... ")
      #analytic_signal = hilbert(signal.real)
      #instantaneous_amplitude = np.abs(analytic_signal)
      #getStats(instantaneous_amplitude, pulse_length, exact_type)

      #""" process gaussian """
      #self.debug.info_message("GAUSSIAN 7 real... ")
      #getStats(gaussian_filter(np.abs(signal.real), sigma=7), pulse_length, exact_type)
      
      """ process gaussian """
      self.debug.info_message("GAUSSIAN 7... ")
      gaussian_signal = gaussian_filter(np.abs(signal), sigma=pulse_train_sigma)
      index_min = getStats(gaussian_signal, pulse_length, exact_type)
      #index_min = getStats(np.abs(signal), pulse_length, exact_type)




      #index_min = getStats(gaussian_filter(np.abs(signal), sigma=7.2), pulse_length, exact_type)

      #""" process gaussian """
      #self.debug.info_message("GAUSSIAN 10... ")
      #getStats(gaussian_filter(np.abs(signal), sigma=10), pulse_length, exact_type)

      #""" process gaussian """
      #self.debug.info_message("GAUSSIAN... ")
      #getStats(gaussian_filter(np.abs(signal), sigma=7), pulse_length, exact_trpe)

      """ process gaussian """
      #self.debug.info_message("GAUSSIAN pulse_length / 2 ... ")
      #getStats(gaussian_filter(np.abs(signal), sigma=7), int(pulse_length / 2))

      #self.debug.info_message("GAUSSIAN 150 pulse length... ")
      #getStats(gaussian_filter(np.abs(signal), sigma=7), 150)

      #self.debug.info_message("PHASE WAVE HILBERT... ")
      #analytic_signal = hilbert(signal.real)
      #instantaneous_amplitude = np.abs(analytic_signal)
      #getStats(instantaneous_amplitude, pulse_length * 3)

      identifyModeFromSignal(gaussian_signal, index_min)



      #if sample_rate == 8000:
      #  difference = int((index_min - 26 + pulse_length) % pulse_length)
      #  return signal[difference::]
      #elif  sample_rate == 48000:
      #  difference = int((index_min - 90 + pulse_length) % pulse_length)
      #  return signal[difference::]



      """
      if True:
        pulse_length_phase_wave = pulse_length * 3
        sum_points = []
        range_lo = 0
        range_hi = pulse_length_phase_wave

        for location in range(range_lo, range_hi, 1): 
          sum_at_location = np.sum(test_signal[np.arange(len(test_signal)) % pulse_length_phase_wave == location])
          sum_points.append(sum_at_location)

        phase_wave_index_min = np.where(sum_points == np.min(sum_points))[0]
        self.debug.info_message("phase wave min sum_points: " + str(np.min(sum_points)))
        self.debug.info_message("phase wave index_min: " + str(phase_wave_index_min))
      """


      """
      if sample_rate == 8000:
        difference = int((index_min - 99 + pulse_length) % pulse_length)
        #difference = int((index_min - 200 + pulse_length_phase_wave) % pulse_length_phase_wave)

        #difference = difference + (2 * pulse_length) + 13
        return signal[difference::]
      elif sample_rate == 48000:
        difference = int((index_min - 142 + pulse_length) % pulse_length)
        #difference = int((index_min - 1242 + pulse_length_phase_wave) % pulse_length_phase_wave)

        #difference = difference + (2 * pulse_length) + 0
        return signal[difference::]
      """

    except:
      self.debug.error_message("Exception in alignTimePointT0: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def decodeSignalTypes(self, signal):
    self.debug.info_message("decodeSignalTypes")

    try:
      self.identifyModeFromSignal(signal)


      def findPeaksOverSample(granularity):
        self.debug.info_message("findPeaksOverSample()")
        #nonlocal offset
        #nonlocal index

        self.debug.info_message("granularity: " + str(granularity))

        start = (start_pulse * pulse_length) + (start_block * self.osmod.pulses_per_block * pulse_length)
        num_full_blocks = int((len(audio_array) - start ) // self.osmod.symbol_block_size)
        #self.pulse_train_alignment_struct = {'location_points': [], 'blocks': [], 'current_point_index':0, 'locus': 0, 'diff': 0, 'pulses': [] }

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

          #if aligh_type == ocn.ALIGN_RETAIN_LOCATION:
          #  self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets
          #elif aligh_type == ocn.ALIGN_MOVE_TO_MID:
          #  self.pulse_train_alignment_struct['blocks'][block_count] = self.pulse_train_offsets_mid

        self.debug.info_message("pulse_train_alignment_struct: " + str(self.pulse_train_alignment_struct))

        median_index = self.osmod.demodulation_object.getMode(self.pulse_train_alignment_struct['pulses'])
        self.debug.info_message("median_index: " + str(median_index))

        #return self.pulse_train_alignment_struct['location_points']

    except:
      self.debug.error_message("Exception in decodeSignalTypes: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def findDecodeCandidates(self, data):
    self.debug.info_message("findDecodeCandidates")

    try:
      self.osmod.form_gui.window['text_input_detected_mode'].update("")
      self.osmod.form_gui.window['text_input_detected_block_start'].update("")

      #magnitude_trigger = 0.00001
      magnitude_trigger = 0.0000001
      N = 10
      lo = 250
      high = 2750
      sub_signal = data[0:40000] # first 5 seconds of signal
      fft_output = np.fft.fft(sub_signal)
      frequencies = np.fft.fftfreq(len(sub_signal), 1/self.osmod.getRxSampleRate())
      positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
      fft_magnitudes = np.abs(fft_output)[positive_frequency_indices]
      frequencies_2 = frequencies[positive_frequency_indices]
      top_n_indices = np.argsort(fft_magnitudes)[-N:][::-1]
      strongest_frequencies = frequencies_2[top_n_indices]
      strongest_magnitudes  = fft_magnitudes[top_n_indices] / 40000

      self.debug.info_message("strongest_frequencies: " + str(strongest_frequencies))
      self.debug.info_message("strongest_magnitudes: " + str(strongest_magnitudes))

      candidate_segments = []
      segment_width = 40
      for frequency_count in range(0, N):
        frequency = strongest_frequencies[frequency_count]
        self.debug.info_message("frequency: " + str(frequency))
        magnitude = strongest_magnitudes[frequency_count]
        self.debug.info_message("magnitude: " + str(magnitude))
        segment = int((frequency - 240 - (segment_width // 2)) // segment_width)
        #if (segment not in candidate_segments) and magnitude > 100 : 
        if (segment not in candidate_segments) and magnitude > magnitude_trigger : 
          candidate_segments.append(segment)

      self.debug.info_message("candidate_segments: " + str(candidate_segments))


      N = 1
      all_magnitudes = []
      carrier_transitions = []
      dict_last_frequency = {}
      dict_carrier_transitions = {}
      dict_signal_start = {}
      dict_signal_end = {}  # last block of signal detected
      dict_signal_complete = {}

      for segment_count in range(0, len(candidate_segments)):
        center_frequency = ((candidate_segments[segment_count] + 1) * segment_width) + 240
        dict_carrier_transitions[center_frequency] = []

      average_magnitude = 10000000

      for segment_count in range(0, len(candidate_segments)):
        center_frequency = ((candidate_segments[segment_count] + 1) * segment_width) + 240 #+ (segment_width // 2))
        self.debug.info_message("center_frequency: " + str(center_frequency))
        lo = center_frequency - 18.5
        high = center_frequency + 18.5

        for signal_scan in range(0, len(data), 100): # analyze first 5 seconds of data only. use increments of 10 pulses (100 samples each)
          sub_signal = data[signal_scan:signal_scan + 1000]
          dict_signal_complete[center_frequency] = True

          fft_output = np.fft.fft(sub_signal)
          frequencies = np.fft.fftfreq(len(sub_signal), 1/self.osmod.getRxSampleRate())

          total_magnitude = 0

          positive_frequency_indices = np.where((frequencies > lo) & (frequencies < high))[0]
          fft_magnitudes = np.abs(fft_output)[positive_frequency_indices]
          frequencies_2 = frequencies[positive_frequency_indices]

          top_n_indices = np.argsort(fft_magnitudes)[-N:][::-1]
          strongest_frequencies = frequencies_2[top_n_indices]
          strongest_magnitudes  = fft_magnitudes[top_n_indices] / 1000
          #self.debug.info_message("strongest_magnitudes: " + str(strongest_magnitudes))

          if len(strongest_frequencies) != 0: # and sum_magnitude > average_magnitude * 1.5  :
            current_frequency = strongest_frequencies[0]
            current_magnitude = strongest_magnitudes[0]
            #self.debug.info_message("current_magnitude: " + str(current_magnitude))

            if current_magnitude > magnitude_trigger:
              dict_signal_end[center_frequency] = signal_scan
              dict_signal_complete[center_frequency] = False
              if center_frequency not in dict_signal_start:
                self.debug.info_message("found signal start")
                self.debug.info_message("signal_scan: " + str(signal_scan))
                self.debug.info_message("current_magnitude: " + str(current_magnitude))
                if signal_scan > 0:
                  dict_signal_start[center_frequency] = signal_scan + 800 # 1st two pulses in 100 pulse sequence triggers detection
                else:
                  dict_signal_start[center_frequency] = signal_scan

            if current_magnitude > magnitude_trigger:
              if center_frequency not in dict_last_frequency:
                dict_last_frequency[center_frequency] = current_frequency
              elif current_frequency != dict_last_frequency[center_frequency]:
                #self.debug.info_message("abs(current_frequency - center_frequency): " + str(abs(current_frequency - center_frequency)))
                if abs(current_frequency - center_frequency) > 15:

                  if current_frequency - center_frequency > 0 and dict_last_frequency[center_frequency] - center_frequency < 0:
                    dict_last_frequency[center_frequency] = current_frequency
                    dict_carrier_transitions[center_frequency].append(signal_scan)
                    self.debug.info_message("found carrier transition")
                    self.debug.info_message("signal_scan: " + str(signal_scan))
                  elif current_frequency - center_frequency < 0 and dict_last_frequency[center_frequency] - center_frequency > 0:
                    dict_last_frequency[center_frequency] = current_frequency
                    dict_carrier_transitions[center_frequency].append(signal_scan)
                    self.debug.info_message("found carrier transition")
                    self.debug.info_message("signal_scan: " + str(signal_scan))

                  #self.debug.info_message("found carrier transition")
                  #self.debug.info_message("signal_scan: " + str(signal_scan))
                  #dict_last_frequency[center_frequency] = current_frequency
                  #dict_carrier_transitions[center_frequency].append(signal_scan)

      self.debug.info_message("dict_signal_start: " + str(dict_signal_start))
      self.debug.info_message("dict_signal_end: " + str(dict_signal_end))
      self.debug.info_message("dict_signal_complete: " + str(dict_signal_complete))

      min_block_start_location = 100000000

      for segment_count in range(0, len(candidate_segments)):
        center_frequency = ((candidate_segments[segment_count] + 1) * segment_width) + 240 

        if len(dict_carrier_transitions[center_frequency]) > 4:
          self.debug.info_message("center_frequency: " + str(center_frequency))
          self.debug.info_message("dict_carrier_transitions: " + str(dict_carrier_transitions[center_frequency]))

          new_array = np.array(dict_carrier_transitions[center_frequency])
          for test in range(7, 2, -1):
            test_len = 2 ** test

            new_array_2 = new_array % (test_len * 100)

            """ rebase """
            unique_items, counts = np.unique(new_array_2, return_counts = True)
            min_count = 100000000
            for rebase_count in range(0, len(counts)):
              if counts[rebase_count] > len(new_array_2) * 0.3 and unique_items[rebase_count] < min_count:
                min_count = unique_items[rebase_count]
            if min_count != 100000000:
              new_array_2 = new_array_2 - min_count

            new_array_3 = new_array_2 // 400

            unique_items, counts = np.unique(new_array_3, return_counts = True)

            self.debug.info_message("B unique_items: " + str(unique_items))
            self.debug.info_message("B counts: " + str(counts))

            if np.max(counts) > len(new_array_3) * 0.6 and center_frequency == self.osmod.center_frequency:
              self.debug.info_message("center_frequency: " + str(center_frequency))
              self.debug.info_message("MATCH. length is: " + str(test_len * 200))
              self.debug.info_message("dict_carrier_transitions: " + str(dict_carrier_transitions[center_frequency]))
              block_start_location = np.min(dict_carrier_transitions[center_frequency])
              if block_start_location < min_block_start_location:
                min_block_start_location = block_start_location
              mode_name = str(test_len * 200)
              self.osmod.form_gui.window['text_input_detected_mode'].update("LB28-" + str(mode_name) + "-I3")

              break

      if self.osmod.center_frequency in dict_signal_start:
        signal_start = dict_signal_start[self.osmod.center_frequency]
        signal_end   = dict_signal_end[self.osmod.center_frequency]
        signal_length_seconds = (signal_end - signal_start) // 8000
 
        self.osmod.form_gui.window['text_input_detected_block_start'].update(str(signal_start) + "," + str(signal_length_seconds))
        return data[signal_start:]
      else:
        return data

    except:
      self.debug.error_message("Exception in findDecodeCandidates: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def translateOutbound(self, message):
    self.debug.info_message("translateOutbound")
    try:
      #encoding_b64    = ' abcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+`-={}.[]\\:\";\'<'

      return self.getEncodeEscapes(message)

    except:
      self.debug.error_message("Exception in translateOutbound: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return "12345"

  def translateInbound(self, message):
    self.debug.info_message("translateInbound")
    try:
      #encoding_b64    = ' abcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+`-={}.[]\\:\";\'<'

      return self.getDecodeEscapes(message)

    except:
      self.debug.error_message("Exception in translateInbound: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getEncodeEscapes(self, message):
    self.debug.info_message("getEncodeEscapes")

    try:
      first_seq_1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      dict_second_seq_1 = {'=':'==', '[':'=a', ']':'=b', '{':'=c', '}':'=d', '~':'=f', '|':'=g', '>':'=p', '?':'=q', '<':'=r', '"':'=s', '\\':'=t', '^':'=u', '`':'=v'}
      translated = ''

      self.debug.info_message("message: " + str(message))

      """ replace / with // """ 
      message = message.replace('/','//')
      modified_message = message
      previous_char = message[0]
      ignore = False
      for index in range(1, len(message)):
        if message[index] == previous_char and not previous_char.isdigit() :
          if ignore == False and previous_char != '/':
            modified_message = self.getRunLengthEncode(modified_message, previous_char)
            ignore = True
        else:
          ignore = False
        previous_char = message[index]

      message = modified_message

      #message = modified_message.replace('=','==')
      if (platform.system() == 'Windows'):
        message = message.replace('\r\n','=n')
        message = message.replace('\r','=n')
        message = message.replace('\n','=n')
      else:
        message = message.replace('\r','=n')
        message = message.replace('\n','=n')

      caps_lock = False
      for index in range(0, len(message)):
        current_char = message[index]
        if current_char in first_seq_1:
          if  caps_lock == True:
            translated = translated + current_char.lower()
          elif index + 1 < len(message) and message[index + 1] in first_seq_1:
            translated = translated + "=i" + current_char.lower() # 2 CAPS in a row so CAPS LOCK ON
            caps_lock = True
          else:
            translated = translated + "/" + current_char.lower()
        elif current_char in dict_second_seq_1:
          translated = translated + dict_second_seq_1[current_char]
        else:
          if current_char.isalpha() and caps_lock == True:
            caps_lock = False
            translated = translated + "=m" + current_char # revert to normal mode
          else:
            translated = translated + current_char

      self.debug.info_message("translated: " + str(translated))

      """ /E control character for end of message """
      """ /F used ~ character """


    except:
      self.debug.error_message("Exception in getEncodeEscapes: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return translated



  def getDecodeEscapes(self, message):
    self.debug.info_message("getDecodeEscapes")

    try:
      first_seq = 'abcdefghijklmnopqrstuvwxyz'
      dict_seq = {'==':'=', '=a':'[', '=b':']', '=c':'{', '=d':'}', '=f':'~', '=g':'|', '=p':'>', '=q':'?', '=r':'<', '=s':'"', '=t':'\\', '=u':'^', '=v':'`'}
      dict_seq_2 = {'=i':'U', '=m':'L'}

      string_out = ''
      char_count = 0
      message_len = len(message)
      while char_count < message_len:
        if(char_count+1 < message_len):
          if(message[char_count] == '/'):
            """ test to see if this is an escape sequence"""
            if message[char_count+1] in first_seq:
              string_out = string_out + message[char_count+1].upper()
              char_count = char_count + 2
            elif message[char_count+1] == '/':
              string_out = string_out + '//'
              char_count = char_count + 2
            else:
              string_out = string_out + message[char_count]
              char_count = char_count + 1
          else:
            string_out = string_out + message[char_count]
            char_count = char_count + 1
        else:
          string_out = string_out + message[char_count]
          char_count = char_count + 1

      self.debug.info_message("getDecodeEscapes 1 : " + str(string_out))


      caps_lock = False
      message = string_out
      string_out = ''
      char_count = 0
      message_len = len(message)
      while char_count < message_len:
        if(char_count+1 < message_len):
          if(message[char_count] == '='):
            """ test to see if this is an escape sequence"""
            if message[char_count] + message[char_count+1] in dict_seq:
              string_out = string_out + dict_seq[message[char_count] + message[char_count+1]]
              char_count = char_count + 2
            elif message[char_count] + message[char_count+1] in dict_seq_2:
              if dict_seq_2[message[char_count] + message[char_count+1]] == 'U':
                caps_lock = True
              elif dict_seq_2[message[char_count] + message[char_count+1]] == 'L':
                caps_lock = False
              #string_out = string_out + dict_seq[message[char_count] + message[char_count+1]]
              char_count = char_count + 2
            elif(message[char_count+1] == 'n'):
              if (platform.system() == 'Windows'):
                string_out = string_out + '\r\n'
              else:
                string_out = string_out + '\n'
              char_count = char_count + 2
            else:
              string_out = string_out + message[char_count]
              char_count = char_count + 1
          else:
            if message[char_count] != '|':
              if message[char_count].isalpha():
                if caps_lock == True:
                  string_out = string_out + message[char_count].upper()
                else:
                  string_out = string_out + message[char_count]
              else:
                string_out = string_out + message[char_count]

              #string_out = string_out + message[char_count]
            char_count = char_count + 1
        else:
          if message[char_count] != '|':
            if message[char_count].isalpha():
              if caps_lock == True:
                string_out = string_out + message[char_count].upper()
              else:
                string_out = string_out + message[char_count]
            else:
              string_out = string_out + message[char_count]

            #string_out = string_out + message[char_count]
          char_count = char_count + 1

      self.debug.info_message("getDecodeEscapes 2 : " + str(string_out))

      modified_message = self.getRunLengthDecode(string_out)
      message = modified_message.replace('//', '/')

      return message

    except:
      self.debug.error_message("Exception in getDecodeEscapes: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getRunLengthEncode(self, message, delimeter_char):
    self.debug.info_message("getRunLengthEncode")

    complete_outer = False

    find_it = delimeter_char + delimeter_char

    while(complete_outer == False):
      inner_count = 2
      complete_inner = False
      while(complete_inner == False):
        if( find_it in message):
          message = message.replace(find_it, cn.ESCAPE_CHAR + str(inner_count) + delimeter_char,1 )
          find_it = cn.ESCAPE_CHAR + str(inner_count) + delimeter_char + delimeter_char
          inner_count = inner_count + 1
          #self.debug.info_message("getRunLengthEncode message: " + str(message))
        else:
          complete_inner = True

      find_it = delimeter_char + delimeter_char
      if( find_it not in message):
        complete_outer = True

    """ replace the 2 character ones with the original as it is shorter."""
    message = message.replace(cn.ESCAPE_CHAR + '2' + delimeter_char, delimeter_char + delimeter_char)

    return message


  def getRunLengthDecode(self, message):
    self.debug.info_message("getRunLengthDecode")

    try:
      char_count = 0
      string_out = ''
      message_len = len(message)
      while char_count < message_len:
        if(char_count+1 < message_len):
          if(message[char_count] == '/'):
            """ test to see if this is an escape sequence"""
            if '/' + message[char_count+1] != '//':
              """ make sure this is an RLE escape sequence """
              if(message[char_count+1].isdigit()):
                delimeter_char = message[char_count+2]
                if(message[char_count+2].isdigit()):
                  delimeter_char = message[char_count+3]
                  if(message[char_count+3].isdigit()):
                    delimeter_char = message[char_count+4]
                    """ four digit RLE codes and up not supported"""
                    if(message[char_count+4].isdigit()):
                      self.debug.info_message("do nothing")
                      string_out = string_out + message[char_count]
                      char_count = char_count + 1
                    elif(message[char_count+4] == delimeter_char):
                      """ process triple digit RLE code"""
                      string_out = string_out + (delimeter_char * ((int(message[char_count+1])*100) + (int(message[char_count+2])*10)+ (int(message[char_count+3]))) )
                      char_count = char_count + 5
                    else:
                      string_out = string_out + message[char_count]
                      char_count = char_count + 1
                  elif(message[char_count+3] == delimeter_char):
                    """ process double digit RLE code"""
                    string_out = string_out + (delimeter_char * ((int(message[char_count+1])*10) + (int(message[char_count+2]))) )
                    char_count = char_count + 4
                  else:
                    string_out = string_out + message[char_count]
                    char_count = char_count + 1
                elif(message[char_count+2] == delimeter_char):
                  """ process single digit RLE code"""
                  string_out = string_out + (delimeter_char * int(message[char_count+1]) )
                  char_count = char_count + 3
                else:
                  string_out = string_out + message[char_count]
                  char_count = char_count + 1
              else:
                string_out = string_out + message[char_count]
                char_count = char_count + 1
            else:
              string_out = string_out + message[char_count]
              char_count = char_count + 1
          else:
            string_out = string_out + message[char_count]
            char_count = char_count + 1
        else:
          string_out = string_out + message[char_count]
          char_count = char_count + 1

      message = string_out
      self.debug.info_message("completed getRunLengthDecode. unescaped message: " + str(message) )

      return message

    except:
      self.debug.error_message("Exception in getRunLengthDecode: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))





  def protectMessage(self, message_text, truncate_to_max_msglength, num_rotation_chars, max_message_length):
    self.debug.info_message("protectMessage")
    try:
      sequence_identifier = "0123456789abcdefghijklmnopqrstuvwxyz"
      protected_string = ''

      crc_fragment_size = int(self.osmod.form_gui.window['in_crc_fragment_size'].get())

      addCallsignEOM = self.osmod.form_gui.window['cb_enable_eom_callsign'].get()
      #addCallsignEOM = True
      callsign = self.getTranslatedCallsign()
      callsign_len = len(callsign)

      if truncate_to_max_msglength:
        if addCallsignEOM:
          num_fragments = (max_message_length - num_rotation_chars) // (crc_fragment_size + 4)
          remainder = (max_message_length - ((num_fragments * (crc_fragment_size + 4)) + num_rotation_chars)) - 4 - 3
          remainder = max(remainder, 0)
          additional_chars = (num_fragments * 4) + num_rotation_chars + callsign_len
          if remainder > 0:
            additional_chars = additional_chars + 4
          message_text = message_text[0: max_message_length - additional_chars - 3] + callsign
          self.debug.info_message("addCallsignEOM: ")
          self.debug.info_message("message_text: " + str(message_text))

        else:
          num_fragments = (max_message_length - num_rotation_chars) // (crc_fragment_size + 4)
          remainder = (max_message_length - ((num_fragments * (crc_fragment_size + 4)) + num_rotation_chars)) - 4 - 3
          remainder = max(remainder, 0)
      else:
        num_fragments = len(message_text) // crc_fragment_size
        remainder = len(message_text) - (num_fragments * crc_fragment_size)
      self.debug.info_message("remainder: " + str(remainder))

      for frag_count in range(0, num_fragments):
        frag_string = '|' + sequence_identifier[frag_count] + message_text[frag_count * crc_fragment_size:(frag_count+1) * crc_fragment_size]
        checksum = self.calcFragmentCRC(frag_string)
        protected_string = protected_string + frag_string + checksum
        self.debug.info_message("frag_string: " + str(frag_string))

      self.debug.info_message("processing remainder...")

      if remainder > 0:
        remainder_location = num_fragments * crc_fragment_size
        frag_string = '|' + sequence_identifier[num_fragments] + message_text[remainder_location:remainder_location + remainder]
        checksum = self.calcFragmentCRC(frag_string)
        protected_string = protected_string + frag_string + checksum + '|||'
        self.debug.info_message("frag_string: " + str(frag_string))

      self.debug.info_message("protected_string: " + str(protected_string))

    except:
      self.debug.error_message("Exception in protectMessage: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return protected_string

  """
  Attribution - CRC Polynomials - Philip Koopman - Carnegie Mellon University

  The general purpose polynomials used are derived from the Best CRC Polynomials document by Philip Koopman, Carnegie Mellon University -

  https://users.ece.cmu.edu/~koopman/crc/ 

  published under Creative Commons License: https://creativecommons.org/licenses/by/4.0/
  """

  def getChecksum(self, mystr):
    return self.calcFragmentCRC(mystr)


  """
  to avoid unnecessary complexity, assume a worst case 8 bit character size for CRC calculations
  """
  def calcFragmentCRC(self, string):
    if(len(string)<=60):
      return self.calcTwoDigitCRCShort(string)
    elif(len(string)<=120):
      return self.calcTwoDigitCRCLong(string)
    else:
      return self.calcThreeDigitCRC(string)


  """
  CRC calculation uses 5 bit nibbles in base 32 so two digits is 10 bits
  This is used for the short fragments 10 thru 60 characters
  0x247 polynomial protects up to 501 bit data word (62 x 8 bit characters) length at HD=4
  """
  def calcTwoDigitCRCShort(self, string):
    return self.calcCRC(10, 0x247, string)

  """
  CRC calculation uses 5 bit nibbles in base 32 so two digits is 10 bits
  This is used for the longer fragments 70 thru 120 characters
  0x327 polynomial protects up to 1013 bit data word (126 x 8 bit characters) length at HD=3
  """
  def calcTwoDigitCRCLong(self, string):
    return self.calcCRC(10, 0x327, string)

  """
  CRC calculation uses 5 bit nibbles in base 32 so three digits is 15 bits
  0x4306 polynomial protects up to 16368 bit data word (2046 x 8 bit characters) length at HD=4
  this is used for longer fragments and end of message checksum for messages <= 2046 characters
  """
  def calcThreeDigitCRC(self, string):
    return self.calcCRC(15, 0x4306, string)



  """ always use a 20 bit / 4 digit CRC for end of message checksum"""
  def calcEOMCRC(self, string):
    return self.calcFourDigitCRC(string)

  #base32_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
  base32_chars = "0123456789abcdefghijklmnopqrstuv"

  """
  CRC calculation uses 5 bit nibbles in base 32 so four digits is 20 bits
  0xc1acf polynomial protects up to 524267 bit data word (65533 x 8 bit characters) length at HD=4
  this is used for end of message checksum for messages > 2046 characters
  """
  def calcFourDigitCRC(self, string):
    return self.calcCRC(20, 0xc1acf, string)

  def calcCRC(self, width, poly, string):

    self.debug.info_message('calcCRC')

    data = bytes(string,"ascii")

    init_value=0x00
    final_xor_value=0x00
    reverse_input=False
    reverse_output=False

    configuration = Configuration(width, poly, init_value, final_xor_value, reverse_input, reverse_output)

    use_table = True
    crc_calculator = Calculator(configuration, use_table)

    checksum = crc_calculator.checksum(data)
    self.debug.info_message(str(checksum))

    if(width == 10):
      high, low = checksum >> 5, checksum & 0x1F
      self.debug.info_message('10 bit checksum: ' + str(self.base32_chars[high] + self.base32_chars[low]))
      return self.base32_chars[high] + self.base32_chars[low]
    elif(width == 15):
      high, mid, low = checksum >> 10, (checksum >> 5) & 0x1F, checksum & 0x1F
      self.debug.info_message('15 bit checksum: ' + str(self.base32_chars[high] + self.base32_chars[mid] + self.base32_chars[low]))
      return self.base32_chars[high] + self.base32_chars[mid] + self.base32_chars[low]
    elif(width == 20):
      high, mid_high, mid_low, low = checksum >> 15, (checksum >> 10) & 0x1F, (checksum >> 5) & 0x1F, checksum & 0x1F
      self.debug.info_message('20 bit checksum: ' + str(self.base32_chars[high] + self.base32_chars[mid_high] + self.base32_chars[mid_low] + self.base32_chars[low]))
      return self.base32_chars[high] + self.base32_chars[mid_high] + self.base32_chars[mid_low] + self.base32_chars[low]

    return ''







  """ add the callsign at the start of the message"""
  def addCallsignSOM(self, message):
    self.debug.info_message("addCallsignSOM")
    try:

      callsign = self.osmod.form_gui.window['in_station_callsign'].get()
      #group = self.osmod.form_gui.window['in_group'].get()

      return callsign.upper() + ' ' + message

    except:
      self.debug.error_message("Exception in addCallsignSOM: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  """ add the callsign at the start of the message"""
  def addCallsignSOM_WithColon(self, message):
    self.debug.info_message("addCallsignSOM")
    try:

      callsign = self.osmod.form_gui.window['in_station_callsign'].get()
      #group = self.osmod.form_gui.window['in_group'].get()

      return callsign.upper() + ': ' + message

    except:
      self.debug.error_message("Exception in addCallsignSOM: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getTranslatedCallsign(self):
    self.debug.info_message("getTranslatedCallsign")
    try:
      callsign = self.osmod.form_gui.window['in_station_callsign'].get()

      return ' =i' + callsign.lower()

    except:
      self.debug.error_message("Exception in getTranslatedCallsign: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

