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

from numpy import pi
from scipy.signal import butter, filtfilt, firwin, sosfiltfilt, hilbert
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io.wavfile import write, read
from datetime import datetime, timedelta
from scipy.fft import fft

from numpy.fft import ifft
from scipy.signal import periodogram
from numpy.polynomial import Chebyshev as T
from scipy import stats
from scipy.interpolate import CubicSpline, splrep, splev, PchipInterpolator, UnivariateSpline
from osmod_c_interface import ptoc_float_array, ptoc_double_array, ptoc_float, ctop_int, ptoc_int_array, ptoc_numpy_int_array, ptoc_double, ptoc_double_pointer_array

from scipy import signal as scipy_signal
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



class ModemCoreUtils(object):

  osmod = None
  dict_binpair_to_quad = {'00':0, '01':1, '10':2, '11':3}
  """ optimized for 64 bit encodings """

  """ temporary base 64 char format """
  encoding_b64    = 'abcdefghijklmnopqrstuvwxyz 0123456789~!@#$%^&*()_+`-={}|[]\\:\";\'<'
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

  def writeFileWav(self, filename, multi_block):
    self.debug.info_message("writeFileWav")
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
      write(filename, self.osmod.sample_rate, multi_block)
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

  # 'tx_filter' : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
  def apply_filter(self, signal, params, center_frequency):
    try:
      filter_type = params[0]
      filter_pass_type = params[1]
      filter_width = params[2]
      repeats = params[3]
      filter_order = params[4]

      if filter_type == ocn.FILTER_NONE:
        return signal
      elif filter_type == ocn.FILTER_BUTTERWORTH:
        if filter_pass_type == ocn.FILTER_BAND_PASS:
          """ filter the output signal """
          for _ in range(repeats):
            sig1 = self.osmod.modulation_object.filter_sharp_cutoff_low_pass(signal, center_frequency + filter_width/2, filter_order)
            signal = sig1
          for _ in range(repeats):
            sig2 = self.osmod.modulation_object.filter_sharp_cutoff_high_pass(signal, center_frequency - filter_width/2, filter_order)
            signal = sig2
        return signal

    except:
      self.debug.error_message("Exception in apply_filter: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def filter_sharp_cutoff_low_pass(self, signal, cutoff_freq, filter_order):
  #def filter_sharp_cutoff_low_pass(self, signal, cutoff_freq):
    #return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'low',50)
    return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'low',filter_order)

  def filter_sharp_cutoff_high_pass(self, signal, cutoff_freq, filter_order):
  #def filter_sharp_cutoff_high_pass(self, signal, cutoff_freq):
    #return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'highpass',50)
    return self.filter_sharp_cutoff_common(signal, cutoff_freq, 'highpass',filter_order)

  """ This method works well """
  def filter_sharp_cutoff_common(self, signal, cutoff_freq, filter_type, order):
    try:
      nyquist_frequency = 0.5 * self.osmod.sample_rate
      normalized_cutoff = cutoff_freq / nyquist_frequency
      sos = butter(order, normalized_cutoff, btype=filter_type, analog=False, output='sos')
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



  def getStrongestFrequencies(self, data, N, lo, high):
    self.debug.info_message("getStrongestFrequencies")
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
    self.debug.info_message("getIsSignalPresent")
    try:
      delta = 5
      fft_output = np.fft.fft(data)
      frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      positive_frequency_indices = np.where((frequencies > watch_freq - delta) & (frequencies < watch_freq + delta))[0]

      fft_magnitude = np.abs(fft_output)[positive_frequency_indices]
      frequencies = frequencies[positive_frequency_indices]
      strongest_index = np.argmax(fft_magnitude)

      fft_data = np.abs(fft_output)
      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      sys.stdout.write("strongest frequency: " + str(strong_freqs) + "\n")
      sys.stdout.write("strong_magnitudes: " + str(strong_magnitudes) + "\n")
  
    except:
      self.debug.error_message("Exception in getIsSignalPresent: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed getIsSignalPresent: ")

    return strong_freqs, strong_magnitudes



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
      fft_data = np.abs(fft_output)
      interpolated_peak_index = self.fft_parabolic_interpolation(fft_data, np.argmax(fft_data[:len(fft_data)//2]) )
      interpolated_frequency = interpolated_peak_index * self.osmod.sample_rate / len(data)

      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      sys.stdout.write("strongest frequency: " + str(strong_freqs) + "\n")
      sys.stdout.write("strongest interpolated frequency: " + str(interpolated_frequency) + "\n")
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



  """ calculate the noise of the entire passband"""
  def calculateNoisePowerSNR(self, signal):
    fft_output = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_output), 1/self.osmod.sample_rate)
    psd = np.abs(fft_output)**2

    f_low = 250
    f_high = 2750

    indices = np.where((frequencies >= f_low) & (frequencies <= f_high))
    psd_selected = psd[indices]
    noise_power = np.sum(psd_selected)

    return noise_power


  def calculateSNR(self, signal, signal_frequency):
    t = np.arange(0, 1, 1/self.osmod.sample_rate)
    fft_output = np.fft.fft(signal)
    psd = np.abs(fft_output)**2
    frequencies = np.fft.fftfreq(len(fft_output), 1/self.osmod.sample_rate)

    signal_index1 = np.where(np.isclose(frequencies, signal_frequency[0], atol=1/(t[-1]*2)))[0][0]
    signal_power1 = np.sum(psd[signal_index1-2:signal_index1+3])
    signal_index2 = np.where(np.isclose(frequencies, signal_frequency[1], atol=1/(t[-1]*2)))[0][0]
    signal_power2 = np.sum(psd[signal_index2-2:signal_index2+3])

    signal_power = (signal_power1 + signal_power2)
    noise_power = self.calculateNoisePowerSNR(signal) - signal_power

    snr = 10 * np.log10( signal_power / noise_power)
    self.debug.info_message("SNR: " + "{:.2f}".format(snr) + "dB")

    return snr


  def calculate_EbN0(self, signal, signal_frequency, numbits, bit_rate, noise_free_signal, center_frequency):
    self.debug.info_message("calculateSNR_EbN0")

    gc.collect()

    try:
      fft_signal = np.fft.fft(noise_free_signal)
      frequencies = np.fft.fftfreq(len(fft_signal), 1/self.osmod.sample_rate)

      """ use best method to determine signal width"""
      if self.osmod.tx_filter[2] > 0:
        freq_low_signal  = center_frequency - (self.osmod.tx_filter[2]/2)    
        freq_high_signal = center_frequency + (self.osmod.tx_filter[2]/2)    
      else:
        freq_low_signal  = signal_frequency[0] + self.osmod.fft_filter[0]
        freq_high_signal = signal_frequency[1] + self.osmod.fft_filter[3]

      freq_indices = np.where((frequencies >= freq_low_signal) & (frequencies <= freq_high_signal))
      signal_power = np.abs(fft_signal[freq_indices])**2
      self.debug.info_message("signal_power: " + str(signal_power) )

      fft_noise = np.fft.fft(signal)
      frequencies = np.fft.fftfreq(len(fft_noise), 1/self.osmod.sample_rate)
      freq_low_noise = 0
      freq_indices = np.where(((frequencies > freq_low_noise) & (frequencies <= freq_low_signal)) | ((frequencies >= freq_high_signal) ))
      noise_power = np.abs(fft_noise[freq_indices])**2
      self.debug.info_message("noise_psd: " + str(noise_power) )

      Eb = np.mean(signal_power) / bit_rate
      self.debug.info_message("Eb: " + str(Eb) )

      """ N0 is often derived using average (mean)"""
      N0 = np.mean(noise_power)
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


  """ Eb/N0 = SNR(dB) + 10 * log10(Bandwidth / bit_rate)   """
  def calculate_EbN0_old(self, signal, signal_frequency, numbits, bit_rate, noise_free_signal):
    self.debug.info_message("calculateSNR_EbN0")

    gc.collect()

    try:
      t = np.arange(0, 1, 1/self.osmod.sample_rate)
      fft_output = np.fft.fft(noise_free_signal)
      psd = np.abs(fft_output)**2
      frequencies = np.fft.fftfreq(len(fft_output), 1/self.osmod.sample_rate)


      freq_low_signal  = signal_frequency[0] + self.osmod.fft_filter[0]
      freq_low_signal_hi  = signal_frequency[0] + self.osmod.fft_filter[1]
      freq_high_signal = signal_frequency[1] + self.osmod.fft_filter[3]
      freq_high_signal_lo = signal_frequency[1] + self.osmod.fft_filter[2]


      #freq_low_signal  = signal_frequency[0] - 2
      #freq_low_signal_hi  = signal_frequency[0] + 2
      #freq_high_signal = signal_frequency[1] + 2
      #freq_high_signal_lo = signal_frequency[1] - 2

      freq_indices = np.where(((frequencies >= freq_low_signal) & (frequencies <= freq_low_signal_hi)) | ((frequencies >= freq_high_signal_lo) & (frequencies <= freq_high_signal)) )[0]
      signal_psd = np.abs(fft_output[freq_indices])**2
      #self.debug.info_message("signal_psd: " + str(signal_psd) )

      fft_output = np.fft.fft(signal)
      psd = np.abs(fft_output)**2
      frequencies = np.fft.fftfreq(len(fft_output), 1/self.osmod.sample_rate)

      freq_low_noise = 250
      freq_high_noise = 2750
      freq_indices = np.where(((frequencies >= freq_low_noise) & (frequencies <= freq_low_signal)) | ((frequencies >= freq_low_signal_hi) & (frequencies <= freq_high_signal_lo)) | ((frequencies >= freq_high_signal) & (frequencies <= freq_high_noise)))
      noise_psd = np.abs(fft_output[freq_indices])**2
      self.debug.info_message("noise_psd: " + str(noise_psd) )

      signal_energy = np.sum(signal_psd)
      self.debug.info_message("signal_energy: " + str(signal_energy) )
      eb = signal_energy / numbits
      self.debug.info_message("eb: " + str(eb) )
      """ N0 is often derived using average (mean)"""
      N0 = np.mean(noise_psd)
      self.debug.info_message("N0: " + str(N0) )
      """ ...but the definition states that N0 is psd in 1Hz of bandwidth..."""

      ebn0 = eb / N0
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


  def shiftAllFrequenciesOld(self, signal, freq_shift_amount):

    try:
      fs = 92800 # self.osmod.sample_rate
      duration_seconds = len(signal) / fs
      t = np.linspace(0, duration_seconds, fs, endpoint=False)
    
      self.debug.info_message("signal shape: " + str(signal.shape))
      self.debug.info_message("t shape: " + str(t.shape))

      #t_col = t[:, np.newaxis]
      #shifted_signal = signal * np.exp(1j * 2 * np.pi * freq_shift_amount * t_col)

      #shifted_signal = signal * np.exp(1j * 2 * np.pi * freq_shift_amount * t)
      #return (shifted_signal.real + shifted_signal.imag ) / 2

      #return shifted_signal

      #for -ve freq shift
      analytic_signal = hilbert(signal)
      #shifted_signal = analytic_signal * np.exp(1j * 2 * np.pi * freq_shift_amount * t)
      shifted_signal = signal * np.exp(1j * 2 * np.pi * freq_shift_amount * t)
      #return (shifted_signal.real + shifted_signal.imag ) / 2
      return shifted_signal.astype(np.float64)

    except:
      self.debug.error_message("Exception in shiftAllFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def shiftAllFrequencies(self, signal, shift_hz):
    self.debug.info_message("shiftAllFrequencies")

    try:
      n = len(signal)
      fs = self.osmod.sample_rate

      window = np.ones(n)
      #window = np.hanning(n)
      #window = np.hamming(n)
      #window = np.blackman(n)
      #window = np.kaiser(n, beta=5)
      #window = np.kaiser(n, beta=10)

      #window = np.hanning(n)
      fft_data = np.fft.rfft(signal * window)

      #fft_signal  = sp.fft.fft(signal)
      frequencies = np.fft.rfftfreq(len(signal), d=1/self.osmod.sample_rate)

      bins_to_shift = int(round(shift_hz * n / fs))

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


  def resampleDopplerShift(self, signal, orig_fs, new_fs):
    self.debug.info_message("resampleDopplerShift")

    try:
      #from scipy import signal
      #orig_fs = self.osmod.sample_rate
      #new_fs = 
      num_samples_new_signal = int(len(signal) * new_fs / orig_fs)
      resampled_signal = scipy_signal.resample(signal, num_samples_new_signal)

      return resampled_signal
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
        new_signal = self.osmod.modulation_object.resampleDopplerShift(test_signal, fs / adjust_ratio, fs)
        new_frequency_test = self.osmod.modulation_object.getStrongestFrequency(new_signal, target_strongest_frequency - frequency_delta, target_strongest_frequency + frequency_delta)
        error_value = abs((new_frequency_test - frequency_test) * 1000)
        self.debug.info_message("error_value: " + str(error_value))
        if error_value < 5:
          break
        else:
          frequency_test = new_frequency_test
          test_signal = new_signal
          self.debug.info_message("frequency_test: " + str(frequency_test))

      return new_signal
    except:
      self.debug.error_message("Exception in linearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))





  def adjustFrequencyShiftAndDopplerShift(self, noise_free_signal, values, center_frequency):

    self.debug.info_message("adjustFrequencyShiftAndDopplerShift")
    try:

      """ TEST CODE shift frequencies"""
      self.debug.info_message(" TEST CODE FOR DOPPLER SHIFT AND FREQUENCY SHIFT")

      """ Apply manual frequency shift """
      enable_fine_tune_frequency = self.osmod.form_gui.window['cb_enable_fine_tune_frequency'].get()
      if enable_fine_tune_frequency:
        frequency_shift_value = values['slider_freq_fine_tune']
        noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, frequency_shift_value)
        #noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, -1 * frequency_shift_value)
      enable_fine_tune_resample = self.osmod.form_gui.window['cb_enable_fine_tune_resample'].get()

      """ Auto correct frequency shift """
      enable_resample_auto_correct = self.osmod.form_gui.window['cb_enable_resample_auto_correct'].get()
      enable_frequency_shift_auto_correct = self.osmod.form_gui.window['cb_enable_frequency_shift_auto_correct'].get()
      if enable_frequency_shift_auto_correct and enable_resample_auto_correct == False:
        target_freq_low = center_frequency + self.osmod.resample_params[1]
        actual_low_freq = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal, target_freq_low - 5, target_freq_low + 5)
        frequency_shift_value = target_freq_low - actual_low_freq
        noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, frequency_shift_value)

      """ Apply manual doppler shift """
      if enable_fine_tune_resample:
        doppler_shift_value = values['slider_resample_fine_tune']
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000, 8000 * doppler_shift_value)
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 800000, 800002). # need to be this accurate for correct decode
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8000, 8100)
        #noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, 8100, 8000)
        noise_free_signal = self.osmod.modulation_object.resampleDopplerShift(noise_free_signal, self.osmod.sample_rate, self.osmod.sample_rate * doppler_shift_value)

      """ automatic adjust for linear doppler shift """
      enable_resample_auto_correct = self.osmod.form_gui.window['cb_enable_resample_auto_correct'].get()
      if enable_resample_auto_correct:
        if self.osmod.resample_params[0] == ocn.RESAMPLE_AVAILABLE and self.osmod.resample_params[3] != 0:
          self.debug.info_message("RESAMPLE_AVAILABLE ")
          if enable_frequency_shift_auto_correct == False:
            target_freq_low = center_frequency + self.osmod.resample_params[1]
            #noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, 1382.5, 5)   #this for 6400 mode
            #noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, 1381.875, 5)  #this for 12800 mode
            noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  #this for 12800 mode
          else:
            self.debug.info_message("correcting for frequency shift and doppler shift ")

            if self.osmod.resample_params[0] == ocn.RESAMPLE_AVAILABLE:

              """ first doppler shift correct """
              target_freq_low = center_frequency + self.osmod.resample_params[1]
              noise_free_signal_temp = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  

              """ second calc residual frequency shift component"""
              frequency_test_lower  = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal_temp, 1380, 1385)
              frequency_test_higher = self.osmod.modulation_object.getStrongestFrequency(noise_free_signal_temp, 1414, 1424)
              difference = ((frequency_test_higher - frequency_test_lower) - (self.osmod.resample_params[2] - self.osmod.resample_params[1])) * 10000
              self.debug.info_message("difference: " + str(difference))

              partial_result = difference 
              self.debug.info_message("partial_result: " + str(partial_result))
              calculated_freq_offset = partial_result / self.osmod.resample_params[3] 
              self.debug.info_message("calculated_freq_offset: " + str(calculated_freq_offset))

              """ third apply frequency shift component to original signal """
              noise_free_signal = self.osmod.modulation_object.shiftAllFrequencies(noise_free_signal, calculated_freq_offset)

              """ fourth reapply auto doppler shift correction """
              target_freq_low = center_frequency + self.osmod.resample_params[1]
              noise_free_signal = self.osmod.modulation_object.linearDopplerShiftAutoCorrect(noise_free_signal, target_freq_low, 5)  
            else:
              self.debug.info_message("RESAMPLE_UNAVAILABLE ")

        else:
          self.debug.info_message("RESAMPLE_UNAVAILABLE ")

      return noise_free_signal

    except:
      self.debug.error_message("Exception in linearDopplerShiftAutoCorrect: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def appendTableRow(self, message):
    self.debug.info_message("appendTableRow")
    try:
      center_frequency = self.osmod.getCenterFrequency()
      timestamp = datetime.utcnow().strftime('%Y/%m/%d %H:%M')

      if self.osmod.form_gui.window['cb_use_prod_modes'].get() == True:
        mode = self.osmod.form_gui.window['combo_main_modem_prod_modes'].get()
      else:
        mode = self.osmod.form_gui.window['combo_main_modem_modes'].get()

      data_key = str(center_frequency) + "_" + str(mode)
      #if data_key in self.osmod.dict_rcvd:
      #else:
      self.osmod.dict_rcvd[data_key] = [timestamp, message]
      self.debug.info_message("self.osmod.dict_rcvd: " + str(self.osmod.dict_rcvd))

      data_table = []
      for key, value in self.osmod.dict_rcvd.items():
        data_row = key.split('_') + value
        self.debug.info_message("data_row: " + str(data_row))
        data_table.append(data_row)

      self.debug.info_message("data_table: " + str(data_table))

      self.osmod.form_gui.window['tbl_tmplt_templates'].update(data_table)

      #self.osmod.form_gui.window['tbl_tmplt_templates'].update([[center_frequency,mode,timestamp,message]])
      self.osmod.form_gui.window['ml_txrx_sendtext'].update(value="")


    except:
      self.debug.error_message("Exception in appendTableRow: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
