#!/usr/bin/env python

import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
import matplotlib.pyplot as plt
import scipy as sp
import gc
import FreeSimpleGUI as sg

from numpy import pi
from scipy.signal import butter, filtfilt, firwin, sosfiltfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io.wavfile import write, read
from datetime import datetime, timedelta
from scipy.fft import fft
from numpy.fft import ifft
from scipy.signal import periodogram

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

    #return signal.astype(np.float64)
    return signal.astype(np.longdouble)
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

      multi_block = multi_block * (2**15 - 1) / np.max(np.abs(multi_block))

      self.debug.info_message("writing audio file")
      multi_block = multi_block.astype(np.float32)
      #multi_block = multi_block.astype(np.float64)
      write(filename, self.osmod.sample_rate, multi_block)
    except:
      self.debug.error_message("Exception in writeFileWav: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


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

  def filter_low_pass(self, signal, cutoff_freq):
    return self.filter_common(signal, cutoff_freq, 'low')

  def filter_high_pass(self, signal, cutoff_freq):
    return self.filter_common(signal, cutoff_freq, 'highpass')

  """ This method works well """
  def filter_common(self, signal, cutoff_freq, filter_type):
    self.debug.info_message("filter_low_pass_2")
    try:
      nyquist_frequency = 0.5 * self.osmod.sample_rate
      normalized_cutoff = cutoff_freq / nyquist_frequency

      sos = butter(2, normalized_cutoff, btype=filter_type, analog=False, output='sos')

      return_value = sosfiltfilt(sos, signal)

    except:
      self.debug.error_message("Exception in filter_low_pass_2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed filter_low_pass_2: ")

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
      bit_triplets1 = []
      bit_triplets2 = []

      sent_triplets_1 = []
      sent_triplets_2 = []

      for char in string:
        self.debug.info_message("processing char: " + str(char) )

        self.osmod.form_gui.txwindowQueue.put(str(char))

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
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row1), end="", text_color='green', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_sendtext'].print(str(row2), end="", text_color='green', background_color = 'white')

    except:
      sys.stdout.write("Exception in stringToTriplet: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return [bit_triplets1, bit_triplets2], [sent_triplets_1, sent_triplets_2]


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


  def drawPhaseCharts(self, data, chart_type, window, form_gui, canvas_name):

    self.debug.info_message("drawPhaseCharts")
    try:
      graph = self.osmod.form_gui.window[canvas_name]
      graph.erase()

      x_max = 1300
      y_max = 400
      x_chart_offset = 130
      y_chart_offset = 60
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)

    except:
      self.debug.error_message("Exception in drawPhaseCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """
DATA_MODE           = 0
DATA_EBN0_DB        = 1
DATA_SNR_EQUIV_DB   = 2
DATA_BER            = 3
DATA_CPS            = 4
DATA_BPS            = 5
DATA_NOISE_FACTOR   = 6
DATA_AMPLITUDE      = 7
DATA_CHUNK_SIZE     = 8
  """
  def compare_a_equals_b(self, a, b):
    if a == b:
      return True
    else:
      return False

  def compare_a_greater_than_b(self, a, b):
    if a > b:
      return True
    else:
      return False

  def compare_a_less_than_b(self, a, b):
    if a < b:
      return True
    else:
      return False

  def drawDotPlotCharts(self, data, chart_type, window, values, form_gui):

    self.debug.info_message("drawDotPlotCharts")
    try:
      colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'cyan', 'black', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
      dict_name_colors = {}
      color_count = 0
      dict_lookup = {'Eb/N0': ocn.DATA_EBN0_DB, 'BER':ocn.DATA_BER ,'BPS':ocn.DATA_BPS ,'CPS':ocn.DATA_CPS ,'Chunk Size':ocn.DATA_CHUNK_SIZE ,'SNR':ocn.DATA_SNR_EQUIV_DB, 'Noise Factor':ocn.DATA_NOISE_FACTOR, 'Amplitude':ocn.DATA_AMPLITUDE }

      filter_1 = window['cb_analysis_filter1'].get()
      filter_2 = window['cb_analysis_filter2'].get()
      mode_to_match = window['combo_analysis_modes'].get()
      item_to_compare_index = dict_lookup[window['combo_analysis_itemtocompare'].get()]
      compare_operator = window['combo_analysis_campare_operator'].get()
      if filter_2:
        compare_value = float(window['in_analysis_comparewithvalue'].get())
      compare_func = None
      if compare_operator == '=':
        compare_func = self.compare_a_equals_b
      elif compare_operator == '>':
        compare_func = self.compare_a_greater_than_b
      elif compare_operator == '<':
        compare_func = self.compare_a_less_than_b

      graph = self.osmod.form_gui.window['graph_dotplotdata']
      graph.erase()

      x_max = 1300
      y_max = 400
      x_chart_offset = 130
      y_chart_offset = 60
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)


      if chart_type == 'X:Eb/N0 Y:BER':
        x_index = ocn.DATA_EBN0_DB
        y_index = ocn.DATA_BER
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:Eb/No':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:ChunkSize Y:Eb/N0':
        x_index = ocn.DATA_CHUNK_SIZE
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Chunk Size', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:BER':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_BER
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:BER Y:Eb/N0':
        x_index = ocn.DATA_BER
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Bit Error Rate', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:Eb/N0+ABS(Eb/N0)*BER':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_CALC_1
        for point in range(0, len(data)):
          calculated_value = float(data[point][ocn.DATA_EBN0_DB]) + (abs(float(data[point][ocn.DATA_EBN0_DB]) * float(data[point][ocn.DATA_BER] )))
          data[point].append(calculated_value)
          self.debug.info_message("data[point]: " + str(data[point]))
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb/N0+ABS(Eb/N0)*BER', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')




      data_x_max = -1000000000
      data_x_min = 1000000000
      data_y_max = -1000000000
      data_y_min = 1000000000
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if data[point][ocn.DATA_MODE] == mode_to_match:
            if filter_2:
              if compare_func(float(data[point][item_to_compare_index]), compare_value):
                include = True
              else:
                include = False
            else:
              include = True
        elif filter_2:
          if compare_func(float(data[point][item_to_compare_index]), compare_value):
            include = True
          else:
            include = False
        else:
          include = True

        if include:
          self.debug.info_message("data_x: " + str(float(data[point][x_index])))
          data_x_max = max(data_x_max, float(data[point][x_index]))
          data_x_min = min(data_x_min, float(data[point][x_index]))
          self.debug.info_message("data_y: " + str(float(data[point][y_index])))
          data_y_max = max(data_y_max, float(data[point][y_index]))
          data_y_min = min(data_y_min, float(data[point][y_index]))
          if data[point][ocn.DATA_MODE] not in dict_name_colors:
            dict_name_colors[data[point][ocn.DATA_MODE]] = color_count
            color_count = color_count + 1

      graph.draw_text("{:.2f}".format(data_x_min), location = (x_chart_offset, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_x_max), location = (x_chart_offset + x_max, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_min), location = (x_chart_offset -50 , y_chart_offset), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_max), location = (x_chart_offset -50 , y_chart_offset + y_max), angle = 0, font = '_ 12', color = 'black', text_location = 'center')


      self.debug.info_message("data_x_max: " + str(data_x_max))
      self.debug.info_message("data_y_max: " + str(data_y_max))

      if (data_x_max - data_x_min) == 0 and data_x_max > 0:
        data_x_min = 0
      if (data_x_max - data_x_min) == 0 and data_x_max < 0:
        data_x_max = 0
      if (data_y_max - data_y_min) == 0 and data_y_max > 0:
        data_y_min = 0
      if (data_y_max - data_y_min) == 0 and data_y_max < 0:
        data_y_max = 0

      x_scaling = x_max / (data_x_max - data_x_min)
      y_scaling = y_max / (data_y_max - data_y_min)
      
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if data[point][ocn.DATA_MODE] == mode_to_match:
            if filter_2:
              if compare_func(float(data[point][item_to_compare_index]), compare_value):
                include = True
              else:
                include = False
            else:
              include = True
        elif filter_2:
          if compare_func(float(data[point][item_to_compare_index]), compare_value):
            include = True
          else:
            include = False
        else:
          include = True

        if include:
          x_point = (float(data[point][x_index]) - data_x_min) * x_scaling
          y_point = (float(data[point][y_index]) - data_y_min) * y_scaling
          self.debug.info_message("x_point: " + str(x_point))
          self.debug.info_message("y_point: " + str(y_point))
          plot_color_index = dict_name_colors[data[point][ocn.DATA_MODE]]
          self.debug.info_message("plot_color_index: " + str(plot_color_index))
          plot_color = colors[plot_color_index]
          self.debug.info_message("plot_color: " + str(plot_color))
          graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

      count = 0
      for mode_name, color_index in dict_name_colors.items():
        plot_color = colors[color_index]
        graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*20)), size=16, color=plot_color)
        graph.draw_text(mode_name, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*20)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
        count = count + 1

    except:
      self.debug.error_message("Exception in drawDotPlotCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def re_drawWaveCharts(self, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2):
    self.drawWaveCharts(self.saved_data_for_chart_1, self.saved_data_for_chart_2, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2)

  """ This method works fine"""
  def drawWaveCharts(self, chart_1_data, chart_2_data, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2):

    self.debug.info_message("drawWaveCharts")

    self.saved_data_for_chart_1 = chart_1_data
    self.saved_data_for_chart_2 = chart_2_data

    try:
      peak_y = 2
      y_offset = 0.5
      #y_offset = 0.0
      self.debug.info_message("peak y: " + str(peak_y))
      last_y1 = 0
      last_y2 = 0
      last_x = 0
      len_chart_1_data = len(chart_1_data)
      self.debug.info_message("len_chart_1_data: " + str(len_chart_1_data))
      max_x_index = int(len(chart_1_data)/x_magnifier1)
      min_x_index = max_x_index * (x_magnifier2 / 100)
      self.debug.info_message("max_x_index: " + str(max_x_index))

      step_increment = int(1000 / (max_x_index - min_x_index))
      self.debug.info_message("step_increment: " + str(step_increment))

      graph1 = self.osmod.form_gui.window['graph_wavedata1']
      graph2 = self.osmod.form_gui.window['graph_wavedata2']

      graph1.erase()
      graph2.erase()

      for x in range (0,1000, max(step_increment, 1) ):
        x_value = x 
        x_index  = int((((x + (x_magnifier2 *10))/1000) * len(chart_1_data))/x_magnifier1)
        y1_value = (((chart_1_data[ x_index ] / peak_y)*y_magnifier1) + y_offset) * 250 
        y2_value = (((chart_2_data[ x_index ] / peak_y)*y_magnifier2) + y_offset) * 250 

        self.osmod.form_gui.window['graph_wavedata1'].draw_line(point_from=(last_x,last_y1), point_to=(x_value,y1_value), width=2, color='black')
        self.osmod.form_gui.window['graph_wavedata2'].draw_line(point_from=(last_x,last_y2), point_to=(x_value,y2_value), width=2, color='black')

        last_y1 = y1_value
        last_y2 = y2_value
        last_x = x_value

    except:
      self.debug.error_message("Exception in drawWaveCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed drawWaveCharts: ")

  """ This method works fine"""
  def getStrongestFrequency(self, data):
    self.debug.info_message("getStrongestFrequency")
    try:
      fft_output = np.fft.fft(data)
      frequencies = np.fft.fftfreq(len(data), 1/self.osmod.sample_rate)
      positive_frequency_indices = np.where((frequencies > 0) & (frequencies < 3000))[0]
      fft_magnitude = np.abs(fft_output)[positive_frequency_indices]

      frequencies = frequencies[positive_frequency_indices]
      strongest_index = np.argmax(fft_magnitude)
      strong_freqs = frequencies[strongest_index]
      strong_magnitudes = fft_magnitude[strongest_index]

      sys.stdout.write("strongest index: " + str(strongest_index) + "\n")
      sys.stdout.write("strong_freqs: " + str(strong_freqs) + "\n")
      sys.stdout.write("strong_magnitudes: " + str(strong_magnitudes) + "\n")
  
    except:
      self.debug.error_message("Exception in getStrongestFrequency: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed getStrongestFrequency: ")

    return strong_freqs

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

  """ Eb/N0 = SNR(dB) + 10 * log10(Bandwidth / bit_rate)   """
  def calculate_EbN0(self, signal, signal_frequency, numbits, bit_rate, noise_free_signal):
    self.debug.info_message("calculateSNR_EbN0")

    gc.collect()

    try:
      t = np.arange(0, 1, 1/self.osmod.sample_rate)
      fft_output = np.fft.fft(noise_free_signal)
      psd = np.abs(fft_output)**2
      frequencies = np.fft.fftfreq(len(fft_output), 1/self.osmod.sample_rate)

      freq_low_signal  = signal_frequency[0] - 2
      freq_low_signal_hi  = signal_frequency[0] + 2
      freq_high_signal = signal_frequency[1] + 2
      freq_high_signal_lo = signal_frequency[1] - 2
      freq_indices = np.where(((frequencies >= freq_low_signal) & (frequencies <= freq_low_signal_hi)) | ((frequencies >= freq_high_signal_lo) & (frequencies <= freq_high_signal)) )[0]
      signal_psd = np.abs(fft_output[freq_indices])**2
      self.debug.info_message("signal_psd: " + str(signal_psd) )

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
      self.debug.error_message("Exception in getStrongestFrequency: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return filtered_signal


  """ bandpass filter fft"""
  def bandpass_filter_fft(self, signal, freq_lo, freq_hi):
    self.debug.info_message("bandpass_filter_fft")

    try:
      fft_signal  = sp.fft.fft(signal)
      frequencies = sp.fft.fftfreq(len(signal), 1/self.osmod.sample_rate)
      mask         = (np.abs(frequencies) >= freq_lo) & (np.abs(frequencies) <= freq_hi)
      fft_filtered = fft_signal * mask
      filtered_signal = sp.fft.ifft(fft_filtered)
    except:
      self.debug.error_message("Exception in getStrongestFrequency: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return filtered_signal


