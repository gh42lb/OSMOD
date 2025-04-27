import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
import matplotlib.pyplot as plt
from numpy import pi
from scipy.signal import butter, filtfilt, firwin
from modem_core_utils import ModemCoreUtils
from modulators import ModulatorPSK
from demodulators import DemodulatorPSK
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

"""
possible modulation schemes...

FSK with PSK: 2 fsk carriers + 8psk for 64 bit encoding....
Cycle FSK carriers in fixed sequence as follows...
  
2FSK * 8PSK (64 bit)

|
 |

2FSK * 4PSK (64 bit)

|
 |
 |

2FSK * BPSK (64 bit)
|
|
|
|
 |
 |
 |
 |

3FSK * 4PSK (64 bit)

|
 |
  |

3FSK * BPSK (64 bit)
|
|
 |
 |
  |
  |


2FSK * 4PSK (256 bit)

|
|
 |
 |


3FSK * 4PSK (256 bit)

|
  |
 |
  |


3FSK * BPSK (256 bit)
|
|
  |
  |
 |
 |
  |
  |


"""


class mod_2FSK8PSK(ModulatorPSK):
  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MOD)
    self.debug.info_message("__init__")
    super().__init__(osmod)


  def modulate_2fsk_8psk_optimized(self, frequency, bit_sequence):
    self.debug.info_message("modulate_2fsk_8psk_optimized: ")

    """ call common routine based on 3 bits per grouping"""
    return self.modulate_2fsk_npsk_optimized(frequency, bit_sequence, 3, 2)

  def modulate(self, frequency, bit_triplets):
    self.debug.info_message("modulate: ")

    try:
      """ modulate the signal """
      data2 = np.array([])
      for triplet1, triplet2 in zip(bit_triplets[0], bit_triplets[1]):
        self.debug.info_message("modulating triplet: " + str(triplet1))
        self.debug.info_message("modulating triplet: " + str(triplet2))
        combined = self.osmod.mod_2fsk8psk.modulate_2fsk_8psk_optimized(frequency, [triplet1, triplet2])
        data2 = np.append(data2, combined)

      return data2

    except:
      self.debug.error_message("Exception in modulate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



class demod_2FSK8PSK(DemodulatorPSK):
  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_DEMOD)
    self.debug.info_message("__init__")
    super().__init__(osmod)

  def demodulate_2fsk_8psk(self, audio_block, frequency):
    self.debug.info_message("demodulate_2fsk_8psk")

    """ reset the block_start index for each set of plocks sent for processing i.e. here!"""
    self.block_start_candidates = []

    where1 = int(self.osmod.symbol_block_size* (self.osmod.extraction_points[0]))
    where2 = int(self.osmod.symbol_block_size* (self.osmod.extraction_points[1]))

    decoded_bitstring_1, decoded_bitstring_2 = self.demodulate_2x8psk_common(audio_block, frequency, where1, where2, True)

    return decoded_bitstring_1, decoded_bitstring_2


  def demodulate_2x8psk_common(self, audio_block, frequency, where1, where2, display):
    self.debug.info_message("demodulate_2x8psk_common")

    try:
      pre_signal = audio_array = np.append(self.remainder, audio_block)

      self.osmod.startTimer('demodulate_2x8psk_common')
      if self.osmod.phase_extraction == ocn.EXTRACT_INTERPOLATE:
        """ apply receive pre filters"""

        """ calculate receive side RRC filter"""
        self.debug.info_message("gettting pulse_start_index")
        pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
        pulse_start_index = int(self.findPulseStartIndex(audio_array) + pulse_length/2)

        self.debug.info_message("pulse_start_index: " + str(pulse_start_index))
        median_block_offset = pulse_start_index
        median_block_offset_sig2 = median_block_offset

        self.debug.info_message("elapsed time for findPulseStartIndex: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

        self.debug.info_message("finding interpolated pulses")
        interpolated_lower, interpolated_higher = self.receive_pre_filters_interpolate(pulse_start_index, audio_array, frequency)
        self.debug.info_message("block_start_candidates: " + str(self.block_start_candidates))

        self.debug.info_message("elapsed time for receive_pre_filters_interpolate: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

        self.debug.info_message("fft filtering data")
        fft_filtered_lower, fft_filtered_higher = self.receive_pre_filters_filter_wave(pulse_start_index, audio_array, frequency)
        audio_array1 = fft_filtered_lower
        audio_array2 = fft_filtered_higher

        if display:
          self.debug.info_message("charting data")
          self.drawWaveCharts(fft_filtered_lower, fft_filtered_higher, 150, 150, 22, 22)


        self.debug.info_message("elapsed time for receive_pre_filters_filter_wave: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

        """ convert to baseband """
        self.debug.info_message("converting to baseband")
        if self.osmod.baseband_conversion == 'costas_loop':
          recovered_signal1, error_1 = self.recoverBasebandSignalOptimized(frequency[0], audio_array1)
          recovered_signal2, error_2 = self.recoverBasebandSignalOptimized(frequency[1], audio_array2)

          self.debug.info_message("elapsed time for recoverBasebandSignalOptimized: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          self.debug.info_message("averaging the phases")
          temp, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1[0], recovered_signal2[0], frequency, interpolated_lower, interpolated_higher)
          recovered_signal1a = temp[0] 
          recovered_signal2a = temp[1]
          temp, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1[1], recovered_signal2[1], frequency, interpolated_lower, interpolated_higher)
          recovered_signal1b = temp[0] 
          recovered_signal2b = temp[1]

          self.debug.info_message("elapsed time for receive_pre_filters_average_data: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          self.debug.info_message("detecting offset")
          """ detector to recover signal offset."""
          median_block_offset = self.detectSampleOffset(recovered_signal1[0])
          self.debug.info_message("detected offset: " + str(median_block_offset))
          median_block_offset_sig2 = median_block_offset
          self.debug.info_message("elapsed time for detectSampleOffset: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

          """ calculate remainder to be tacked on front of next sample"""
          remainder_start = ((len(audio_array) - pulse_start_index) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
          self.remainder = pre_signal[remainder_start::]

          """ extract the baseband samples and convert constellation values"""
          where1_pulse = int(np.median(interpolated_lower))
          where2_pulse = int(np.median(interpolated_higher))
          where1 = where1_pulse * pulse_length
          where2 = where2_pulse * pulse_length

          decoded_values1 = self.extractPhaseValuesWithOffsetDouble(recovered_signal1[0], recovered_signal1[1], pulse_start_index, frequency[0], self.osmod.parameters[4], where1)
          self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          decoded_values1_real = self.convertToInt(decoded_values1[0], self.osmod.parameters[0], 0.0)
          decoded_values1_imag = self.convertToInt(decoded_values1[1], self.osmod.parameters[0], 0.0)

          decoded_values2 = self.extractPhaseValuesWithOffsetDouble(recovered_signal2[0], recovered_signal2[1], pulse_start_index, frequency[1], self.osmod.parameters[4], where2)
          self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          decoded_values2_real = self.convertToInt(decoded_values2[0], self.osmod.parameters[0], 0.0)
          decoded_values2_imag = self.convertToInt(decoded_values2[1], self.osmod.parameters[0], 0.0)
    
      elif self.osmod.phase_extraction == ocn.EXTRACT_NORMAL:

        """ convert to baseband """
        recovered_signal1, error_1 = self.recoverBasebandSignalOptimized(frequency[0], audio_array)
        recovered_signal2, error_2 = self.recoverBasebandSignalOptimized(frequency[1], audio_array)

        """ detector to recover signal offset."""
        median_block_offset = self.detectSampleOffset(recovered_signal1[0])
        median_block_offset_sig2 = median_block_offset

        """ calculate remainder to be tacked on front of next sample"""
        remainder_start = ((len(audio_array) - median_block_offset) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
        self.remainder = pre_signal[remainder_start::]

        """ extract the baseband samples and convert constellation values"""
        decoded_values1 = self.extractPhaseValuesWithOffsetDouble(recovered_signal1[0], recovered_signal1[1], median_block_offset, frequency[0], self.osmod.parameters[4], where1)
        self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
        decoded_values1_real = self.convertToInt(decoded_values1[0], self.osmod.parameters[0], 0.0)
        decoded_values1_imag = self.convertToInt(decoded_values1[1], self.osmod.parameters[0], 0.0)
        decoded_values2 = self.extractPhaseValuesWithOffsetDouble(recovered_signal2[0], recovered_signal2[1], median_block_offset_sig2, frequency[1], self.osmod.parameters[4], where2)
        self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
        decoded_values2_real = self.convertToInt(decoded_values2[0], self.osmod.parameters[0], 0.0)
        decoded_values2_imag = self.convertToInt(decoded_values2[1], self.osmod.parameters[0], 0.0)


      self.debug.info_message("elapsed time for convertToInt: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

      """ convert constellation values to text"""

      decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag)

      if display:
        """ display baseband signals """
        self.debug.info_message("charting data")
        if self.osmod.phase_extraction == ocn.EXTRACT_NORMAL:
          self.displayChartResults(recovered_signal1[0], recovered_signal1[1])
        elif self.osmod.phase_extraction == ocn.EXTRACT_INTERPOLATE:
          self.debug.info_message("phase error history: " + str(error_1))
          self.displayChartResults(recovered_signal1[0], recovered_signal1[1])

      return decoded_bitstring_1, decoded_bitstring_2

    except:
      self.debug.error_message("Exception in demodulate_2x8psk_common: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


