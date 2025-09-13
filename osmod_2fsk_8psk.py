import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
import matplotlib.pyplot as plt
import gc

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
      gc.collect()

      """ modulate the signal """
      data2 = np.array([])
      for triplet1, triplet2 in zip(bit_triplets[0], bit_triplets[1]):
        #self.debug.info_message("modulating triplet: " + str(triplet1))
        #self.debug.info_message("modulating triplet: " + str(triplet2))
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

    gc.collect()

    """ reset the block_start index for each set of plocks sent for processing i.e. here!"""
    self.block_start_candidates = []

    where1 = int(self.osmod.symbol_block_size* (self.osmod.extraction_points[0]))
    where2 = int(self.osmod.symbol_block_size* (self.osmod.extraction_points[1]))

    decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = self.demodulate_2x8psk_common(audio_block, frequency, where1, where2, True)

    return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec



  def demodulate_2x8psk_common(self, audio_block, frequency, where1, where2, display):
    self.debug.info_message("demodulate_2x8psk_common")

    if self.osmod.pulse_detection == ocn.PULSE_DETECTION_I3 and self.osmod.baseband_conversion == 'I3_rel_exp':
      decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = self.demodulate_2x8psk_common_I3_Relative_Phase(audio_block, frequency, where1, where2, display)
      return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec
    else:
      decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = self.demodulate_2x8psk_common_base(audio_block, frequency, where1, where2, display)
      return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec

  """ demodulate base modes...I and N """
  def demodulate_2x8psk_common_base(self, audio_block, frequency, where1, where2, display):
    self.debug.info_message("demodulate_2x8psk_common")

    self.osmod.getDurationAndReset('init')

    try:
      fine_tune_adjust    = [0] * 2
      fine_tune_adjust[0] = 0
      fine_tune_adjust[1] = 0

      def averageDataTriple(fft_filtered_lower, fft_filtered_higher):
        nonlocal binary_array_post_fec

        test_lower_real  = fft_filtered_lower.copy().real
        test_lower_imag  = fft_filtered_lower.copy().imag
        test_higher_real = fft_filtered_higher.copy().real
        test_higher_imag = fft_filtered_higher.copy().imag

        self.chart_data_dict['baseband_lower_complex']  = fft_filtered_lower.copy()
        self.chart_data_dict['baseband_higher_complex'] = fft_filtered_higher.copy()
        self.chart_data_dict['baseband_lower_real']     = test_lower_real.copy()
        self.chart_data_dict['baseband_lower_imag']     = test_lower_imag.copy() 
        self.chart_data_dict['baseband_higher_real']    = test_higher_real.copy() 
        self.chart_data_dict['baseband_higher_imag']    = test_higher_imag.copy() 

        self.receive_pre_filters_average_data_intra_triple(pulse_start_index, fft_filtered_lower, fft_filtered_higher, None, interpolated_lower, interpolated_higher, fine_tune_adjust)

        self.chart_data_dict['averaged_lower_real'] = fft_filtered_lower.real
        self.chart_data_dict['averaged_lower_imag'] = fft_filtered_lower.imag
        self.chart_data_dict['averaged_higher_real'] = fft_filtered_higher.real
        self.chart_data_dict['averaged_higher_imag'] = fft_filtered_higher.imag
        self.chart_data_dict['averaged_lower_complex'] = fft_filtered_lower
        self.chart_data_dict['averaged_higher_complex'] = fft_filtered_higher

        inter_pulse_phase_delta_lower  = self.osmod.test.calculateInterPulsePhaseDelta(frequency[0], len(interpolated_lower))
        inter_pulse_phase_delta_higher = self.osmod.test.calculateInterPulsePhaseDelta(frequency[1], len(interpolated_higher))
        self.debug.info_message("inter_pulse_phase_delta_lower: "  + str(inter_pulse_phase_delta_lower))
        self.debug.info_message("inter_pulse_phase_delta_higher: " + str(inter_pulse_phase_delta_higher))


        where1_pulse_a = int(np.median(interpolated_lower))
        where1_pulse_b = int(np.min(interpolated_lower))
        where1_pulse_c = int(np.max(interpolated_lower))
        where2_pulse_a = int(np.median(interpolated_higher))
        where2_pulse_b = int(np.min(interpolated_higher))
        where2_pulse_c = int(np.max(interpolated_higher))

        intlist_lower, decoded_bitstring_1, decoded_intvalues1, intra_triple_charts = self.extractPhaseValuesIntraTriple(fft_filtered_lower, pulse_start_index, frequency[0], pulse_length, where1_pulse_a * pulse_length, where1_pulse_b * pulse_length, where1_pulse_c * pulse_length, fine_tune_adjust, 0, residuals)
        self.chart_data_dict['smoothed_angle_a_lower'] = intra_triple_charts[0]
        self.chart_data_dict['smoothed_angle_b_lower'] = intra_triple_charts[1]
        self.chart_data_dict['smoothed_angle_c_lower'] = intra_triple_charts[2]
        self.chart_data_dict['smoothed_a_real_lower']  = intra_triple_charts[3]
        self.chart_data_dict['smoothed_b_real_lower']  = intra_triple_charts[4]
        self.chart_data_dict['smoothed_c_real_lower']  = intra_triple_charts[5]
        self.chart_data_dict['smoothed_a_imag_lower']  = intra_triple_charts[6]
        self.chart_data_dict['smoothed_b_imag_lower']  = intra_triple_charts[7]
        self.chart_data_dict['smoothed_c_imag_lower']  = intra_triple_charts[8]

        intlist_higher, decoded_bitstring_2, decoded_intvalues2, intra_triple_charts = self.extractPhaseValuesIntraTriple(fft_filtered_higher, pulse_start_index, frequency[1], pulse_length, where2_pulse_a * pulse_length, where2_pulse_b * pulse_length, where2_pulse_c * pulse_length, fine_tune_adjust, 1, residuals)
        self.chart_data_dict['smoothed_angle_a_higher'] = intra_triple_charts[0]
        self.chart_data_dict['smoothed_angle_b_higher'] = intra_triple_charts[1]
        self.chart_data_dict['smoothed_angle_c_higher'] = intra_triple_charts[2]
        self.chart_data_dict['smoothed_a_real_higher']  = intra_triple_charts[3]
        self.chart_data_dict['smoothed_b_real_higher']  = intra_triple_charts[4]
        self.chart_data_dict['smoothed_c_real_higher']  = intra_triple_charts[5]
        self.chart_data_dict['smoothed_a_imag_higher']  = intra_triple_charts[6]
        self.chart_data_dict['smoothed_b_imag_higher']  = intra_triple_charts[7]
        self.chart_data_dict['smoothed_c_imag_higher']  = intra_triple_charts[8]

        binary_array_post_fec = self.displayTextFromIntlist(intlist_lower, intlist_higher)
        #decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_intvalues1[0], decoded_intvalues1[1], decoded_intvalues2[0], decoded_intvalues2[1])


        return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec

      def averageDataSingle(recovered_signal1, recovered_signal2):
        phase_data_before_averaging1 = recovered_signal1[0].copy()
        phase_data_before_averaging2 = recovered_signal1[1].copy()

        self.debug.info_message("averaging the phases")
        temp, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1[0], recovered_signal2[0], frequency, interpolated_lower, interpolated_higher)
        recovered_signal1a = temp[0] 
        recovered_signal2a = temp[1]
        temp, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1[1], recovered_signal2[1], frequency, interpolated_lower, interpolated_higher)
        recovered_signal1b = temp[0] 
        recovered_signal2b = temp[1]

        self.osmod.getDurationAndReset('receive_pre_filters_average_data')
        self.osmod.getDurationAndReset('detectSampleOffset')

        """ calculate remainder to be tacked on front of next sample"""
        remainder_start = ((len(audio_array) - pulse_start_index) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
        self.remainder = pre_signal[remainder_start::]

        """ extract the baseband samples and convert constellation values"""
        where1_pulse = int(np.median(interpolated_lower))
        where2_pulse = int(np.median(interpolated_higher))
        where1 = where1_pulse * pulse_length
        where2 = where2_pulse * pulse_length

        decoded_values1 = self.extractPhaseValuesWithOffsetDouble(recovered_signal1[0], recovered_signal1[1], pulse_start_index, frequency[0], self.osmod.parameters[4], where1)
        decoded_values1_real = self.convertToInt(decoded_values1[0], self.osmod.parameters[0], 0.0)
        decoded_values1_imag = self.convertToInt(decoded_values1[1], self.osmod.parameters[0], 0.0)

        decoded_values2 = self.extractPhaseValuesWithOffsetDouble(recovered_signal2[0], recovered_signal2[1], pulse_start_index, frequency[1], self.osmod.parameters[4], where2)
        decoded_values2_real = self.convertToInt(decoded_values2[0], self.osmod.parameters[0], 0.0)
        decoded_values2_imag = self.convertToInt(decoded_values2[1], self.osmod.parameters[0], 0.0)

        self.osmod.getDurationAndReset('extractPhaseValuesWithOffsetDouble')

        decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag)

        return decoded_bitstring_1, decoded_bitstring_2


      """ demodulation start..."""
      binary_array_post_fec = []

      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pre_signal = audio_array = np.append(self.remainder, audio_block)

      if self.osmod.phase_extraction == ocn.EXTRACT_INTERPOLATE:

        """ processing *before* fft """
        if self.osmod.pulse_detection == ocn.PULSE_DETECTION_I3:
          ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, 0, 0, ocn.LOCATE_PULSE_START_INDEX)
          pulse_start_index = ret_values[0]
          ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.CALC_PULSE_OFFSETS)
          pulse_offsets = ret_values[3]
          pulse_start_index, audio_array = self.osmod.detector.processAudioArrayPulses(audio_array, pulse_start_index, 0, 0, pulse_offsets)
          ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.CALC_BLOCK_OFFSETS)
          block_offsets = ret_values[2]
          pulse_start_index, audio_array = self.osmod.detector.processAudioArray(audio_array, pulse_start_index, 0, 0, block_offsets)
          #TEST DEBUG CODE
          self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.FIND_TRIPLET_MAX_POINT)



        fft_filtered = [None]*2
        fft_filtered[0], masked_fft_lower  = self.bandpass_filter_fft(audio_array, frequency[0] + self.osmod.fft_interpolate[0], frequency[0] + self.osmod.fft_interpolate[1])
        fft_filtered[1], masked_fft_higher = self.bandpass_filter_fft(audio_array, frequency[1] + self.osmod.fft_interpolate[2], frequency[1] + self.osmod.fft_interpolate[3])

        """ processing *after* fft """
        if self.osmod.pulse_detection == ocn.PULSE_DETECTION_I3:
          persistent_lower, persistent_higher = self.osmod.interpolator.derivePersistentLists(pulse_start_index, fft_filtered, frequency)

        self.osmod.getDurationAndReset('findPulseStartIndex')

        self.debug.info_message("finding interpolated pulses")
        if self.osmod.pulse_detection == ocn.PULSE_DETECTION_NORMAL:
          """ detect pulses with correlation template"""
          pulse_start_index, test_1, test_3 = self.osmod.detector.pulseDetector(audio_array, frequency)

          """ find the max index in the interpolated lower list and align so that it is last index in the first half of block"""
          interpolated_lower, interpolated_higher = self.osmod.interpolator.receive_pre_filters_interpolate(pulse_start_index, audio_array, frequency, fft_filtered)
          min_index = np.min(interpolated_lower)
          shift_amount_index = min_index
          self.debug.info_message("shift_amount_index: " + str(shift_amount_index))
          shift_amount = shift_amount_index * pulse_length
          pulse_start_index = pulse_start_index + shift_amount

          median_block_offset = pulse_start_index
          median_block_offset_sig2 = median_block_offset
          interpolated_lower, interpolated_higher = self.osmod.interpolator.receive_pre_filters_interpolate(pulse_start_index, audio_array, frequency, fft_filtered)
          self.debug.info_message("shift_amount: " + str(shift_amount))

          self.debug.info_message("pulse_start_index: " + str(pulse_start_index))
          median_block_offset = pulse_start_index
          median_block_offset_sig2 = median_block_offset

          self.osmod.getDurationAndReset('receive_pre_filters_interpolate')
          self.debug.info_message("fft filtering data")
          fft_filtered_lower, fft_filtered_higher = self.receive_pre_filters_filter_wave(pulse_start_index, audio_array, frequency)


        elif self.osmod.pulse_detection == ocn.PULSE_DETECTION_I3:
          """ test routine to detect standing waves"""
          self.debug.info_message("*********************************************" )
          self.debug.info_message("non-filtered pulse_start_index: " + str(pulse_start_index) )

          self.debug.info_message("=============================================" )
          self.debug.info_message("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
          self.debug.info_message("persistent_lower: " + str(persistent_lower) )
          self.debug.info_message("persistent_higher: " + str(persistent_higher) )
          interpolated_lower, interpolated_higher, shift_amount = self.osmod.interpolator.interpolatePulseTrain([persistent_lower, persistent_higher])


          self.osmod.detector.calcPulseTrainSectionAngles(frequency, interpolated_lower, interpolated_higher)


          residuals = self.osmod.detector.calcResidualPhaseAngles(audio_array, frequency, pulse_offsets, interpolated_lower, interpolated_higher)

          pulse_start_index, audio_array = self.osmod.detector.processShiftAmount(audio_array, pulse_start_index, 0, shift_amount)

          self.debug.info_message("pulse_start_index: " + str(pulse_start_index))
          median_block_offset = pulse_start_index
          median_block_offset_sig2 = median_block_offset

          self.osmod.getDurationAndReset('receive_pre_filters_interpolate')
          self.debug.info_message("fft filtering data")
          fft_filtered_lower, fft_filtered_higher = self.receive_pre_filters_filter_wave(pulse_start_index, audio_array, frequency)


        audio_array1 = (fft_filtered_lower.real + fft_filtered_lower.imag ) / 2
        audio_array2 = (fft_filtered_higher.real + fft_filtered_higher.imag) / 2

        self.osmod.getDurationAndReset('receive_pre_filters_filter_wave')

        if display:
          self.debug.info_message("charting data")
          self.osmod.analysis.drawWaveCharts(fft_filtered_lower, fft_filtered_higher, 150, 150, 22, 22)

        self.osmod.getDurationAndReset('init')

        """ convert to baseband """
        if self.osmod.form_gui.window['cb_override_downconvertmethod'].get():
          self.osmod.baseband_conversion = self.osmod.form_gui.window['combo_downconvert_type'].get()

        self.debug.info_message("converting to baseband")
        if self.osmod.baseband_conversion == 'I3_rel_exp':
          self.debug.info_message("downconvert I3_rel_exp")
          self.debug.info_message("audio data type: " + str(fft_filtered_lower.dtype))
          self.debug.info_message("audio data type: " + str(fft_filtered_higher.dtype))

          self.downconvert_I3_RelExp(pulse_start_index, [fft_filtered_lower, fft_filtered_higher], frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)

          if self.osmod.getOptionalParam('phase_encoding') == ocn.PHASE_INTRA_TRIPLE:
            decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = averageDataTriple(fft_filtered_lower, fft_filtered_higher)
            #decoded_bitstring_1, decoded_bitstring_2 = averageDataSingle([fft_filtered_lower.real, fft_filtered_lower.imag], [fft_filtered_higher.real, fft_filtered_higher.imag])

          """ calculate remainder to be tacked on front of next sample"""
          remainder_start = ((len(audio_array) - pulse_start_index) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
          self.remainder = pre_signal[remainder_start::]

        elif self.osmod.baseband_conversion == 'costas_loop':
          recovered_signal1, error_1 = self.recoverBasebandSignalOptimized(frequency[0], audio_array1)
          recovered_signal2, error_2 = self.recoverBasebandSignalOptimized(frequency[1], audio_array2)
          self.osmod.getDurationAndReset('recoverBasebandSignalOptimized')

          decoded_bitstring_1, decoded_bitstring_2 = averageDataSingle(recovered_signal1, recovered_signal2)

        elif self.osmod.baseband_conversion == 'fast':
          """ This process is 100 x faster"""
          error_1 = np.zeros_like(audio_array1)
          error_2 = np.zeros_like(audio_array1)
          complex1 = audio_array1.astype(complex)
          complex2 = audio_array2.astype(complex)


          phase_offset = (1*(2*np.pi))/8
          sig1_normal = self.downconvert8pskToBaseband(frequency[0], audio_array1, 0)
          sig1_rotate = self.downconvert8pskToBaseband(frequency[0], audio_array1, phase_offset)
          sig2_normal = self.downconvert8pskToBaseband(frequency[1], audio_array2, 0)
          sig2_rotate = self.downconvert8pskToBaseband(frequency[1], audio_array2, phase_offset)

          frequency_test_sig1_low     = self.downconvert8pskToBaseband(frequency[0]       , audio_array1, 0)
          frequency_test_sig1_low_45  = self.downconvert8pskToBaseband(frequency[0]       , audio_array1, 45)
          frequency_test_sig2_low  = self.downconvert8pskToBaseband(frequency[0] - 0.01, audio_array1, 0)
          frequency_test_sig3_low  = self.downconvert8pskToBaseband(frequency[0] - 0.02, audio_array1, 0)
          frequency_test_sig4_low  = self.downconvert8pskToBaseband(frequency[0] + 0.01, audio_array1, 0)
          frequency_test_sig1_high    = self.downconvert8pskToBaseband(frequency[1]       , audio_array2, 0)
          frequency_test_sig1_high_45 = self.downconvert8pskToBaseband(frequency[1]       , audio_array2, 45)
          frequency_test_sig2_high = self.downconvert8pskToBaseband(frequency[1] - 0.01, audio_array2, 0)
          frequency_test_sig3_high = self.downconvert8pskToBaseband(frequency[1] - 0.02, audio_array2, 0)
          frequency_test_sig4_high = self.downconvert8pskToBaseband(frequency[1] + 0.01, audio_array2, 0)

          if self.osmod.getOptionalParam('phase_encoding') == ocn.PHASE_INTRA_SINGLE:
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig1_low.real, frequency_test_sig1_high.real, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig1_low.imag, frequency_test_sig1_high.imag, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig2_low.real, frequency_test_sig2_high.real, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig2_low.imag, frequency_test_sig2_high.imag, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig3_low.real, frequency_test_sig3_high.real, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig3_low.imag, frequency_test_sig3_high.imag, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig4_low.real, frequency_test_sig4_high.real, frequency, interpolated_lower, interpolated_higher)
            self.receive_pre_filters_average_data(pulse_start_index, frequency_test_sig4_low.imag, frequency_test_sig4_high.imag, frequency, interpolated_lower, interpolated_higher)
          elif self.osmod.getOptionalParam('phase_encoding') == ocn.PHASE_INTRA_TRIPLE:
            #test_lower_1  = frequency_test_sig1_low.real.copy()
            #test_lower_2  = frequency_test_sig1_low.imag.copy()
            #test_higher_1 = frequency_test_sig1_high.real.copy()
            #test_higher_2 = frequency_test_sig1_high.imag.copy()
            #self.receive_pre_filters_average_data_intra_triple(pulse_start_index, frequency_test_sig1_low_90.real, frequency_test_sig1_high_90.real, frequency, interpolated_lower, interpolated_higher)
            #self.receive_pre_filters_average_data_intra_triple(pulse_start_index, frequency_test_sig1_low_90.imag, frequency_test_sig1_high_90.imag, frequency, interpolated_lower, interpolated_higher)

            test_lower_1  = frequency_test_sig1_low.copy().real
            test_lower_2  = frequency_test_sig1_low.copy().imag
            test_higher_1 = frequency_test_sig1_high.copy().real
            test_higher_2 = frequency_test_sig1_high.copy().imag
            self.receive_pre_filters_average_data_intra_triple(pulse_start_index, test_lower_1, test_higher_1, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)
            self.receive_pre_filters_average_data_intra_triple(pulse_start_index, test_lower_2, test_higher_2, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)

            test_lower_1_45  = frequency_test_sig1_low_45.copy().real
            test_lower_2_45  = frequency_test_sig1_low_45.copy().imag
            test_higher_1_45 = frequency_test_sig1_high_45.copy().real
            test_higher_2_45 = frequency_test_sig1_high_45.copy().imag
            self.receive_pre_filters_average_data_intra_triple(pulse_start_index, test_lower_1_45, test_higher_1_45, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)
            self.receive_pre_filters_average_data_intra_triple(pulse_start_index, test_lower_2_45, test_higher_2_45, frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)

            where1_pulse_a = int(np.median(interpolated_lower))
            self.debug.info_message("where1_pulse_a: " + str(where1_pulse_a))
            #where1_pulse_b = where1_pulse_a + 1
            where1_pulse_b = int(np.min(interpolated_lower))
            self.debug.info_message("where1_pulse_b: " + str(where1_pulse_b))
            #where1_pulse_c = where1_pulse_a - 1
            where1_pulse_c = int(np.max(interpolated_lower))
            self.debug.info_message("where1_pulse_c: " + str(where1_pulse_c))
            where2_pulse_a = int(np.median(interpolated_higher))
            self.debug.info_message("where2_pulse_a: " + str(where2_pulse_a))
            #where2_pulse_b = where2_pulse_a + 1
            where2_pulse_b = int(np.min(interpolated_higher))
            self.debug.info_message("where2_pulse_b: " + str(where2_pulse_b))
            #where2_pulse_c = where2_pulse_a - 1
            where2_pulse_c = int(np.max(interpolated_higher))
            self.debug.info_message("where2_pulse_c: " + str(where2_pulse_c))
            new_complex_wave_low  = test_lower_1 + 1j * test_lower_2
            new_complex_wave_high = test_higher_1 + 1j * test_higher_2


            #self.locateStartSequence([test_lower_1, test_lower_2], [test_higher_1, test_higher_2], pulse_start_index, frequency, self.osmod.parameters[4], where1_pulse_a, where2_pulse_a)


            diff_codes_group_1, diff_codes_group_2, decoded_intvalues1, intra_triple_charts = self.extractPhaseValuesIntraTriple(new_complex_wave_low, pulse_start_index, frequency[0], pulse_length, where1_pulse_a * pulse_length, where1_pulse_b * pulse_length, where1_pulse_c * pulse_length, fine_tune_adjust, 0, residuals)
            self.debug.info_message("diff_codes_group_1 low: " + str(diff_codes_group_1))
            self.osmod.form_gui.window['ml_txrx_recvtext'].print("  diff_codes_group_1 low: " + str(diff_codes_group_1), end="", text_color='red', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_recvtext'].print("  diff_codes_group_2 low: " + str(diff_codes_group_2), end="", text_color='red', background_color = 'white')
            diff_codes_group_1, diff_codes_group_2, decoded_intvalues2, intra_triple_charts = self.extractPhaseValuesIntraTriple(new_complex_wave_high, pulse_start_index, frequency[1], pulse_length, where2_pulse_a * pulse_length, where2_pulse_b * pulse_length, where2_pulse_c * pulse_length, fine_tune_adjust, 1, residuals)
            self.debug.info_message("diff_codes_group_1 high: " + str(diff_codes_group_1))
            self.osmod.form_gui.window['ml_txrx_recvtext'].print("  diff_codes_group_1 high: " + str(diff_codes_group_1), end="", text_color='red', background_color = 'white')
            self.osmod.form_gui.window['ml_txrx_recvtext'].print("  diff_codes_group_2 high: " + str(diff_codes_group_2), end="", text_color='red', background_color = 'white')



            #decoded_values1a_normal, decoded_values1b_normal = self.extractPhaseValuesWithOffsetDouble(frequency_test_sig1_low.real, frequency_test_sig1_low.imag, pulse_start_index, frequency[0], self.osmod.parameters[4], where1_pulse_a * pulse_length)
            #self.debug.info_message("1st triplet decode a: " + str(decoded_values1a_normal))
            #self.debug.info_message("1st triplet decode b: " + str(decoded_values1b_normal))
            #decoded_values1a_normal, decoded_values1b_normal = self.extractPhaseValuesWithOffsetDouble(frequency_test_sig1_low.real, frequency_test_sig1_low.imag, pulse_start_index, frequency[0], self.osmod.parameters[4], where1_pulse_b * pulse_length)
            #self.debug.info_message("2nd triplet decode a: " + str(decoded_values1a_normal))
            #self.debug.info_message("2nd triplet decode b: " + str(decoded_values1b_normal))
            #decoded_values1a_normal, decoded_values1b_normal = self.extractPhaseValuesWithOffsetDouble(frequency_test_sig1_low.real, frequency_test_sig1_low.imag, pulse_start_index, frequency[0], self.osmod.parameters[4], where1_pulse_c * pulse_length)
            #self.debug.info_message("3rd triplet decode a: " + str(decoded_values1a_normal))
            #self.debug.info_message("3rd triplet decode b: " + str(decoded_values1b_normal))
            #decoded_values2a_normal, decoded_values2b_normal = self.extractPhaseValuesWithOffsetDouble(frequency_test_sig1_high.real, frequency_test_sig1_high.imag, pulse_start_index, frequency[0], self.osmod.parameters[4], where1_pulse_a * pulse_length)




          exp_lower  = sig1_normal.copy()
          exp_higher = sig2_normal.copy()
          test_chart_real, not_relevant = self.receive_pre_filters_average_data(pulse_start_index, exp_lower.real, exp_higher.real, frequency, interpolated_lower, interpolated_higher)
          test_chart_imag, not_relevant = self.receive_pre_filters_average_data(pulse_start_index, exp_lower.imag, exp_higher.imag, frequency, interpolated_lower, interpolated_higher)
          phase_data_before_averaging1 = sig1_normal.copy()
          phase_data_before_averaging2 = sig1_rotate.copy()


          max1 = np.max(np.abs(sig1_normal))
          max2 = np.max(np.abs(sig2_normal))
          max3 = np.max(np.abs(sig1_rotate))
          max4 = np.max(np.abs(sig2_rotate))
          ratio = max(max(max1, max2), max(max3, max4)) / self.osmod.parameters[3]
          sig1_normal = sig1_normal/ratio
          sig2_normal = sig2_normal/ratio
          sig1_rotate = sig1_rotate/ratio
          sig2_rotate = sig2_rotate/ratio

          recovered_signal1a_normal = sig1_normal.real
          recovered_signal1b_normal = sig1_normal.imag
          recovered_signal2a_normal = sig2_normal.real
          recovered_signal2b_normal = sig2_normal.imag
          recovered_signal1a_rotate = sig1_rotate.real
          recovered_signal1b_rotate = sig1_rotate.imag
          recovered_signal2a_rotate = sig2_rotate.real
          recovered_signal2b_rotate = sig2_rotate.imag

          self.debug.info_message("elapsed time for recoverBasebandSignalOptimized: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          self.debug.info_message("averaging the phases")
          #recovered_signal1a_normal, recovered_signal2a_normal, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1a_normal, recovered_signal2a_normal, frequency, interpolated_lower, interpolated_higher)
          recovered_signala_normal_avg, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1a_normal, recovered_signal2a_normal, frequency, interpolated_lower, interpolated_higher)
          #recovered_signal1b_normal, recovered_signal2b_normal, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1b_normal, recovered_signal2b_normal, frequency, interpolated_lower, interpolated_higher)
          recovered_signalb_normal_avg, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1b_normal, recovered_signal2b_normal, frequency, interpolated_lower, interpolated_higher)
          #recovered_signal1a_rotate, recovered_signal2a_rotate, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1a_rotate, recovered_signal2a_rotate, frequency, interpolated_lower, interpolated_higher)
          recovered_signala_rotate_avg, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1a_rotate, recovered_signal2a_rotate, frequency, interpolated_lower, interpolated_higher)
          #recovered_signal1b_rotate, recovered_signal2b_rotate, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1b_rotate, recovered_signal2b_rotate, frequency, interpolated_lower, interpolated_higher)
          recovered_signalb_rotate_avg, median_block_offset = self.receive_pre_filters_average_data(pulse_start_index, recovered_signal1b_rotate, recovered_signal2b_rotate, frequency, interpolated_lower, interpolated_higher)

          recovered_signal1a_normal = recovered_signala_normal_avg[0]
          recovered_signal2a_normal = recovered_signala_normal_avg[1]
          recovered_signal1b_normal = recovered_signalb_normal_avg[0]
          recovered_signal2b_normal = recovered_signalb_normal_avg[1]
          recovered_signal1a_rotate = recovered_signala_rotate_avg[0]
          recovered_signal2a_rotate = recovered_signala_rotate_avg[1]
          recovered_signal1b_rotate = recovered_signalb_rotate_avg[0]
          recovered_signal2b_rotate = recovered_signalb_rotate_avg[1]

          recovered_signal1a = recovered_signal1a_normal
          recovered_signal1b = recovered_signal1b_normal

          self.debug.info_message("elapsed time for receive_pre_filters_average_data: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          self.debug.info_message("detecting offset")
          """ detector to recover signal offset."""
          #median_block_offset = self.detectSampleOffset(recovered_signal1a_normal)
          #median_block_offset_sig2 = median_block_offset
          self.debug.info_message("elapsed time for detectSampleOffset: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))

          """ calculate remainder to be tacked on front of next sample"""
          remainder_start = ((len(audio_array) - pulse_start_index) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
          self.remainder = pre_signal[remainder_start::]

          """ extract the baseband samples and convert constellation values"""
          where1_pulse = int(np.median(interpolated_lower))
          where2_pulse = int(np.median(interpolated_higher))
          where1 = where1_pulse * pulse_length
          where2 = where2_pulse * pulse_length

          decoded_values1a_normal, decoded_values1b_normal = self.extractPhaseValuesWithOffsetDouble(recovered_signal1a_normal, recovered_signal1b_normal, pulse_start_index, frequency[0], self.osmod.parameters[4], where1)
          decoded_values1a_rotate, decoded_values1b_rotate = self.extractPhaseValuesWithOffsetDouble(recovered_signal1a_rotate, recovered_signal1b_rotate, pulse_start_index, frequency[0], self.osmod.parameters[4], where1)
          self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          decoded_values1_real, decoded_values1_imag = self.convertToIntMerge(decoded_values1a_normal, decoded_values1a_rotate, decoded_values1b_normal, decoded_values1b_rotate, self.osmod.parameters[0], 0.0)
          #decoded_values1_real = self.convertToInt(decoded_values1a_normal, self.osmod.parameters[0], 0.0)
          #decoded_values1_imag = self.convertToInt(decoded_values1b_normal, self.osmod.parameters[0], 0.0)


          decoded_values2a_normal, decoded_values2b_normal = self.extractPhaseValuesWithOffsetDouble(recovered_signal2a_normal, recovered_signal2b_normal, pulse_start_index, frequency[1], self.osmod.parameters[4], where2)
          decoded_values2a_rotate, decoded_values2b_rotate = self.extractPhaseValuesWithOffsetDouble(recovered_signal2a_rotate, recovered_signal2b_rotate, pulse_start_index, frequency[1], self.osmod.parameters[4], where2)
          self.debug.info_message("elapsed time for extractPhaseValuesWithOffsetDouble: " + str(self.osmod.getDurationAndReset('demodulate_2x8psk_common')))
          decoded_values2_real, decoded_values2_imag = self.convertToIntMerge(decoded_values2a_normal, decoded_values2a_rotate, decoded_values2b_normal, decoded_values2b_rotate, self.osmod.parameters[0], 0.0)
          #decoded_values2_real = self.convertToInt(decoded_values2a_normal, self.osmod.parameters[0], 0.0)
          #decoded_values2_imag = self.convertToInt(decoded_values2b_normal, self.osmod.parameters[0], 0.0)

          decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag)
    
      elif self.osmod.phase_extraction == ocn.EXTRACT_NORMAL:

        """ convert to baseband """
        recovered_signal1, error_1 = self.recoverBasebandSignalOptimized(frequency[0], audio_array)
        recovered_signal2, error_2 = self.recoverBasebandSignalOptimized(frequency[1], audio_array)

        self.osmod.getDurationAndReset('recoverBasebandSignalOptimized')

        """ detector to recover signal offset."""
        median_block_offset = self.detectSampleOffset(recovered_signal1[0])
        median_block_offset_sig2 = median_block_offset

        self.osmod.getDurationAndReset('detectSampleOffset')

        """ calculate remainder to be tacked on front of next sample"""
        remainder_start = ((len(audio_array) - median_block_offset) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
        self.remainder = pre_signal[remainder_start::]

        """ extract the baseband samples and convert constellation values"""
        decoded_values1 = self.extractPhaseValuesWithOffsetDouble(recovered_signal1[0], recovered_signal1[1], median_block_offset, frequency[0], self.osmod.parameters[4], where1)
        decoded_values1_real = self.convertToInt(decoded_values1[0], self.osmod.parameters[0], 0.0)
        decoded_values1_imag = self.convertToInt(decoded_values1[1], self.osmod.parameters[0], 0.0)
        decoded_values2 = self.extractPhaseValuesWithOffsetDouble(recovered_signal2[0], recovered_signal2[1], median_block_offset_sig2, frequency[1], self.osmod.parameters[4], where2)
        decoded_values2_real = self.convertToInt(decoded_values2[0], self.osmod.parameters[0], 0.0)
        decoded_values2_imag = self.convertToInt(decoded_values2[1], self.osmod.parameters[0], 0.0)

        self.osmod.getDurationAndReset('extractPhaseValuesWithOffsetDouble')

        decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag)


      """ convert constellation values to text"""

      #decoded_bitstring_1, decoded_bitstring_2 = self.displayTextResults(decoded_values1_real, decoded_values1_imag, decoded_values2_real, decoded_values2_imag)

      self.osmod.getDurationAndReset('displayTextResults')

      self.osmod.detector.writePulseTrainRotationDetails()


      enable_display = self.osmod.form_gui.window['cb_display_phases'].get()
      if enable_display:
        """ display baseband signals """
        self.debug.info_message("charting data")
        if self.osmod.phase_extraction == ocn.EXTRACT_NORMAL:
          self.displayChartResults(recovered_signal1[0], recovered_signal1[1], 'red', 'blue', True, 'mychart')
        elif self.osmod.phase_extraction == ocn.EXTRACT_INTERPOLATE:
          #self.debug.info_message("phase error history: " + str(error_1))

          chart_type = self.osmod.form_gui.window['option_chart_options'].get()
          if chart_type == 'Before Mean Average':
            self.displayChartResults(phase_data_before_averaging1, phase_data_before_averaging2, 'red', 'blue', True, 'mychart')
          elif chart_type == 'After Mean Average':
            self.displayChartResults(recovered_signal1a, recovered_signal1b, 'red', 'blue', True, 'mychart')
          elif chart_type == 'Both':
            self.displayChartResults(phase_data_before_averaging1, recovered_signal1a, 'red', 'blue', True, 'mychart')
          elif chart_type == 'FFT':
            self.displayChartResults(np.angle(masked_fft_lower), np.angle(masked_fft_higher), 'red', 'blue', True, 'mychart')
          elif chart_type == 'EXP':
            #self.displayChartResults(exp_lower.real, exp_lower.imag)
            self.displayChartResults(test_chart_real[0], test_chart_imag[0], 'red', 'blue', True, 'low frequency real(red) & imag(blue)', ocn.CHART_ONE)
            self.displayChartResults(np.angle(exp_lower), np.angle(exp_lower)%8, 'green', 'yellow', False, '', ocn.CHART_ONE)
            self.displayChartResults(test_chart_real[1], test_chart_imag[1], 'red', 'blue', True, 'high frequency real(red) & imag(blue)', ocn.CHART_TWO)
            self.displayChartResults(np.angle(exp_higher), np.angle(exp_higher)%8, 'green', 'yellow', False, '', ocn.CHART_TWO)
          elif chart_type == 'Phase Error':
            two_times_pi = np.pi * 2
            max_phase_value = two_times_pi / 8
            """ display ACTUAL phase errors"""
            self.displayChartResults(np.angle(exp_lower)%max_phase_value, np.angle(exp_higher)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_ONE)

            interpolated1 = self.interpolatePhaseDrift(exp_lower, median_block_offset, frequency[0], where1, ocn.PHASE_ERROR_ROUGH)
            interpolated2 = self.interpolatePhaseDrift(exp_higher, median_block_offset, frequency[1], where2, ocn.PHASE_ERROR_ROUGH)
            self.displayChartResults(interpolated1, interpolated2, 'red', 'blue', True, 'Phase Error Rough & Smooth', ocn.CHART_TWO)

            interpolated1 = self.interpolatePhaseDrift(exp_lower, median_block_offset, frequency[0], where1, ocn.PHASE_ERROR_SMOOTH)
            interpolated2 = self.interpolatePhaseDrift(exp_higher, median_block_offset, frequency[1], where2, ocn.PHASE_ERROR_SMOOTH)
            self.displayChartResults(interpolated1, interpolated2, 'red', 'blue', False, '', ocn.CHART_TWO)

            """ temporarily remove as time consuming for testing 
            self.displayChartResults(interpolated1 + interpolated2, interpolated1 + interpolated2, 'black', 'black', False, '', ocn.CHART_TWO)
            """
          elif chart_type == 'Frequency & EXP':
            #two_times_pi = np.pi * 2
            #max_phase_value = two_times_pi / 8
            """ display ACTUAL phase values"""
            #self.displayChartResults(np.angle(frequency_test_sig1_low)%max_phase_value, np.angle(frequency_test_sig1_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_ONE)
            #self.displayChartResults(np.angle(frequency_test_sig1_low), np.angle(frequency_test_sig1_high), 'red', 'blue', True, 'Phase Values Actual', ocn.CHART_ONE)

            #self.displayChartResults(np.angle(frequency_test_sig1_low), np.angle(frequency_test_sig1_low), 'red', 'red', True, 'downconverted data angle low', ocn.CHART_ONE_A)
            #self.displayChartResults(np.angle(frequency_test_sig1_high), np.angle(frequency_test_sig1_high), 'blue', 'blue', True, 'downconverted data angle high', ocn.CHART_TWO_A)

            complex_wave1 = test_lower_1 + 1j * test_lower_2
            complex_wave2 = test_higher_1 + 1j * test_higher_2
            self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, 'Angles Low (Red) & High (Blue)', ocn.CHART_ONE)
            self.displayChartResults(complex_wave1.real, complex_wave1.imag, 'red',  'blue', True, 'Real & Imag - Low(red,blue) - High(pink,cyan)', ocn.CHART_TWO)
            self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_TWO)

            complex_wave1 = test_lower_1_45 + 1j * test_lower_2_45
            complex_wave2 = test_higher_1_45 + 1j * test_higher_2_45
            self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, '45 - Angles Low (Red) & High (Blue)', ocn.CHART_THREE)
            self.displayChartResults(complex_wave1.real, complex_wave1.imag, 'red', 'blue', True, '45 - Real & Imag - Low(red,blue) - High(pink,cyan)', ocn.CHART_FOUR)
            self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_FOUR)


            #self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'blue', 'cyan', True, 'Phase Values Actual', ocn.CHART_FOUR)

            #self.displayChartResults(np.angle(frequency_test_sig1_low_90), np.angle(frequency_test_sig1_high_90), 'red', 'blue', True, 'Phase Values Actual 90', ocn.CHART_TWO)
            #self.displayChartResults(np.angle(frequency_test_sig2_low)%max_phase_value, np.angle(frequency_test_sig2_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_TWO)
            #self.displayChartResults(np.angle(frequency_test_sig3_low)%max_phase_value, np.angle(frequency_test_sig3_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_THREE)
            #self.displayChartResults(np.angle(frequency_test_sig4_low)%max_phase_value, np.angle(frequency_test_sig4_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_FOUR)

          elif chart_type == 'EXP Intra Triple':
            #complex_wave1 = test_lower_1 + 1j * test_lower_2
            #complex_wave2 = test_higher_1 + 1j * test_higher_2

            #self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, 'Angles Low (Red) & High (Blue), Real & Imag - Low(pink,cyan)', ocn.CHART_ONE)
            self.displayChartResults(np.angle(new_complex_wave_low), np.angle(new_complex_wave_high), 'red', 'blue', True, 'Angles Low (Red) & High (Blue)', ocn.CHART_ONE)
            self.displayChartResults(new_complex_wave_low.real, new_complex_wave_low.imag, 'red',  'blue', False, 'Real & Imag - Low(red,blue)', ocn.CHART_TWO)
            #self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_TWO)

            #self.displayChartResults(new_complex_wave_high.real, new_complex_wave_high.imag, 'red',  'blue', False, 'Real & Imag - High(red,blue)', ocn.CHART_THREE)

            self.displayChartResults(intra_triple_charts[0], intra_triple_charts[1], 'red',  'blue', True, 'Intra Extract Angles', ocn.CHART_THREE)
            self.displayChartResults(intra_triple_charts[2], intra_triple_charts[2], 'pink', 'cyan', False, '', ocn.CHART_THREE_A)

            self.displayChartResults(intra_triple_charts[3], intra_triple_charts[4], 'red',  'blue', True, 'Intra Extract Reference (red) Phase Corrected (blue, cyan)', ocn.CHART_FOUR)
            self.displayChartResults(intra_triple_charts[5], intra_triple_charts[5], 'pink', 'cyan', False, '', ocn.CHART_FOUR_A)

            #self.displayChartResults(intra_triple_charts[6], intra_triple_charts[7], 'red',  'blue', True, 'Intra Extract Diff/Total 1,2,3 (red, blue, pink)', ocn.CHART_FOUR)
            #self.displayChartResults(intra_triple_charts[8], intra_triple_charts[8], 'pink', 'cyan', False, '', ocn.CHART_FOUR_A)


          elif chart_type == 'Chart Data Dictionary':
            self.displayChartResults(self.chart_data_dict['averaged_lower_complex'].real, self.chart_data_dict['averaged_lower_complex'].imag, 'red', 'blue', True, 'Averaged Lower Real (Red) Imag (Blue) Higher Real (Pink) Imag (Cyan)', ocn.CHART_ONE)
            self.displayChartResults(self.chart_data_dict['averaged_higher_complex'].real, self.chart_data_dict['averaged_higher_complex'].imag, 'pink', 'cyan', False, '', ocn.CHART_ONE)
            self.displayChartResults(np.angle(self.chart_data_dict['averaged_lower_complex']), np.angle(self.chart_data_dict['averaged_higher_complex']), 'red', 'blue', True, 'Angles Lower (Red) & Higher (Blue)', ocn.CHART_TWO)
            self.displayChartResults(self.chart_data_dict['smoothed_a_real_lower'], self.chart_data_dict['smoothed_a_imag_lower'], 'red', 'blue', True, 'Smoothed Pulse A Lower Real (Red)  Imag (Blue) Higher Real (green) Imag (cyan)', ocn.CHART_THREE)
            self.displayChartResults(self.chart_data_dict['smoothed_a_real_higher'], self.chart_data_dict['smoothed_a_imag_higher'], 'green', 'cyan', False, '', ocn.CHART_THREE_A)
            self.displayChartResults(self.chart_data_dict['smoothed_angle_a_lower'], self.chart_data_dict['smoothed_angle_a_higher'], 'red', 'blue', True, 'Smoothed Pulse Angles Lower A (Red) Higher A (Blue)', ocn.CHART_FOUR)


      return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec

    except:
      self.debug.error_message("Exception in demodulate_2x8psk_common: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def demodulate_2x8psk_common_I3_Relative_Phase(self, audio_block, frequency, where1, where2, display):
    self.debug.info_message("demodulate_2x8psk_common_I3_Relative_Phase")

    self.osmod.getDurationAndReset('init')

    try:
      fine_tune_adjust    = [0] * 2
      fine_tune_adjust[0] = 0
      fine_tune_adjust[1] = 0

      def averageDataTriple(fft_filtered_lower, fft_filtered_higher):
        nonlocal binary_array_post_fec

        test_lower_real  = fft_filtered_lower.copy().real
        test_lower_imag  = fft_filtered_lower.copy().imag
        test_higher_real = fft_filtered_higher.copy().real
        test_higher_imag = fft_filtered_higher.copy().imag

        self.chart_data_dict['baseband_lower_complex']  = fft_filtered_lower.copy()
        self.chart_data_dict['baseband_higher_complex'] = fft_filtered_higher.copy()
        self.chart_data_dict['baseband_lower_real']     = test_lower_real.copy()
        self.chart_data_dict['baseband_lower_imag']     = test_lower_imag.copy() 
        self.chart_data_dict['baseband_higher_real']    = test_higher_real.copy() 
        self.chart_data_dict['baseband_higher_imag']    = test_higher_imag.copy() 

        self.receive_pre_filters_average_data_intra_triple(pulse_start_index, fft_filtered_lower, fft_filtered_higher, None, interpolated_lower, interpolated_higher, fine_tune_adjust)

        self.chart_data_dict['averaged_lower_real'] = fft_filtered_lower.real
        self.chart_data_dict['averaged_lower_imag'] = fft_filtered_lower.imag
        self.chart_data_dict['averaged_higher_real'] = fft_filtered_higher.real
        self.chart_data_dict['averaged_higher_imag'] = fft_filtered_higher.imag
        self.chart_data_dict['averaged_lower_complex'] = fft_filtered_lower
        self.chart_data_dict['averaged_higher_complex'] = fft_filtered_higher

        inter_pulse_phase_delta_lower  = self.osmod.test.calculateInterPulsePhaseDelta(frequency[0], len(interpolated_lower))
        inter_pulse_phase_delta_higher = self.osmod.test.calculateInterPulsePhaseDelta(frequency[1], len(interpolated_higher))
        self.debug.info_message("inter_pulse_phase_delta_lower: "  + str(inter_pulse_phase_delta_lower))
        self.debug.info_message("inter_pulse_phase_delta_higher: " + str(inter_pulse_phase_delta_higher))


        where1_pulse_a = int(np.median(interpolated_lower))
        where1_pulse_b = int(np.min(interpolated_lower))
        where1_pulse_c = int(np.max(interpolated_lower))
        where2_pulse_a = int(np.median(interpolated_higher))
        where2_pulse_b = int(np.min(interpolated_higher))
        where2_pulse_c = int(np.max(interpolated_higher))

        intlist_lower, decoded_bitstring_1, decoded_intvalues1, intra_triple_charts = self.extractPhaseValuesIntraTriple(fft_filtered_lower, pulse_start_index, frequency[0], pulse_length, where1_pulse_a * pulse_length, where1_pulse_b * pulse_length, where1_pulse_c * pulse_length, fine_tune_adjust, 0, residuals)
        self.chart_data_dict['smoothed_angle_a_lower'] = intra_triple_charts[0]
        self.chart_data_dict['smoothed_angle_b_lower'] = intra_triple_charts[1]
        self.chart_data_dict['smoothed_angle_c_lower'] = intra_triple_charts[2]
        self.chart_data_dict['smoothed_a_real_lower']  = intra_triple_charts[3]
        self.chart_data_dict['smoothed_b_real_lower']  = intra_triple_charts[4]
        self.chart_data_dict['smoothed_c_real_lower']  = intra_triple_charts[5]
        self.chart_data_dict['smoothed_a_imag_lower']  = intra_triple_charts[6]
        self.chart_data_dict['smoothed_b_imag_lower']  = intra_triple_charts[7]
        self.chart_data_dict['smoothed_c_imag_lower']  = intra_triple_charts[8]

        intlist_higher, decoded_bitstring_2, decoded_intvalues2, intra_triple_charts = self.extractPhaseValuesIntraTriple(fft_filtered_higher, pulse_start_index, frequency[1], pulse_length, where2_pulse_a * pulse_length, where2_pulse_b * pulse_length, where2_pulse_c * pulse_length, fine_tune_adjust, 1, residuals)
        self.chart_data_dict['smoothed_angle_a_higher'] = intra_triple_charts[0]
        self.chart_data_dict['smoothed_angle_b_higher'] = intra_triple_charts[1]
        self.chart_data_dict['smoothed_angle_c_higher'] = intra_triple_charts[2]
        self.chart_data_dict['smoothed_a_real_higher']  = intra_triple_charts[3]
        self.chart_data_dict['smoothed_b_real_higher']  = intra_triple_charts[4]
        self.chart_data_dict['smoothed_c_real_higher']  = intra_triple_charts[5]
        self.chart_data_dict['smoothed_a_imag_higher']  = intra_triple_charts[6]
        self.chart_data_dict['smoothed_b_imag_higher']  = intra_triple_charts[7]
        self.chart_data_dict['smoothed_c_imag_higher']  = intra_triple_charts[8]

        binary_array_post_fec = self.displayTextFromIntlist(intlist_lower, intlist_higher)

        return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec

      def decodeChunkCharacters(signal, interpolated, shift):
        nonlocal pulse_start_index
        nonlocal residuals

        self.osmod.detector.calcPulseTrainSectionAngles(frequency, interpolated[0], interpolated[1])
        residuals = self.osmod.detector.calcResidualPhaseAngles(signal, frequency, pulse_offsets, interpolated[0], interpolated[1])
        pulse_start_index, signal = self.osmod.detector.processShiftAmount(signal, pulse_start_index, 0, shift)

        fft_filtered_lower, fft_filtered_higher = self.receive_pre_filters_filter_wave(pulse_start_index, signal, frequency)

        self.downconvert_I3_RelExp(pulse_start_index, [fft_filtered_lower, fft_filtered_higher], frequency, interpolated[0], interpolated[1], fine_tune_adjust)
        decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec = averageDataTriple(fft_filtered_lower, fft_filtered_higher)
        return decoded_bitstring_1, decoded_bitstring_2, signal



      """ demodulation start..."""
      binary_array_post_fec = []
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pre_signal = audio_array = np.append(self.remainder, audio_block)

      """ processing *before* fft """
      ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, 0, 0, ocn.LOCATE_PULSE_START_INDEX)
      pulse_start_index = ret_values[0]
      ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.CALC_PULSE_OFFSETS)
      pulse_offsets = ret_values[3]
      pulse_start_index, audio_array = self.osmod.detector.processAudioArrayPulses(audio_array, pulse_start_index, 0, 0, pulse_offsets)
      ret_values = self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.CALC_BLOCK_OFFSETS)
      block_offsets = ret_values[2]
      pulse_start_index, audio_array = self.osmod.detector.processAudioArray(audio_array, pulse_start_index, 0, 0, block_offsets)
      #TEST DEBUG CODE
      self.osmod.detector.detectStandingWavePulseNew([audio_array, audio_array], frequency, pulse_start_index, 0, ocn.FIND_TRIPLET_MAX_POINT)

      fft_filtered = [None]*2
      fft_filtered[0], masked_fft_lower  = self.bandpass_filter_fft(audio_array, frequency[0] + self.osmod.fft_interpolate[0], frequency[0] + self.osmod.fft_interpolate[1])
      fft_filtered[1], masked_fft_higher = self.bandpass_filter_fft(audio_array, frequency[1] + self.osmod.fft_interpolate[2], frequency[1] + self.osmod.fft_interpolate[3])

      """ processing *after* fft """
      #if self.osmod.pulse_detection == ocn.PULSE_DETECTION_I3:
      persistent_lower, persistent_higher = self.osmod.interpolator.derivePersistentLists(pulse_start_index, fft_filtered, frequency)

      self.osmod.getDurationAndReset('findPulseStartIndex')

      self.debug.info_message("finding interpolated pulses")
      """ test routine to detect standing waves"""
      self.debug.info_message("*********************************************" )
      self.debug.info_message("non-filtered pulse_start_index: " + str(pulse_start_index) )

      self.debug.info_message("=============================================" )
      self.debug.info_message("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
      self.debug.info_message("persistent_lower: " + str(persistent_lower) )
      self.debug.info_message("persistent_higher: " + str(persistent_higher) )
      interpolated_lower, interpolated_higher, shift_amount = self.osmod.interpolator.interpolatePulseTrain([persistent_lower, persistent_higher])


      """ start of disposition loop goes here """

      residuals = None
      if self.osmod.extrapolate == 'no':
        self.extrapolate_step = ocn.EXTRAPOLATE_NONE
        decoded_bitstring_1, decoded_bitstring_2, audio_array = decodeChunkCharacters(audio_array.copy(), [interpolated_lower, interpolated_higher], shift_amount)
      elif self.osmod.extrapolate == 'yes':
        #if self.osmod.holographic_decode == ocn.HOLOGRAPH_DECODE_NONE:

        self.extrapolate_step = ocn.EXTRAPOLATE_FIND_DISPOSITION_ROTATION
        saved_shift_amount = shift_amount
        saved_pulse_start_index = pulse_start_index
        saved_audio_array = audio_array.copy()
        decoded_bitstring_1, decoded_bitstring_2, audio_array = decodeChunkCharacters(audio_array.copy(), [interpolated_lower, interpolated_higher], shift_amount)
        disposition = self.osmod.detector.findDisposition(interpolated_lower, interpolated_higher)
        #disposition, match_type, best_ambiguous_match = self.osmod.detector.findDisposition(interpolated_lower, interpolated_higher)
        #if match_type == ocn.DISPOSITION_MATCH_SINGLE:
        if disposition >= 0:
          audio_array = saved_audio_array
          shift_amount = saved_shift_amount + disposition
          pulse_start_index = saved_pulse_start_index
          interpolated = self.osmod.detector.createInterpolated()
          interpolated_lower  = interpolated[0]
          interpolated_higher = interpolated[1]
          self.extrapolate_step = ocn.EXTRAPOLATE_FIXED_ROTATION_DECODE
          decoded_bitstring_1, decoded_bitstring_2, audio_array = decodeChunkCharacters(audio_array.copy(), interpolated, shift_amount)
          self.osmod.detector.findDisposition(interpolated[0], interpolated[1])
          #disposition, match_type, best_ambiguous_match = self.osmod.detector.findDisposition(interpolated[0], interpolated[1])
        #elif match_type == ocn.DISPOSITION_MATCH_AMBIGUOUS:
        #elif match_type == ocn.DISPOSITION_NO_MATCH:


        #pulse_start_index, shift_amount = self.osmod.detector.findDisposition(interpolated_lower, interpolated_higher)


      #  num_extrap_chars = self.osmod.extrapolate_seqlen

      #decoded_bitstring_1, decoded_bitstring_2 = self.decodeChunkCharacters(audio_array.copy(), ocn.EXTRAPOLATE_DECODE_WITH_FIXED_ROTATION, [interpolated_lower, interpolated_higher])
      #self.decodeChunkCharacters(audio_array.copy(), ocn.EXTRAPOLATE_ACQUIRE_ROTATION, [interpolated_lower, interpolated_higher])

      """
      self.osmod.detector.calcPulseTrainSectionAngles(frequency, interpolated_lower, interpolated_higher)
      residuals = self.osmod.detector.calcResidualPhaseAngles(audio_array, frequency, pulse_offsets, interpolated_lower, interpolated_higher)
      pulse_start_index, audio_array = self.osmod.detector.processShiftAmount(audio_array, pulse_start_index, 0, shift_amount)

      self.debug.info_message("pulse_start_index: " + str(pulse_start_index))
      median_block_offset = pulse_start_index
      median_block_offset_sig2 = median_block_offset

      self.osmod.getDurationAndReset('receive_pre_filters_interpolate')
      self.debug.info_message("fft filtering data")
      fft_filtered_lower, fft_filtered_higher = self.receive_pre_filters_filter_wave(pulse_start_index, audio_array, frequency)

      audio_array1 = (fft_filtered_lower.real + fft_filtered_lower.imag ) / 2
      audio_array2 = (fft_filtered_higher.real + fft_filtered_higher.imag) / 2

      self.osmod.getDurationAndReset('receive_pre_filters_filter_wave')

      if display:
        self.debug.info_message("charting data")
        self.osmod.analysis.drawWaveCharts(fft_filtered_lower, fft_filtered_higher, 150, 150, 22, 22)

      self.osmod.getDurationAndReset('init')

      if self.osmod.form_gui.window['cb_override_downconvertmethod'].get():
        self.osmod.baseband_conversion = self.osmod.form_gui.window['combo_downconvert_type'].get()

      self.downconvert_I3_RelExp(pulse_start_index, [fft_filtered_lower, fft_filtered_higher], frequency, interpolated_lower, interpolated_higher, fine_tune_adjust)
      decoded_bitstring_1, decoded_bitstring_2 = averageDataTriple(fft_filtered_lower, fft_filtered_higher)
      """

      """ end """



      """ calculate remainder to be tacked on front of next sample"""
      remainder_start = ((len(audio_array) - pulse_start_index) // self.osmod.symbol_block_size) * self.osmod.symbol_block_size
      self.remainder = pre_signal[remainder_start::]

      """ convert constellation values to text"""
      self.osmod.getDurationAndReset('displayTextResults')

      #self.osmod.detector.writePulseTrainRotationDetails()

      enable_display = self.osmod.form_gui.window['cb_display_phases'].get()
      if enable_display:
        self.chart_results()

      return decoded_bitstring_1, decoded_bitstring_2, binary_array_post_fec

    except:
      self.debug.error_message("Exception in demodulate_2x8psk_common_I3_Relative_Phase: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def chart_results(self):
    self.debug.info_message("chart_results")
    try:
      """ display baseband signals """
      self.debug.info_message("charting data")
      if self.osmod.phase_extraction == ocn.EXTRACT_NORMAL:
        self.displayChartResults(recovered_signal1[0], recovered_signal1[1], 'red', 'blue', True, 'mychart')
      elif self.osmod.phase_extraction == ocn.EXTRACT_INTERPOLATE:
        #self.debug.info_message("phase error history: " + str(error_1))

        chart_type = self.osmod.form_gui.window['option_chart_options'].get()
        if chart_type == 'Before Mean Average':
          self.displayChartResults(phase_data_before_averaging1, phase_data_before_averaging2, 'red', 'blue', True, 'mychart')
        elif chart_type == 'After Mean Average':
          self.displayChartResults(recovered_signal1a, recovered_signal1b, 'red', 'blue', True, 'mychart')
        elif chart_type == 'Both':
          self.displayChartResults(phase_data_before_averaging1, recovered_signal1a, 'red', 'blue', True, 'mychart')
        elif chart_type == 'FFT':
          self.displayChartResults(np.angle(masked_fft_lower), np.angle(masked_fft_higher), 'red', 'blue', True, 'mychart')
        elif chart_type == 'EXP':
          #self.displayChartResults(exp_lower.real, exp_lower.imag)
          self.displayChartResults(test_chart_real[0], test_chart_imag[0], 'red', 'blue', True, 'low frequency real(red) & imag(blue)', ocn.CHART_ONE)
          self.displayChartResults(np.angle(exp_lower), np.angle(exp_lower)%8, 'green', 'yellow', False, '', ocn.CHART_ONE)
          self.displayChartResults(test_chart_real[1], test_chart_imag[1], 'red', 'blue', True, 'high frequency real(red) & imag(blue)', ocn.CHART_TWO)
          self.displayChartResults(np.angle(exp_higher), np.angle(exp_higher)%8, 'green', 'yellow', False, '', ocn.CHART_TWO)
        elif chart_type == 'Phase Error':
          two_times_pi = np.pi * 2
          max_phase_value = two_times_pi / 8
          """ display ACTUAL phase errors"""
          self.displayChartResults(np.angle(exp_lower)%max_phase_value, np.angle(exp_higher)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_ONE)

          interpolated1 = self.interpolatePhaseDrift(exp_lower, median_block_offset, frequency[0], where1, ocn.PHASE_ERROR_ROUGH)
          interpolated2 = self.interpolatePhaseDrift(exp_higher, median_block_offset, frequency[1], where2, ocn.PHASE_ERROR_ROUGH)
          self.displayChartResults(interpolated1, interpolated2, 'red', 'blue', True, 'Phase Error Rough & Smooth', ocn.CHART_TWO)

          interpolated1 = self.interpolatePhaseDrift(exp_lower, median_block_offset, frequency[0], where1, ocn.PHASE_ERROR_SMOOTH)
          interpolated2 = self.interpolatePhaseDrift(exp_higher, median_block_offset, frequency[1], where2, ocn.PHASE_ERROR_SMOOTH)
          self.displayChartResults(interpolated1, interpolated2, 'red', 'blue', False, '', ocn.CHART_TWO)

          """ temporarily remove as time consuming for testing 
          self.displayChartResults(interpolated1 + interpolated2, interpolated1 + interpolated2, 'black', 'black', False, '', ocn.CHART_TWO)
          """
        elif chart_type == 'Frequency & EXP':
          #two_times_pi = np.pi * 2
          #max_phase_value = two_times_pi / 8
          """ display ACTUAL phase values"""
          #self.displayChartResults(np.angle(frequency_test_sig1_low)%max_phase_value, np.angle(frequency_test_sig1_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_ONE)
          #self.displayChartResults(np.angle(frequency_test_sig1_low), np.angle(frequency_test_sig1_high), 'red', 'blue', True, 'Phase Values Actual', ocn.CHART_ONE)

          #self.displayChartResults(np.angle(frequency_test_sig1_low), np.angle(frequency_test_sig1_low), 'red', 'red', True, 'downconverted data angle low', ocn.CHART_ONE_A)
          #self.displayChartResults(np.angle(frequency_test_sig1_high), np.angle(frequency_test_sig1_high), 'blue', 'blue', True, 'downconverted data angle high', ocn.CHART_TWO_A)

          complex_wave1 = test_lower_1 + 1j * test_lower_2
          complex_wave2 = test_higher_1 + 1j * test_higher_2
          self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, 'Angles Low (Red) & High (Blue)', ocn.CHART_ONE)
          self.displayChartResults(complex_wave1.real, complex_wave1.imag, 'red',  'blue', True, 'Real & Imag - Low(red,blue) - High(pink,cyan)', ocn.CHART_TWO)
          self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_TWO)

          complex_wave1 = test_lower_1_45 + 1j * test_lower_2_45
          complex_wave2 = test_higher_1_45 + 1j * test_higher_2_45
          self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, '45 - Angles Low (Red) & High (Blue)', ocn.CHART_THREE)
          self.displayChartResults(complex_wave1.real, complex_wave1.imag, 'red', 'blue', True, '45 - Real & Imag - Low(red,blue) - High(pink,cyan)', ocn.CHART_FOUR)
          self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_FOUR)


          #self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'blue', 'cyan', True, 'Phase Values Actual', ocn.CHART_FOUR)

          #self.displayChartResults(np.angle(frequency_test_sig1_low_90), np.angle(frequency_test_sig1_high_90), 'red', 'blue', True, 'Phase Values Actual 90', ocn.CHART_TWO)
          #self.displayChartResults(np.angle(frequency_test_sig2_low)%max_phase_value, np.angle(frequency_test_sig2_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_TWO)
          #self.displayChartResults(np.angle(frequency_test_sig3_low)%max_phase_value, np.angle(frequency_test_sig3_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_THREE)
          #self.displayChartResults(np.angle(frequency_test_sig4_low)%max_phase_value, np.angle(frequency_test_sig4_high)%max_phase_value, 'red', 'blue', True, 'Phase Error Actual', ocn.CHART_FOUR)

        elif chart_type == 'EXP Intra Triple':
          #complex_wave1 = test_lower_1 + 1j * test_lower_2
          #complex_wave2 = test_higher_1 + 1j * test_higher_2

          #self.displayChartResults(np.angle(complex_wave1), np.angle(complex_wave2), 'red', 'blue', True, 'Angles Low (Red) & High (Blue), Real & Imag - Low(pink,cyan)', ocn.CHART_ONE)
          self.displayChartResults(np.angle(new_complex_wave_low), np.angle(new_complex_wave_high), 'red', 'blue', True, 'Angles Low (Red) & High (Blue)', ocn.CHART_ONE)
          self.displayChartResults(new_complex_wave_low.real, new_complex_wave_low.imag, 'red',  'blue', False, 'Real & Imag - Low(red,blue)', ocn.CHART_TWO)
          #self.displayChartResults(complex_wave2.real, complex_wave2.imag, 'pink', 'cyan', False, '', ocn.CHART_TWO)

          #self.displayChartResults(new_complex_wave_high.real, new_complex_wave_high.imag, 'red',  'blue', False, 'Real & Imag - High(red,blue)', ocn.CHART_THREE)

          self.displayChartResults(intra_triple_charts[0], intra_triple_charts[1], 'red',  'blue', True, 'Intra Extract Angles', ocn.CHART_THREE)
          self.displayChartResults(intra_triple_charts[2], intra_triple_charts[2], 'pink', 'cyan', False, '', ocn.CHART_THREE_A)

          self.displayChartResults(intra_triple_charts[3], intra_triple_charts[4], 'red',  'blue', True, 'Intra Extract Reference (red) Phase Corrected (blue, cyan)', ocn.CHART_FOUR)
          self.displayChartResults(intra_triple_charts[5], intra_triple_charts[5], 'pink', 'cyan', False, '', ocn.CHART_FOUR_A)

          #self.displayChartResults(intra_triple_charts[6], intra_triple_charts[7], 'red',  'blue', True, 'Intra Extract Diff/Total 1,2,3 (red, blue, pink)', ocn.CHART_FOUR)
          #self.displayChartResults(intra_triple_charts[8], intra_triple_charts[8], 'pink', 'cyan', False, '', ocn.CHART_FOUR_A)


        elif chart_type == 'Chart Data Dictionary':
          self.displayChartResults(self.chart_data_dict['averaged_lower_complex'].real, self.chart_data_dict['averaged_lower_complex'].imag, 'red', 'blue', True, 'Averaged Lower Real (Red) Imag (Blue) Higher Real (Pink) Imag (Cyan)', ocn.CHART_ONE)
          self.displayChartResults(self.chart_data_dict['averaged_higher_complex'].real, self.chart_data_dict['averaged_higher_complex'].imag, 'pink', 'cyan', False, '', ocn.CHART_ONE)
          self.displayChartResults(np.angle(self.chart_data_dict['averaged_lower_complex']), np.angle(self.chart_data_dict['averaged_higher_complex']), 'red', 'blue', True, 'Angles Lower (Red) & Higher (Blue)', ocn.CHART_TWO)
          self.displayChartResults(self.chart_data_dict['smoothed_a_real_lower'], self.chart_data_dict['smoothed_a_imag_lower'], 'red', 'blue', True, 'Smoothed Pulse A Lower Real (Red)  Imag (Blue) Higher Real (green) Imag (cyan)', ocn.CHART_THREE)
          self.displayChartResults(self.chart_data_dict['smoothed_a_real_higher'], self.chart_data_dict['smoothed_a_imag_higher'], 'green', 'cyan', False, '', ocn.CHART_THREE_A)
          self.displayChartResults(self.chart_data_dict['smoothed_angle_a_lower'], self.chart_data_dict['smoothed_angle_a_higher'], 'red', 'blue', True, 'Smoothed Pulse Angles Lower A (Red) Higher A (Blue)', ocn.CHART_FOUR)

    except:
      self.debug.error_message("Exception in chart_results: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
