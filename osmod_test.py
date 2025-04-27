#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import wave
import sys

from numpy import pi
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from scipy.io.wavfile import write, read

from modulators import ModulatorPSK 
from demodulators import DemodulatorPSK 
from queue import Queue

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

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod

  def testInterpolate(self, mode):
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


      interpolated_lower  = self.osmod.demodulation_object.interpolate_contiguous_items(max_occurrences_lower)
      interpolated_higher = self.osmod.demodulation_object.interpolate_contiguous_items(max_occurrences_higher)
      self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))
      sorted_lower  = self.osmod.demodulation_object.sort_interpolated(interpolated_lower)
      sorted_higher = self.osmod.demodulation_object.sort_interpolated(interpolated_higher)
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

  def testRoutine2(self, mode, form_gui, noise_mode, text_num, chunk_num):
    self.debug.info_message("testRoutine2")
    self.window = form_gui.window
    self.test1(mode, noise_mode, text_num, chunk_num)

  def modDemod(self, mode, form_gui):
    self.debug.info_message("modDemod")
    self.window = form_gui.window
    self.test1('LB28-TURBOx2.5-10', 0, 30)


  def test1(self, mode, noise_mode, text_num, chunk_num):
    self.debug.info_message("test1")

    try:
      """ initialize the block"""
      self.osmod.setInitializationBlock(mode)

      """ figure out the carrier frequencies"""
      center_frequency = 1400
      frequency = self.osmod.calcCarrierFrequencies(center_frequency)
      self.debug.info_message("center frequency: " + str(center_frequency))
      self.debug.info_message("carrier frequencies: " + str(frequency))

      """ convert text to bits"""
      text_examples = [0] * 13
      text_examples[0]  = " peter piper picked a peck of pickled pepper corn "
      text_examples[1]  = "jack be nimble jack be quick jack jump over the candlestick"
      text_examples[2]  = "row row row your boat gently down the stream merrily merrily merrily merrily life is but a dream"
      text_examples[3]  = "hickory dickory dock the mouse ran up the clock the clock struch one the mouse ran down hickory dicory dock"
      text_examples[4]  = "its raining its pouring the old man is snoring he bumped his head and went to bed and he couldnt get up in the morning"
      text_examples[5]  = "jack and jill went up the hill to fetch a pail of water jack fell down and broke his crown and jill came tumbling after"
      text_examples[6]  = "humpty dumpty dat on a wall humpty dumpty had a great fall all the kings forses and all the kings men coudnt put humpty together again"
      text_examples[7]  = "a wise old owl sat in an oak the more he heard the less he spoke the less he spoke the more he heard why arent we all like that wise old bird"
      text_examples[8]  = "hey diddle diddle the cat and the fiddle the cow jumped over the moon the little dog laughed to see such fun and the dish ran away with the spoon"
      text_examples[9]  = "baa baa black sheep have you any wool yes sir yes sir three bags full one for the master and one for the dame and one for the little boy who lives down the lane"
      text_examples[10] = "twinkle twinkle little bat how i wonder what youre at up above the world you fly like a tea tray in the sky twinkle twinkle little bat how i wonder what youre at"
      text_examples[11] = "i can read on a boat i can read with a goat i can read on a train i can read in the rain i can read with a fox i can read in a box i can read with a mouse i can read in a house i can read here or there i can read anywhere"
      text_examples[12] = "the queen of hearts she made some tarts all on a summers day the knave of hearts he stole the tarts and took them clean away the king of hearts called for the tarts and beat the knave full sore the knave of hearts brought back the tarts and vowed hed steal no more"

      text = text_examples[int(text_num)] + ' '
 
      self.debug.info_message("encoding text: " + str(text))

      bit_groups, sent_bitstring = self.osmod.text_encoder(text)
      data2 = self.osmod.modulation_object.modulate(frequency, bit_groups)

      """ write to file """
      self.osmod.modulation_object.writeFileWav('8psktest11.wav', data2)

      """ read file """
      audio_array = self.osmod.modulation_object.readFileWav('8psktest11.wav')
      self.debug.info_message("audio data type: " + str(audio_array.dtype))
      self.debug.info_message("demodulating")
      total_audio_length = len(audio_array)

      """ add noise for testing..."""
      noise_free_signal = audio_array*0.00001

      self.debug.info_message("noise mode: " + str(noise_mode))
      value = float(noise_mode)

      if  self.window['cb_enable_awgn'].get():
        audio_array = self.osmod.modulation_object.addAWGN(noise_free_signal, value, frequency)
      audio_array = self.osmod.modulation_object.addTimingNoise(audio_array)
      audio_array = self.osmod.modulation_object.addPhaseNoise2(audio_array)

      self.osmod.modulation_object.writeFileWav('withnoise.wav', audio_array)

      """ reset the remainder"""
      self.osmod.demod_2fsk8psk.remainder = np.array([])

      """ split into blocks for testing..."""
      self.osmod.startTimer('test12_demod_timer')

      rcvd_bitstring_1 = []
      rcvd_bitstring_2 = []

      how_many_blocks = max(1, int(len(text) // int(chunk_num)))

      audio_block = np.array_split( audio_array , how_many_blocks, axis=0)
      for block_count in range (how_many_blocks):
        self.debug.info_message("num_divisor: " + str(how_many_blocks))
        self.debug.info_message("calling demodulate_2fsk_8psk. block count: " + str(block_count))
        decoded_bitstring_1, decoded_bitstring_2 = self.osmod.demodulation_object.demodulate_2fsk_8psk(audio_block[block_count], frequency)

        self.debug.info_message("appending bitstrings")
        rcvd_bitstring_1.append(decoded_bitstring_1)
        rcvd_bitstring_2.append(decoded_bitstring_2)

      self.debug.info_message("complete")

      self.debug.info_message("elapsed time: " + str(self.osmod.getDuration('test12_demod_timer')))
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
      for block in range(0, len(rcvd_bitstring_1)):
        for bits1, bits2 in zip(rcvd_bitstring_1[block], rcvd_bitstring_2[block]):
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
          self.debug.info_message("error: " + str(error))
          count = count + 1

      self.debug.info_message("total error: " + str(error))
      self.debug.info_message("num bits: " + str(len(sent_bitstring[0])*6))
      ber = error/(len(sent_bitstring[0])*6)
      self.debug.info_message("BER: " + str(ber))
      total_num_bits = len(sent_bitstring[0])*6

      ebn0_db, ebn0, SNR_equiv_db = self.osmod.mod_2fsk8psk.calculate_EbN0(audio_array, frequency, total_num_bits, bits_per_second, noise_free_signal) 
      self.osmod.form_gui.window['text_ber_value'].update("BER: " + str(ber))
      self.osmod.form_gui.window['text_ebn0_value'].update("Eb/N0: " + str(ebn0))
      self.osmod.form_gui.window['text_ebn0db_value'].update("Eb/N0 (dB): " + str(ebn0_db))
      self.osmod.form_gui.window['text_snr_value'].update("SNR Equiv. : " + str(SNR_equiv_db))

    except:
      self.debug.error_message("Exception in test1: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
