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
from numpy import pi
from scipy.signal import butter, filtfilt, firwin
from modem_core_utils import ModemCoreUtils

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



class ModulatorPSK(ModemCoreUtils):

  phase_shifts = np.array([0+1j, 1+1j, 1+0j, 1-1j, 0-1j, -1-1j, -1+0j, -1+1j])
  phase_shift_angles  = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
  qpsk_mapping = {(0,0) : 1+0j, (0,1): 0+1j, (1,0): 0-1j, (1,1): -1+0j}

  inphase_component    = {  0:-1,   0:-1,   0:1,    0:1}
  quadrature_component = {  0:-1,   0:1,    0:-1,   0:1}

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MOD)
    self.debug.info_message("__init__")
    super().__init__(osmod)


  def ones_symbol_wave_function(self, symbol_wave):
    try:
      symbol_wave = ((symbol_wave[0] * self.osmod.filtRRC_wave1) +
                     (symbol_wave[1] * self.osmod.filtRRC_wave2))  / 2
      return symbol_wave
    except:
      self.debug.error_message("Exception in ones_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def halves_symbol_wave_function(self, symbol_wave):
    try:
      symbol_wave = ((symbol_wave[0] * self.osmod.filtRRC_wave1) +
                     (symbol_wave[1] * self.osmod.filtRRC_wave2))  / 2
      return symbol_wave
    except:
      self.debug.error_message("Exception in eights_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def fourths_symbol_wave_function(self, symbol_wave):
    try:
      symbol_wave = ((symbol_wave[0] * self.osmod.filtRRC_fourth_wave[0]) +
                     (symbol_wave[0] * self.osmod.filtRRC_fourth_wave[1]) +
                     (symbol_wave[1] * self.osmod.filtRRC_fourth_wave[2]) +
                     (symbol_wave[1] * self.osmod.filtRRC_fourth_wave[3]))  / 4
      return symbol_wave
    except:
      self.debug.error_message("Exception in eights_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def eighths_symbol_wave_function(self, symbol_wave):
    try:
      symbol_wave = ((symbol_wave[0] * self.osmod.filtRRC_eighth_wave[0]) +
                     (symbol_wave[0] * self.osmod.filtRRC_eighth_wave[1]) +
                     (symbol_wave[0] * self.osmod.filtRRC_eighth_wave[2]) +
                     (symbol_wave[0] * self.osmod.filtRRC_eighth_wave[3]) +
                     (symbol_wave[1] * self.osmod.filtRRC_eighth_wave[4]) +
                     (symbol_wave[1] * self.osmod.filtRRC_eighth_wave[5]) +
                     (symbol_wave[1] * self.osmod.filtRRC_eighth_wave[6]) +
                     (symbol_wave[1] * self.osmod.filtRRC_eighth_wave[7]))  / 8
      return symbol_wave
    except:
      self.debug.error_message("Exception in eights_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def sixteenths_symbol_wave_function(self, symbol_wave):
    try:
      symbol_wave = ((symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[0]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[1]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[2]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[3]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[4]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[5]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[6]) +
                     (symbol_wave[0] * self.osmod.filtRRC_sixteenth_wave[7]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[8]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[9]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[10]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[11]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[12]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[13]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[14]) +
                     (symbol_wave[1] * self.osmod.filtRRC_sixteenth_wave[15]))  / 16
      return symbol_wave
    except:
      self.debug.error_message("Exception in eights_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def thirtyseconds_symbol_wave_function(self, symbol_wave):
    try:
      new_symbol_wave = np.zeros_like(symbol_wave[0])
      for pulse_count in range(0, 32):
        new_symbol_wave = new_symbol_wave + (symbol_wave[int(pulse_count // 16)] * self.osmod.filtRRC_thirtysecond_wave[pulse_count])

    except:
      self.debug.error_message("Exception in thirtyseconds_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return new_symbol_wave / 32

  def sixtyfourths_symbol_wave_function(self, symbol_wave):
    try:
      new_symbol_wave = np.zeros_like(symbol_wave[0])
      for pulse_count in range(0, 64):
        new_symbol_wave = new_symbol_wave + (symbol_wave[int(pulse_count // 32)] * self.osmod.filtRRC_sixtyfourth_wave[pulse_count])

    except:
      self.debug.error_message("Exception in sixtyfourths_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return new_symbol_wave / 64

  def onehundredtwentyeighths_symbol_wave_function(self, symbol_wave):
    try:
      new_symbol_wave = np.zeros_like(symbol_wave[0])
      for pulse_count in range(0, 128):
        new_symbol_wave = new_symbol_wave + (symbol_wave[int(pulse_count // 64)] * self.osmod.filtRRC_onehundredtwentyeighth_wave[pulse_count])

    except:
      self.debug.error_message("Exception in onehundredtwentyeighths_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return new_symbol_wave / 128

  def twohundredfiftysixths_symbol_wave_function(self, symbol_wave):
    try:
      new_symbol_wave = np.zeros_like(symbol_wave[0])
      for pulse_count in range(0, 256):
        new_symbol_wave = new_symbol_wave + (symbol_wave[int(pulse_count // 128)] * self.osmod.filtRRC_twohundredfiftysixth_wave[pulse_count])

    except:
      self.debug.error_message("Exception in twohundredfiftysixths_symbol_wave_function: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return new_symbol_wave / 256



  def modulate_2fsk_npsk_optimized(self, frequency, bit_sequence, n_bits, n_sections):
    self.debug.info_message("modulate_2fsk_npsk_optimized: ")

    self.debug.info_message("full bit_sequence: " + str(bit_sequence) )

    modulated_wave_signal = np.array( [] )
    try:
      phase_sequence1 = []
      phase_sequence2 = []
      phase_sequence3 = []
      self.debug.info_message("bit_sequence: " + str(bit_sequence[0]) )

      if n_sections == 2:
        for i in range(0, len(bit_sequence[0]), n_bits):
          phase_sequence1.append(self.phase_shift_angles[int(''.join(str(bit) for bit in bit_sequence[0][i:i+n_bits]), 2)])
          phase_sequence2.append(self.phase_shift_angles[int(''.join(str(bit) for bit in bit_sequence[1][i:i+n_bits]), 2)])
      elif n_sections == 3:
        for i in range(0, len(bit_sequence[0]), n_bits):
          phase_sequence1.append(self.phase_shift_angles[int(''.join(str(bit) for bit in bit_sequence[0][i:i+n_bits]), 2)])
          phase_sequence2.append(self.phase_shift_angles[int(''.join(str(bit) for bit in bit_sequence[1][i:i+n_bits]), 2)])
          phase_sequence3.append(self.phase_shift_angles[int(''.join(str(bit) for bit in bit_sequence[2][i:i+n_bits]), 2)])

      self.debug.info_message("phase_sequence1: " + str(phase_sequence1) )

      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate

      term5 = 2 * np.pi * time
      term6 = term5 * frequency[0]
      term7 = term5 * frequency[1]
      if n_sections == 2:
        for phase1, phase2 in zip(phase_sequence1, phase_sequence2):
          term8 = term6 + phase1
          term9 = term7 + phase2
          symbol_wave1 = self.amplitude * np.cos(term8) + self.amplitude * np.sin(term8)
          symbol_wave2 = self.amplitude * np.cos(term9) + self.amplitude * np.sin(term9)
          symbol_wave = self.osmod.symbol_wave_function([symbol_wave1, symbol_wave2])
          modulated_wave_signal = np.concatenate((modulated_wave_signal, symbol_wave))
      elif n_sections == 3:
        """ used for abb block format """
        for phase1, phase2, phase3 in zip(phase_sequence1, phase_sequence2, phase_sequence3):
          term8  = term6 + phase1 # a -- freq[0] used for abb
          term9  = term7 + phase2 # b -- freq[1] used for abb
          term10 = term7 + phase3 # b -- freq[1] used for abb
          symbol_wave1 = self.amplitude * np.cos(term8)  + self.amplitude * np.sin(term8)
          symbol_wave2 = self.amplitude * np.cos(term9)  + self.amplitude * np.sin(term9)
          symbol_wave3 = self.amplitude * np.cos(term10) + self.amplitude * np.sin(term10)
          symbol_wave = self.osmod.symbol_wave_function([symbol_wave1, symbol_wave2, symbol_wave3])
          modulated_wave_signal = np.concatenate((modulated_wave_signal, symbol_wave))

    except:
      self.debug.error_message("Exception in modulate_2fsk_npsk_optimized: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return modulated_wave_signal



  def encoder_8psk_callback(self, outdata, frames, time, status):

    hasData = not self.osmod.isDataQueueEmpty()

    if hasData:
      data = self.osmod.popDataQueue()
      outdata[:,0] = self.modulateChunk8PSK(1500, data) 
    else:
      syms = np.zeros(self.osmod.symbol_block_size)
      outdata[:,0] = syms

    return None

  def encoder_qpsk_callback(self, outdata, frames, time, status):
    return



