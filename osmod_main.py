#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import sys

from numpy import pi
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter

from modulators import ModulatorPSK
from demodulators import DemodulatorPSK
from osmod_2fsk_8psk import mod_2FSK8PSK, demod_2FSK8PSK
from osmod_2fsk_4psk import mod_2FSK4PSK, demod_2FSK4PSK
from modem_core_utils import ModemCoreUtils
from queue import Queue
from datetime import datetime, timedelta


class osModem(object):

  mod_psk   = None
  demod_psk = None

  mod_2fsk8psk   = None
  demod_2fsk8psk = None
  mod_2fsk4psk   = None
  demod_2fsk4psk = None

  mod_fsk   = None
  demod_fsk = None

  runDecoder = False

  previousBlocksizeIn  = 0
  previousBlocksizeOut = 0
  inStreamRunning      = False
  outStreamRunning     = False

  spectral_density_queue_counter = 0
  spectral_density_block = None

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)

  def __init__(self, form_gui):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")

    self.form_gui = form_gui

    self.sample_rate = 4410 * 5 #44100
    self.attenuation = 30
    self.center_frequency = 1500
    self.symbols = 32
    self.bandwidth = 1000
    self.bits_per_symbol = int(np.log2(self.symbols))

    """ frequency separation between tones, in Hz (baud rate) """
    self.freq_sep = self.bandwidth/self.symbols
    self.time_sep = int(np.ceil(self.sample_rate/self.freq_sep))

    self.core_utils = ModemCoreUtils(self)

    self.mod_2fsk8psk   = mod_2FSK8PSK(self)
    self.demod_2fsk8psk = demod_2FSK8PSK(self)

    self.mod_2fsk4psk   = mod_2FSK4PSK(self)
    self.demod_2fsk4psk = demod_2FSK4PSK(self)

    self.dataQueue = Queue()
    self.inputBuffer = Queue()

    self.two_times_pi = 2 * np.pi

    """ initialize the initialization blocks for the different modulations"""

    self.modulation_initialization_block = {'LB28-0.15625-10I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 51200,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 256,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                        'LB28-0.3125-10I':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.3125 characters per second, 1.875 baud (bits per second)',
                                                              'symbol_block_size'    : 25600,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (32/128, 96/128),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-2, 2, -2, 2),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 128,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                        'LB28-0.625-10I':    {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.625 characters per second, 3.75 baud (bits per second)',
                                                              'symbol_block_size'    : 12800,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (16/64, 48/64),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 64,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                        'LB28-1.25-10I':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '1.25 characters per second, 7.5 baud (bits per second)',
                                                              'symbol_block_size'    : 6400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.thirtyseconds_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (8/32, 24/32),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 32,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                      'LB28-2.5-10I':        {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '2.5 characters per second, 15 baud (bits per second)',
                                                              'symbol_block_size'    : 3200,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.sixteenths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (4/16, 12/16),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 16,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                             'LB28-10-20':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '10 characters per second, 60 baud (bits per second)',
                                                              'symbol_block_size'    : 800,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.fourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (0/4, 2/4),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 20,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 4,
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                              'LB28-5-10':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : 'Narrow bandwidth 2fsk + 8psk, 64 bit characters: 5 characters per second, 30 baud (bits per second)',
                                                              'extraction_points'    : (2/8, 6/8),
                                                              'symbol_block_size'    : 1600,
                                                              'symbol_wave_function' : self.mod_2fsk8psk.eighths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 10,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 8,
                                                              'parameters'           : (600, 0.8, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                              'LB28-10-40':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '10 characters per second, 60 baud (bits per second)',
                                                              'symbol_block_size'    : 800,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.fourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (0/4, 2/4),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 40,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 4,
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                          'LB28-20-100':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                                                              'info'                 : 'Double Carrier Orthogonal 8psk 64 bit characters:- 20 characters per second, 120 baud (bits per second)',
                                                              'symbol_block_size'    : 400,
                                                              'symbol_wave_function' : self.mod_2fsk8psk.halves_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'extraction_points'    : (0.0, 0.5),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 100,
                                                              'detector_function'    : 'median',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 2,
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98) }}  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


    """ set some default values..."""
    self.setInitializationBlock('LB28-10-40')


  def setInitializationBlock(self, mode):
    self.debug.info_message("setInitializationBlock")
    try:
      self.encoder_callback     = self.modulation_initialization_block[mode]['encoder_callback']
      self.decoder_callback     = self.modulation_initialization_block[mode]['decoder_callback']
      self.text_encoder         = self.modulation_initialization_block[mode]['text_encoder']
      self.mode_selector        = self.modulation_initialization_block[mode]['mode_selector']
      self.symbol_block_size    = self.modulation_initialization_block[mode]['symbol_block_size']
      self.sample_rate          = self.modulation_initialization_block[mode]['sample_rate']
      self.parameters           = self.modulation_initialization_block[mode]['parameters']
      self.carrier_separation   = self.modulation_initialization_block[mode]['carrier_separation']
      self.num_carriers         = self.modulation_initialization_block[mode]['num_carriers']
      self.pulses_per_block     = self.modulation_initialization_block[mode]['pulses_per_block']
      self.detector_function    = self.modulation_initialization_block[mode]['detector_function']
      self.symbol_wave_function = self.modulation_initialization_block[mode]['symbol_wave_function']
      self.extraction_points    = self.modulation_initialization_block[mode]['extraction_points']
      self.phase_extraction     = self.modulation_initialization_block[mode]['phase_extraction']
      self.baseband_conversion  = self.modulation_initialization_block[mode]['baseband_conversion']
      self.process_debug        = self.modulation_initialization_block[mode]['process_debug']
      self.fft_filter           = self.modulation_initialization_block[mode]['fft_filter']
      self.fft_interpolate      = self.modulation_initialization_block[mode]['fft_interpolate']
      self.modulation_object    = self.modulation_initialization_block[mode]['modulation_object']
      self.demodulation_object  = self.modulation_initialization_block[mode]['demodulation_object']


      self.sampling_frequency = self.sample_rate / self.symbol_block_size

      """ Fixed values. ref based on 1500 Hz reference frequency """
      self.carrier_frequency_reference = 1500
      normalized_sample_size           = self.carrier_frequency_reference * 2 # resolution to 1/2 Hz

      """ key inputs """
      self.samples_per_wave       = 30.0

      """ calculate derived values from key inputs """

      """ based on 1500 Hz reference"""
      self.num_waves_ref          = round(normalized_sample_size * (1500.0/self.carrier_frequency_reference)) 
      self.num_samples_total      = self.num_waves_ref * self.samples_per_wave
      self.num_wave_samples_ref   = self.num_samples_total / self.num_waves_ref
      self.waves_per_block_ref    = self.symbol_block_size / self.num_wave_samples_ref
      self.symbols_per_block      = 1
      self.samples_per_symbol     = self.symbol_block_size / self.symbols_per_block
      self.symbols_per_second     = self.sample_rate / self.symbol_block_size 
      self.samples_per_second     = self.samples_per_symbol * self.symbols_per_second

      self.debug.info_message("normalized_sample_size: " + str(normalized_sample_size) )
      self.debug.info_message("num_waves_ref : " + str(self.num_waves_ref ) )
      self.debug.info_message("num_samples_total: " + str(self.num_samples_total) )
      self.debug.info_message("num_wave_samples_ref: " + str(self.num_wave_samples_ref) )
      self.debug.info_message("waves_per_block_ref: " + str(self.waves_per_block_ref) )
      self.debug.info_message("suggested block size: " + str(round(self.waves_per_block_ref) * self.num_wave_samples_ref ) )
      self.debug.info_message("suggested block size @ 20 waves per block at 1500Hz: " + str(20 * self.num_wave_samples_ref ) )
      self.debug.info_message("symbols_per_block: " + str(self.symbols_per_block) )
      self.debug.info_message("samples_per_symbol: " + str(self.samples_per_symbol) )
      self.debug.info_message("symbols_per_second: " + str(self.symbols_per_second) )

      if self.pulses_per_block == 1:
        """ calculate RRC coefficients for single pulse"""
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size), self.parameters[1], self.parameters[2], self.sample_rate)
        self.filtRRC_wave1 = self.filtRRC_coef_main
        self.filtRRC_wave2 = self.filtRRC_coef_main # not required
      elif self.pulses_per_block == 2:
        """ calculate the Orthogonal RRC coefficients for double carrier"""
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/2), self.parameters[1], self.parameters[2], self.sample_rate)
        self.filtRRC_wave1 = np.append(self.filtRRC_coef_main, np.zeros(int(self.symbol_block_size/2)), )
        self.filtRRC_wave2 = np.append(np.zeros(int(self.symbol_block_size/2)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 4:
        """ calculate the RRC coefficients for quad carrier"""
        divisor = 4
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_fourth_wave = [0] * divisor
        self.filtRRC_fourth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_fourth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_fourth_wave[i] = np.append(self.filtRRC_fourth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_fourth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 8:
        """ calculate the RRC coefficients for eighth carrier"""
        divisor = 8
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_eighth_wave = [0] * divisor
        self.filtRRC_eighth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_eighth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_eighth_wave[i] = np.append(self.filtRRC_eighth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_eighth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 12:
        """ calculate the RRC coefficients for twelfth carrier"""
        divisor = 12
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_twelfth_wave = [0] * divisor
        self.filtRRC_twelfth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_twelfth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_twelfth_wave[i] = np.append(self.filtRRC_twelfth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_twelfth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 16:
        """ calculate the RRC coefficients for sixteenth carrier"""
        divisor = 16
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_sixteenth_wave = [0] * divisor
        self.filtRRC_sixteenth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_sixteenth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_sixteenth_wave[i] = np.append(self.filtRRC_sixteenth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_sixteenth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 32:
        """ calculate the RRC coefficients for 32nds carrier"""
        divisor = 32
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_thirtysecond_wave = [0] * divisor
        self.filtRRC_thirtysecond_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_thirtysecond_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_thirtysecond_wave[i] = np.append(self.filtRRC_thirtysecond_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_thirtysecond_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 64:
        """ calculate the RRC coefficients for 64ths carrier"""
        divisor = 64
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_sixtyfourth_wave = [0] * divisor
        self.filtRRC_sixtyfourth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_sixtyfourth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_sixtyfourth_wave[i] = np.append(self.filtRRC_sixtyfourth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_sixtyfourth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 128:
        """ calculate the RRC coefficients for 128ths carrier"""
        divisor = 128
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_onehundredtwentyeighth_wave = [0] * divisor
        self.filtRRC_onehundredtwentyeighth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_onehundredtwentyeighth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_onehundredtwentyeighth_wave[i] = np.append(self.filtRRC_onehundredtwentyeighth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_onehundredtwentyeighth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 256:
        """ calculate the RRC coefficients for 256ths carrier"""
        divisor = 256
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

        self.filtRRC_twohundredfiftysixth_wave = [0] * divisor
        self.filtRRC_twohundredfiftysixth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_twohundredfiftysixth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_twohundredfiftysixth_wave[i] = np.append(self.filtRRC_twohundredfiftysixth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_twohundredfiftysixth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)


      """ initialize the sin cosine optimization lookup tables"""
      self.radTablesInitialize()
      #self.sinRadTest()

    except:
      self.debug.error_message("Exception in setInitializationBlock: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calcCarrierFrequencies(self, center_frequency):
    self.debug.info_message("calcCarrierFrequencies")
    try:
      frequency = []
      span = (self.num_carriers-1) * self.carrier_separation
      if span > 0:
        for i in range(0, self.num_carriers):
          temp_freq = center_frequency - int(span/2) + (i * self.carrier_separation) 
          """ frequency must be on a 20Hz boundary for the 20 characters per second mode to work correctly """
          temp_freq = temp_freq // 20
          temp_freq = temp_freq * 20
          #frequency.append(center_frequency - int(span/2) + (i * self.carrier_separation) )
          frequency.append(temp_freq)
      else:
        frequency.append(center_frequency)

      return frequency

    except:
      self.debug.error_message("Exception in calcCarrierFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  """ optimized sine and cosine with table lookup """
  def radTablesInitialize(self):
    """ pre-calculate sine and cosine table from -pi to +pi"""
    """ example useage...self.cos_rad[int(angle_radians * self.symbol_block_size / 2)]"""

    self.two_times_pi_times_blocksize = self.two_times_pi * self.symbol_block_size
    self.blocksize_over_two_times_pi = (self.symbol_block_size / (2*np.pi))

    try:
      N = self.symbol_block_size
      t = np.arange(0, N+1) * ((2 * np.pi) / N )
      self.sin_rad = np.zeros_like(t)
      self.cos_rad = np.zeros_like(t)
      for i in range(0, len(t)):
        self.sin_rad[i] = np.sin(t[i] )
        self.cos_rad[i] = np.cos(t[i] )

    except:
      self.debug.error_message("Exception in initializeSinCosTables: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def sinRad(self, angle_rad):
    """ for positive only angles...""" 
    return self.sin_rad[int((angle_rad * self.blocksize_over_two_times_pi) % self.symbol_block_size)]

  def cosRad(self, angle_rad):
    """ for positive only angles...""" 
    return self.cos_rad[int((angle_rad * self.blocksize_over_two_times_pi) % self.symbol_block_size)]
   

  def sinRadTest(self):

    try:

      self.two_times_pi_times_blocksize = self.two_times_pi * self.symbol_block_size
      self.blocksize_over_two_times_pi = (self.symbol_block_size / (2*np.pi))

      start = datetime.now()

      self.debug.info_message("symbol_block_size: " + str(self.symbol_block_size))

      for angle_deg in range (361):
        angle_rad = ((angle_deg / 360) * 2 * np.pi) 
        self.debug.info_message("angle_deg: " + str(angle_deg))
        self.debug.info_message("angle_rad: " + str(angle_rad))
        normal_cos = np.cos(angle_rad)
        self.debug.info_message("normal cos angle_rad: " + str(normal_cos))
        self.debug.info_message("table cos angle_rad: " + str(self.cosRad(angle_rad)))

      start = datetime.now()
      angle_rad = np.pi
      for angle_deg in range (10000):
        normal_cos = np.cos(angle_rad)
      now = datetime.now()
      self.debug.info_message("elapsed time regular cos funciton: " + str(now-start))

      start = datetime.now()
      angle_rad = np.pi

      for angle_deg in range (10000):
        normal_cos = self.cosRad(angle_rad)
      now = datetime.now()
      self.debug.info_message("elapsed time table cos funciton: " + str(now-start))


    except:
      self.debug.error_message("Exception in sinRadTest: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def startTimer(self, name):
    self.debug.info_message("startTimer")
    try:
      self.timer_dict = {}
      self.timer_dict[name] = datetime.now()
    except:
      self.debug.error_message("Exception in startTimer: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def getDuration(self, name):
    self.debug.info_message("getDuration")
    try:
      return datetime.now() - self.timer_dict[name] 
    except:
      self.debug.error_message("Exception in getDuration: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def getDurationAndReset(self, name):
    self.debug.info_message("getDurationAndReset")
    try:
      elapsed = datetime.now() - self.timer_dict[name] 
      self.timer_dict[name] = datetime.now()
      return elapsed
    except:
      self.debug.error_message("Exception in getDurationAndReset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

     
  def startEncoder(self, text, mode):
    self.debug.info_message("startEncoder")

    try:
      self.setInitializationBlock(mode)

      self.debug.info_message("encoding text: " + str(text))
      triplets = self.text_encoder(text)
      for triplet in triplets:
        self.debug.info_message("pushing triplet: " + str(triplet) )
        self.pushDataQueue(triplet)

      self.initOutputStream(self.sample_rate)

    except:
      self.debug.error_message("Exception in startEncoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def startDecoder(self, mode, window):
    self.debug.info_message("startDecoder")

    self.runDecoder = True

    try:
      self.setInitializationBlock(mode)
      self.initInputStream(self.sample_rate, window)

    except:
      self.debug.error_message("Exception in startDecoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def stopEncoder(self):
    self.debug.info_message("stopEncoder")

    if self.outStreamRunning == True:
      self.outStream.stop()
      self.outStreamRunning = False


  def stopDecoder(self):
    self.debug.info_message("stopDecoder")
    self.runDecoder = False

    # leave the in stream running
    if self.inStreamRunning == True:
      self.inStream.stop()
      self.inStreamRunning = False


  def resetInputBuffer(self):
    self.inputBuffer = Queue()
    self.inputBufferItemCount = 0
    return self.inputBuffer

  def getInputBuffer(self):
    return self.inputBuffer

  def pushInputBuffer(self, data):    
    self.debug.info_message("pushInputBuffer")
    self.inputBufferItemCount += 1
    self.inputBuffer.put(data)
    self.debug.info_message("pushInputBuffer count: " + str(self.inputBufferItemCount))

  def popInputBuffer(self):
    self.inputBufferItemCount -= 1
    return self.inputBuffer.get_nowait()

  def isInputBufferEmpty(self):
    return self.inputBuffer.empty()

  def getInputBufferItemCount(self):
    return self.inputBufferItemCount


  def modParams(self, frequency, data):
    return {'frequency': frequency, 'data':data}

  def modParamsGetFrequency(self, modParams):
    return modparams['frequency']

  def modParamsGetData(self, modParams):
    return modparams['data']

  def getDataQueue(self):
    return self.dataQueue

  def pushDataQueue(self, data):    
    self.dataQueue.put(data)

  def popDataQueue(self):
    return self.dataQueue.get_nowait()

  def isDataQueueEmpty(self):
    return self.dataQueue.empty()

  def initInputStream(self, sample_rate, window):
    self.debug.info_message("initInputStream" )

    if self.symbol_block_size != self.previousBlocksizeIn or self.inStreamRunning == False:
      self.previousBlocksizeIn = self.symbol_block_size

      if self.inStreamRunning == True:
        self.inStream.stop()

      self.resetInputBuffer()
      self.inStream = sd.InputStream(samplerate=self.sample_rate, channels = 1, blocksize=self.symbol_block_size,
                                     dtype=np.float32, callback = self.sd_instream_callback)
      self.inStream.start()
      self.inStreamRunning = True

    """ start the decoder thread """
    t1 = threading.Thread(target=self.decoder_thread, args=(window, ))
    t1.start()

  def initOutputStream(self, sample_rate):
    self.debug.info_message("initOutputStream" )

    if self.symbol_block_size != self.previousBlocksizeOut or self.outStreamRunning == False:
      self.previousBlocksizeOut = self.symbol_block_size

      if self.outStreamRunning == True:
        self.outStream.stop()

      self.outStream = sd.OutputStream(samplerate=self.sample_rate, blocksize=self.symbol_block_size,
                                       channels=1, callback=self.sd_callback, dtype=np.float32)

      self.outStream.start()
      self.outStreamRunning = True


  def sd_instream_callback(self, indata, frames, time, status):
    self.debug.info_message("sd_instream_callback")

    """ send data to demodulator """
    if self.runDecoder == True:
      self.pushInputBuffer(indata)

    """ send data to spectrum display """
    if self.inStreamRunning == True:
      if self.spectral_density_queue_counter == 0:
        self.spectral_density_block = indata
        self.spectral_density_queue_counter += 1
      elif self.spectral_density_queue_counter > 4:
        self.debug.info_message("pushing data for spectral density plot")
        self.spectral_density_block = np.append(self.spectral_density_block, indata)
        self.form_gui.spectralDensityQueue.put(self.spectral_density_block)
        self.spectral_density_queue_counter = 0
      else:
        self.spectral_density_block = np.append(self.spectral_density_block, indata)
        self.spectral_density_queue_counter += 1


    return None

  def decoder_thread(self, window):
    self.debug.info_message("decoder_thread")

    while self.runDecoder == True:
      num_items = self.getInputBufferItemCount()
      if  num_items >= 5:
        self.debug.info_message("we have 10 items in queue...starting decode")
        multi_block = []
        for i in range(0, num_items ): 
          block = self.popInputBuffer()
          if i == 0:
            multi_block = block
          else:
            multi_block = np.append(multi_block, block)

        self.decoder_callback(multi_block)
      else:
        time.sleep(1)


  """ realtime conversion of pre-processed bitstream into wave audio. input data queued in FIFO buffer """
  def sd_callback(self, outdata, frames, time, status):

    return self.encoder_callback(outdata, frames, time, status)
    
  def openConstellationPlot(self):    

    m = 16
    constellation = exp(1j * arange(0, 2 * pi, 2 * pi / m))

    plt.scatter(constellation.real, constellation.imag)

    plt.title('Constellation')
    plt.grid()
    plt.show()

    

