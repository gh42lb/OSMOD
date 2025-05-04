#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import sounddevice as sd
import numpy as np
#import matplotlib.pyplot as plt
import threading
import sys
import gc

from numpy import pi
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter

from modulators import ModulatorPSK
from demodulators import DemodulatorPSK
from osmod_2fsk_8psk import mod_2FSK8PSK, demod_2FSK8PSK
#from osmod_2fsk_4psk import mod_2FSK4PSK, demod_2FSK4PSK
from modem_core_utils import ModemCoreUtils
from queue import Queue
from datetime import datetime, timedelta

from osmod_dictionary import PersistentData
from osmod_analysis import OsmodAnalysis

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

    self.opd = PersistentData(self)

    self.analysis = OsmodAnalysis(self)

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

    #self.mod_2fsk4psk   = mod_2FSK4PSK(self)
    #self.demod_2fsk4psk = demod_2FSK4PSK(self)

    self.dataQueue = Queue()
    self.inputBuffer = Queue()

    self.two_times_pi = 2 * np.pi

    self.timer_dict_when = {}
    self.timer_dict_elapsed = {}
    self.timer_last_name = ''


    """ initialize the initialization blocks for the different modulations"""

    """ New standard naming convention for LB28 modes: LB28-<pulses_per_block>-<num_carriers>-<carrier_separation>-<I,N,O or E> I=Interpolated, N=Normal, O=Orthogonal E=Experimental."""
    """ For non-standard block size LB28 modes: LB28-<block_size>-<pulses_per_block>-<num_carriers>-<carrier_separation>-<I,N,O or E> I=Interpolated, N=Normal, O=Orthogonal E=Experimental."""
    """ The baud rate and other details are in the info section """
    #self.modulation_initialization_block = {'LB28-0.15625-10I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
    self.modulation_initialization_block = {'LB28-2048-2-10-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 409600,
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
                                                              'pulses_per_block'     : 2048,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                        'LB28-2048-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 409600,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 2048,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                 'LB28-204800-2048-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 204800,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 2048,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                 'LB28-102400-2048-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 102400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 2048,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


                                        'LB28-1024-2-10-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 204800,
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
                                                              'pulses_per_block'     : 1024,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                        'LB28-1024-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 204800,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 1024,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.64, 0.8, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                 'LB28-102400-1024-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 102400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 1024,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-51200-1024-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 1024,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                         'LB28-512-2-10-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 102400,
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
                                                              'pulses_per_block'     : 512,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                         'LB28-512-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 102400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 512,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-51200-512-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 512,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-25600-512-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 25600,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 512,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.87, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


                                         'LB28-256-2-10-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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

                                         'LB28-256-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 256,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-25600-256-2-15-I':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                                                              'symbol_block_size'    : 25600,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (64/256, 192/256),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'pulses_per_block'     : 256,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


#                                        'LB28-0.3125-10I':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                        'LB28-128-2-10-I':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                        'LB28-128-2-15-I':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-2, 2, -2, 2),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 128,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                    'LB28-6400-128-2-15-I':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '0.3125 characters per second, 1.875 baud (bits per second)',
                                                              'symbol_block_size'    : 6400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (32/128, 96/128),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-2, 2, -2, 2),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 128,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

#                                        'LB28-0.625-10I':    {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                        'LB28-64-2-10-I':    {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                        'LB28-64-2-15-I':    {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 64,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-6400-64-2-15-I':    {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '1.25 characters per second, 7.5 baud (bits per second)',
                                                              'symbol_block_size'    : 6400,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (16/64, 48/64),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 64,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.68, 0.89, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

#                                        'LB28-1.25-10I':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                        'LB28-32-2-10-I':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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

                                  'LB28-3200-32-2-15-I':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '2.5 characters per second, 15 baud (bits per second)',
                                                              'symbol_block_size'    : 3200,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.thirtyseconds_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (8/32, 24/32),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 32,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.70, 0.94, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

#                                      'LB28-2.5-10I':        {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                      'LB28-16-2-10-I':        {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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

                                      'LB28-16-2-15-I':        {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
                                                              'carrier_separation'   : 15,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'phase_extraction'     : ocn.EXTRACT_INTERPOLATE,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 16,
                                                              'process_debug'        : False,
                                                              'parameters'           : (600, 0.67, 0.9, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

#                                             'LB28-10-20':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                          'LB28-4-2-20-N':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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
#                                              'LB28-5-10':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                          'LB28-8-2-10-N':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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

                                      'LB28-320-8-2-50-N':   {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '25 characters per second, 150 baud (bits per second)',
                                                              'extraction_points'    : (2/8, 6/8),
                                                              'symbol_block_size'    : 320,
                                                              'symbol_wave_function' : self.mod_2fsk8psk.eighths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 50,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 8,
                                                              'parameters'           : (600, 0.8, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

#                                              'LB28-10-40':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                           'LB28-4-2-40-N':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
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

                                      'LB28-160-4-2-100-N':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '50 characters per second, 300 baud (bits per second)',
                                                              'symbol_block_size'    : 160,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.fourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (0/4, 2/4),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 100,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 4,
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                      'LB28-160-4-2-50-N':  {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                                                              'info'                 : '50 characters per second, 300 baud (bits per second)',
                                                              'symbol_block_size'    : 160,
                                                              'symbols_per_block'    : 1,  # per carrier!
                                                              'symbol_wave_function' : self.mod_2fsk8psk.fourths_symbol_wave_function,
                                                              'modulation_object'    : self.mod_2fsk8psk,
                                                              'demodulation_object'  : self.demod_2fsk8psk,
                                                              'extraction_points'    : (0/4, 2/4),
                                                              'sample_rate'          : 8000,
                                                              'num_carriers'         : 2,
                                                              'carrier_separation'   : 50,
                                                              'detector_function'    : 'mode',
                                                              'baseband_conversion'  : 'costas_loop',
                                                              'process_debug'        : False,
                                                              'phase_extraction'     : ocn.EXTRACT_NORMAL,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 4,
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


#                                          'LB28-20-100':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                       'LB28-2-2-100-N':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                                                              'info'                 : 'Double Carrier 8psk 64 bit characters:- 20 characters per second, 120 baud (bits per second)',
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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-240-2-2-100-N':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                                                              'info'                 : '33.33 characters per second, 200 baud (bits per second)',
                                                              'symbol_block_size'    : 240,
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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-160-2-2-100-N':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                                                              'info'                 : '50 characters per second, 300 baud (bits per second)',
                                                              'symbol_block_size'    : 160,
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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

                                   'LB28-80-2-2-100-N':     {'encoder_callback'     : self.mod_2fsk8psk.encoder_8psk_callback,
                                                              'decoder_callback'     : self.demod_2fsk8psk.demodulate_2fsk_8psk,
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTriplet,
                                                              'text_decoder'         : self.demod_2fsk8psk.displayTextResults,
                                                              'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                                                              'info'                 : '100 characters per second, 600 baud (bits per second)',
                                                              'symbol_block_size'    : 80,
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
    #self.setInitializationBlock('LB28-4-2-40-N')


  def setInitializationBlock(self, mode):
    self.debug.info_message("setInitializationBlock")
    try:
      gc.collect()

      self.has_invalid_decodes = False

      self.info     = self.modulation_initialization_block[mode]['info']

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

      blocksize_override_checked = self.form_gui.window['cb_override_blocksize'].get()
      if blocksize_override_checked:
        self.symbol_block_size = int(self.form_gui.window['in_symbol_block_size'].get())

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
        """ calculate the RRC coefficients for double carrier"""
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
      else:
        """ These modes are only available in C compiled code. Only need to create RRC shape."""
        divisor = int(self.pulses_per_block)
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), self.parameters[1], self.parameters[2], self.sample_rate)

      """ initialize the sin cosine optimization lookup tables"""
      self.radTablesInitialize()
      #self.sinRadTest()

    except:
      self.debug.error_message("Exception in setInitializationBlock: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def updateCachedSettings(self, values, form_gui):
    try:
      self.opd.main_settings = { 'params': {
                             'LB28-2-2-100-N'         : self.getPersistentData('LB28-2-2-100-N',   values),
                             'LB28-160-2-2-100-N'     : self.getPersistentData('LB28-160-2-2-100-N',   values),
                             'LB28-240-2-2-100-N'     : self.getPersistentData('LB28-240-2-2-100-N',   values),
                             'LB28-2-2-100-N'         : self.getPersistentData('LB28-2-2-100-N',   values),
                             'LB28-160-4-2-100-N'      : self.getPersistentData('LB28-160-4-2-100-N',    values),
                             'LB28-160-4-2-50-N'      : self.getPersistentData('LB28-160-4-2-50-N',    values),
                             'LB28-4-2-40-N'          : self.getPersistentData('LB28-4-2-40-N',    values),
                             'LB28-4-2-20-N'          : self.getPersistentData('LB28-4-2-20-N',    values),
                             'LB28-320-8-2-50-N'          : self.getPersistentData('LB28-320-8-2-50-N',    values),
                             'LB28-8-2-10-N'          : self.getPersistentData('LB28-8-2-10-N',    values),
                             'LB28-16-2-10-I'         : self.getPersistentData('LB28-16-2-10-I',   values),
                             'LB28-16-2-15-I'         : self.getPersistentData('LB28-16-2-15-I',   values),
                             'LB28-3200-32-2-15-I'    : self.getPersistentData('LB28-3200-32-2-15-I',   values),
                             'LB28-32-2-10-I'         : self.getPersistentData('LB28-32-2-10-I',   values),
                             'LB28-6400-64-2-15-I'    : self.getPersistentData('LB28-6400-64-2-15-I',   values),
                             'LB28-64-2-15-I'         : self.getPersistentData('LB28-64-2-15-I',   values),
                             'LB28-64-2-10-I'         : self.getPersistentData('LB28-64-2-10-I',   values),
                             'LB28-6400-128-2-15-I'        : self.getPersistentData('LB28-6400-128-2-15-I',  values),
                             'LB28-128-2-15-I'        : self.getPersistentData('LB28-128-2-15-I',  values),
                             'LB28-128-2-10-I'        : self.getPersistentData('LB28-128-2-10-I',  values),
                             'LB28-25600-256-2-15-I'  : self.getPersistentData('LB28-25600-256-2-15-I',  values),
                             'LB28-256-2-15-I'        : self.getPersistentData('LB28-256-2-15-I',  values),
                             'LB28-256-2-10-I'        : self.getPersistentData('LB28-256-2-10-I',  values),
                             'LB28-25600-512-2-15-I'        : self.getPersistentData('LB28-25600-512-2-15-I',  values),
                             'LB28-51200-512-2-15-I'        : self.getPersistentData('LB28-51200-512-2-15-I',  values),
                             'LB28-512-2-15-I'        : self.getPersistentData('LB28-512-2-15-I',  values),
                             'LB28-512-2-10-I'        : self.getPersistentData('LB28-512-2-10-I',  values),
                             'LB28-1024-2-15-I'       : self.getPersistentData('LB28-1024-2-15-I', values),
                             'LB28-51200-1024-2-15-I'       : self.getPersistentData('LB28-51200-1024-2-15-I', values),
                             'LB28-102400-1024-2-15-I'       : self.getPersistentData('LB28-102400-1024-2-15-I', values),
                             'LB28-1024-2-10-I'       : self.getPersistentData('LB28-1024-2-10-I', values),
                             'LB28-2048-2-15-I'       : self.getPersistentData('LB28-2048-2-15-I', values),
                             'LB28-102400-2048-2-15-I'       : self.getPersistentData('LB28-102400-2048-2-15-I', values),
                             'LB28-204800-2048-2-15-I'       : self.getPersistentData('LB28-204800-2048-2-15-I', values),
                             'LB28-2048-2-10-I'       : self.getPersistentData('LB28-2048-2-10-I', values),
                }           }
      
      self.debug.info_message("updateCachedSettings: " + str(self.opd.main_settings))
 
    except:
      self.debug.error_message("Exception in updateCachedSettings: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

 
 
  def getPersistentData(self, mode, values):

    self.debug.info_message("getPersistentData mode: " + str(mode))

    try:
      selected_mode = values['combo_main_modem_modes']

      """ Test to see if this mode is the one currently selected for on screen options """
      if selected_mode == mode:
        self.debug.info_message("getting data from screen for mode")
        return_data = (
          self.form_gui.window['cb_enable_awgn'].get(),
          self.form_gui.window['combo_text_options'].get(),
          self.form_gui.window['cb_override_blocksize'].get(),
          self.form_gui.window['in_symbol_block_size'].get(),
          self.form_gui.window['combo_chunk_options'].get(),
          self.form_gui.window['cb_enable_align'].get(),
          self.form_gui.window['option_carrier_alignment'].get(),
          self.form_gui.window['cb_enable_separation_override'].get(),
          self.form_gui.window['cb_display_phases'].get(),
          self.form_gui.window['option_chart_options'].get(),
          self.form_gui.window['cb_override_rrc_alpha'].get(),
          self.form_gui.window['in_rrc_alpha'].get(),
          self.form_gui.window['cb_override_rrc_t'].get(),
          self.form_gui.window['in_rrc_t'].get(),
          values['btn_slider_awgn'],
          values['slider_amplitude'],
          values['slider_carrier_separation']
                        )
      else:
        self.debug.info_message("getting data from defaults")
        main_settings = self.form_gui.osmod.opd.main_settings
        params = main_settings.get('params')

        if mode not in params:
          """ perform automatic upgrade to new format """
          self.setFromCodeDefaults(mode)

        #if mode in params:
        settings = params.get(mode)
        self.debug.info_message("setScreenOptions retrieved data: " + str(settings))
        return_data = settings
        """
        else:
          return_data = (
            False,
            '3:peter piper',
            False,
            self.modulation_initialization_block[mode]['symbol_block_size'],
            '30',
            True,
            '100',
            True,
            True,
            'Both',
            False,
            '0.7',
            False,
            '0.9',
            '8.0',
            '1.0',
            '15'
                        )
        """                 

      return return_data

    except:
      self.debug.error_message("Exception in getPersistentData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


    
  def setScreenOptions(self, mode, form_gui, main_settings):    
    self.debug.info_message("setScreenOptions")

    try:

      self.debug.info_message("setScreenOptions main_settings: " + str(main_settings))

      params = main_settings.get('params')
      if mode not in params:
        """ perform automatic upgrade to new format """
        self.setFromCodeDefaults(mode)

      settings = params.get(mode)
      self.debug.info_message("setScreenOptions retrieved data: " + str(settings))

      form_gui.window['cb_enable_awgn'].update(settings[0])
      form_gui.window['combo_text_options'].update(settings[1])
      form_gui.window['cb_override_blocksize'].update(settings[2])
      form_gui.window['in_symbol_block_size'].update(settings[3])
      form_gui.window['combo_chunk_options'].update(settings[4])
      form_gui.window['cb_enable_align'].update(settings[5])
      form_gui.window['option_carrier_alignment'].update(settings[6])
      form_gui.window['cb_enable_separation_override'].update(settings[7])
      form_gui.window['cb_display_phases'].update(settings[8])
      form_gui.window['option_chart_options'].update(settings[9])
      form_gui.window['cb_override_rrc_alpha'].update(settings[10])
      form_gui.window['in_rrc_alpha'].update(settings[11])
      form_gui.window['cb_override_rrc_t'].update(settings[12])
      form_gui.window['in_rrc_t'].update(settings[13])
      form_gui.window['btn_slider_awgn'].update(settings[14])
      form_gui.window['slider_amplitude'].update(settings[15])
      form_gui.window['slider_carrier_separation'].update(settings[16])
    except:
      self.debug.error_message("Exception in setScreenOptions: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def resetMode(self, form_gui, mode):

    form_gui.window['cb_enable_awgn'].update(True)
    form_gui.window['combo_text_options'].update('3:peter piper')
    form_gui.window['cb_override_blocksize'].update(False)
    form_gui.window['in_symbol_block_size'].update(self.modulation_initialization_block[mode]['symbol_block_size'])
    form_gui.window['combo_chunk_options'].update('30')
    form_gui.window['cb_enable_align'].update(True)
    form_gui.window['option_carrier_alignment'].update('100')
    form_gui.window['cb_enable_separation_override'].update(True)
    form_gui.window['cb_display_phases'].update(False)
    form_gui.window['option_chart_options'].update('Both')
    form_gui.window['cb_override_rrc_alpha'].update(False)
    form_gui.window['in_rrc_alpha'].update(self.modulation_initialization_block[mode]['parameters'][1])
    form_gui.window['cb_override_rrc_t'].update(False)
    form_gui.window['in_rrc_t'].update(self.modulation_initialization_block[mode]['parameters'][2])
    form_gui.window['btn_slider_awgn'].update(8)
    form_gui.window['slider_amplitude'].update(1.0)
    form_gui.window['slider_carrier_separation'].update(self.modulation_initialization_block[mode]['carrier_separation'])

    self.setFromCodeDefaults(mode)


  def setFromCodeDefaults(self, mode):

    updated_data = (
            True,
            '3:peter piper',
            False,
            self.modulation_initialization_block[mode]['symbol_block_size'],
            '30',
            True,
            '100',
            True,
            False,
            'Both',
            False,
            self.modulation_initialization_block[mode]['parameters'][1],
            False,
            self.modulation_initialization_block[mode]['parameters'][2],
            '8.0',
            '1.0',
            self.modulation_initialization_block[mode]['carrier_separation']
                   )
    self.opd.main_settings.get('params')[mode] = updated_data


  def writeModeToCache(self, mode, form_gui, values):
    updated_data = (
          self.form_gui.window['cb_enable_awgn'].get(),
          self.form_gui.window['combo_text_options'].get(),
          self.form_gui.window['cb_override_blocksize'].get(),
          self.form_gui.window['in_symbol_block_size'].get(),
          self.form_gui.window['combo_chunk_options'].get(),
          self.form_gui.window['cb_enable_align'].get(),
          self.form_gui.window['option_carrier_alignment'].get(),
          self.form_gui.window['cb_enable_separation_override'].get(),
          self.form_gui.window['cb_display_phases'].get(),
          self.form_gui.window['option_chart_options'].get(),
          self.form_gui.window['cb_override_rrc_alpha'].get(),
          self.form_gui.window['in_rrc_alpha'].get(),
          self.form_gui.window['cb_override_rrc_t'].get(),
          self.form_gui.window['in_rrc_t'].get(),
          values['btn_slider_awgn'],
          values['slider_amplitude'],
          values['slider_carrier_separation']
                        )

    self.opd.main_settings.get('params')[mode] = updated_data


  def calcCarrierFrequencies(self, center_frequency, separation_override):
    self.debug.info_message("calcCarrierFrequencies")
    try:

      enable_align_checked = self.form_gui.window['cb_enable_align'].get()
      carrier_alignment = self.form_gui.window['option_carrier_alignment'].get()

      enable_separation_override_checked = self.form_gui.window['cb_enable_separation_override'].get()
      #separation_override = self.form_gui.window['option_separation_options'].get()
      #separation_override = self.form_gui.window['slider_carrier_separation'].get()
      self.debug.info_message("separation_override: " + str(separation_override))

      frequency = []
      span = (self.num_carriers-1) * self.carrier_separation
      if span > 0:
        for i in range(0, self.num_carriers):
          temp_freq = center_frequency - int(span/2) + (i * self.carrier_separation) 
          """ frequency must be on a 20Hz boundary for the 20 characters per second mode to work correctly """
          if enable_align_checked and i == 0:
            temp_freq = temp_freq // int(carrier_alignment)
            temp_freq = temp_freq * int(carrier_alignment)
          else:
            if enable_separation_override_checked:
              temp_freq = frequency[i-1] + int(separation_override)
            else:
              temp_freq = frequency[i-1] + span

          frequency.append(temp_freq)
      else:
        frequency.append(center_frequency)

      self.debug.info_message("calcCarrierFrequencies. frequencies: " + str(frequency))

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
      self.timer_dict_when[name] = datetime.now()
      self.timer_dict_elapsed[name] = datetime.now() - datetime.now()
      self.timer_last_name = name
    except:
      self.debug.error_message("Exception in startTimer: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def getDurationAndReset(self, name):
    self.debug.info_message("getDurationAndReset")
    try:
      elapsed = datetime.now() - self.timer_dict_when[self.timer_last_name] 
      if name in self.timer_dict_when:
        self.timer_dict_when[name] = datetime.now()
        self.timer_dict_elapsed[name] = self.timer_dict_elapsed[name] + elapsed
        self.debug.info_message("total elapsed time for " + name + ": " + str(elapsed))
      else:
        self.timer_dict_when[name] = datetime.now()
        self.timer_dict_elapsed[name] = datetime.now() - self.timer_dict_when[self.timer_last_name] 
        self.debug.info_message("elapsed time for " + name + ": " + str(elapsed))

      self.timer_last_name = name
      return elapsed
    except:
      self.debug.error_message("Exception in getDurationAndReset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def getSummary(self):
    self.debug.info_message("getSummary")
    for key in self.timer_dict_elapsed:
      self.debug.info_message("total elapsed time for " + key + ": " + str(self.timer_dict_elapsed[key]))

     
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

    

