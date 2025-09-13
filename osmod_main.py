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
import pyaudio

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
from osmod_detector import OsmodDetector
from osmod_simulations import OsmodSimulator
from osmod_interpolation import OsmodInterpolator
from osmod_test import OsmodTest
from osmod_fec import OsmodFEC

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

  fec_params = (400, 361, 2, 20, 500)

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

    self.analysis     = OsmodAnalysis(self)
    self.detector     = OsmodDetector(self)
    self.simulator    = OsmodSimulator(self)
    self.interpolator = OsmodInterpolator(self)
    self.test         = OsmodTest(self, form_gui.window)
    self.fec          = OsmodFEC(self, form_gui.window)

    self.form_gui = form_gui

    self.sample_rate = 4410 * 5 #44100
    self.attenuation = 30
    self.center_frequency = 1500
    self.symbols = 32
    self.bandwidth = 1000
    self.bits_per_symbol = int(np.log2(self.symbols))

    self.test_counter = 0

    self.mode = None

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

    """ generator polynomials for range 7 thru 21 defined in the following dictionary """
    fec_gp = { 7: (0o171, 0o133), 8: (0o235, 0o331), 9: (0o557, 0o663), 10: (0o473, 0o725), 11: (0o557, 0o731),
              12: (0o567, 0o723), 13: (0o4341, 0o6265), 14: (0o4561, 0o7065),15: (0o5561, 0o7571), 16: (0o61665, 0o75661),
              17: (0o464453, 0o520265), 18: (0o573441, 0o620261), 19: (0o5423125, 0o7151241), 20: (0o44525367, 0o56357123),
              21: (0o431526613, 0o616146743)}

    """ New standard naming convention for LB28 modes: LB28-<pulses_per_block>-<num_carriers>-<carrier_separation>-<I,N,O or E> I=Interpolated, N=Normal, O=Orthogonal E=Experimental."""
    """ For non-standard block size LB28 modes: LB28-<block_size>-<pulses_per_block>-<num_carriers>-<carrier_separation>-<I,N,O or E> I=Interpolated, N=Normal, O=Orthogonal E=Experimental."""
    """ The baud rate and other details are in the info section """

    """ parameters are defined as (phase extract threshold,
                                   RRC Alpha,
                                   RRC T,
                                   Baseband Normalization value,
                                   extract over num waves,
                                   peak at max %,
                                   Costas Loop Damping Factor
                                   Costas Loop Bandwidth,
                                   Costas Loop K1
                                   Costas Loop K2 
                                   extraction_filter_ratio,
                                   extraction_filter_inc,
                                   extraction gaussian filter sigma
                                   detector thrshold1,
                                   detector threshold2,

        I3_parameters are defined as (standing wave manual offsets[lower b,c, higher b, c],
                                      baseband convert frequency delta,


    """

    self.modulation_specific_pulse_shapes = {}
    #self.modulation_specific_pulse_shapes['LB28-6400-64-2-15-I3S3'] = [(0.161 , 0.209), (0.662 , 0.228), (0.783 , 0.282)]
    self.modulation_specific_pulse_shapes['LB28-6400-64-2-15-I3S3'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255)]
    #self.modulation_specific_pulse_shapes['LB28-6400-64-2-15-I3F']  = [(0.161 , 0.209), (0.662 , 0.228), (0.783 , 0.282)]
    self.modulation_specific_pulse_shapes['LB28-6400-64-2-15-I3F'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255)]
    self.modulation_specific_pulse_shapes['LB28-6400-64-2-15-I3E8'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255)]
    self.modulation_specific_pulse_shapes['LB28-6400-64-2-37-I3E8'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318) ]
    self.modulation_specific_pulse_shapes['LB28-6400-64-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]

    self.modulation_specific_pulse_shapes['LB28-12800-128-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-3200-32-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-1600-16-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-800-8-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]

    self.modulation_specific_pulse_shapes['LB28-25600-256-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-51200-512-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-102400-1024-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]

    #self.test_pulse_shapes = [(0.215 , 0.091), (0.763, 0.107), (0.992 , 0.276),(0.849 , 0.898),(0.586 , 0.78),(0.612 , 0.353),(0.63 , 0.206),(0.638 , 0.318)]

    #self.test_pulse_shapes = [(0.763, 0.107), (0.992 , 0.276),(0.849 , 0.898),(0.586 , 0.78),(0.612 , 0.353),(0.63 , 0.206),(0.638 , 0.318), (0.515 , 0.035),(0.662 , 0.228),(0.937 , 0.172),(0.523 , 0.068),(0.585 , 0.056),(0.308 , 0.024),(0.966 , 0.157),(0.215 , 0.091),(0.708 , 0.104),(0.92 , 0.025),(0.735 , 0.221),(0.161 , 0.209),(0.128 , 0.188),(0.096 , 0.157),(0.881 , 0.266),(0.783 , 0.282),(0.941 , 0.254),(0.778 , 0.177),(0.403 , 0.294),(0.68 , 0.096),(0.579 , 0.248)]

    self.test_pulse_shapes = [(0.595 , 0.255),(0.727 , 0.225),(0.491 , 0.29),(0.945 , 0.346),(0.954 , 0.289),(0.979 , 0.34),(0.657 , 0.173),(0.183 , 0.148),(0.552 , 0.288),(0.569 , 0.191),(0.692 , 0.241),(0.138 , 0.176),(0.285 , 0.268),(0.904 , 0.305),(0.412 , 0.224),(0.785 , 0.323),(0.658 , 0.241),(0.737 , 0.238),(0.336 , 0.223),(0.475 , 0.242),(0.604 , 0.295),(0.824 , 0.34),(0.519 , 0.177),(0.707 , 0.244),(0.557 , 0.252),(0.804 , 0.25),(0.601 , 0.175),(0.638 , 0.32),(0.833 , 0.309),(0.104 , 0.187),(0.887 , 0.322),(0.722 , 0.288),(0.352 , 0.165),(0.367 , 0.25)]

    self.best_pulse_shapes = [(0.104 , 0.187), (0.336 , 0.223), (0.804 , 0.25), (0.722 , 0.288), (0.595 , 0.255), (0.945 , 0.346), (0.737 , 0.238), (0.638 , 0.318), (0.662 , 0.228), (0.161 , 0.209), (0.612 , 0.353), (0.783 , 0.282), (0.735 , 0.221), (0.215 , 0.091)]

    self.all_pulse_shapes = [self.test_pulse_shapes, self.best_pulse_shapes]

    #[,(0.519 , 0.177),(0.569 , 0.191),(0.945 , 0.346),(0.785 , 0.323),(0.887 , 0.322),(0.824 , 0.34),(0.104 , 0.187),(0.183 , 0.148),(0.336 , 0.223),(0.475 , 0.242),(0.737 , 0.238),(0.804 , 0.25),(0.722 , 0.288),(0.904 , 0.305),(0.601 , 0.175),(0.595 , 0.255),(0.833 , 0.309),(0.557 , 0.252),(0.707 , 0.244),(0.979 , 0.34),(0.954 , 0.289),(0.138 , 0.176),(0.285 , 0.268),(0.604 , 0.295),(0.367 , 0.25),(0.657 , 0.173),(0.692 , 0.241),(0.491 , 0.29),(0.552 , 0.288)]

    self.test_sw_patterns_1 = [('A-D', 0.594), ('A-D', 0.657), ('A-D', 0.312), ('A-D', 0.562), ('A-D', 0.605), ('A-D', 0.827), ('A-D', 0.822), ('A-D', 0.373), ('A-D', 0.827)]
    self.test_sw_patterns_2 = [('C-C', 0.133),('B-C', 0.338),('C-C', 0.233),('C-C', 0.144),('C-E', 0.506),('C-C', 0.429),('A-D', 0.821),('A-D', 0.196),('A-D', 0.312),('A-D', 0.827),('A-D', 0.562),('A-D', 0.026),('A-D', 0.612),('A-D', 0.616)]
    self.test_sw_patterns_3  = [('A-D', 0.768),('A-D', 0.656),('A-D', 0.456),('A-D', 0.913),('A-D', 0.385),('A-D', 0.113),('A-D', 0.577),('A-D', 0.426),('A-D', 0.493),('A-D', 0.675),('A-D', 0.747),('A-D', 0.882),('A-D', 0.636),('B-C', 0.581),('A-C', 0.843),('A-D', 0.417),('B-C', 0.314),('A-D', 0.933),('A-D', 0.565),('A-D', 0.678),('A-D', 0.501),('A-D', 0.621),('A-D', 0.492),('A-D', 0.659),('A-D', 0.825),('A-D', 0.939)]
    self.test_sw_patterns = [self.test_sw_patterns_1, self.test_sw_patterns_2, self.test_sw_patterns_3]

    #self.best_sw_patterns_awgn8 = [('C-C', 0.144),('C-C', 0.429),('A-D', 0.493),('A-D', 0.562),('A-D', 0.768), ('A-D', 0.312), ('A-D', 0.417)]
    #self.best_sw_patterns_awgn6 = [('A-D', 0.417), ('C-C', 0.429), ('A-D', 0.822), ('B-C', 0.338), ('A-D', 0.312)]
    #self.best_sw_patterns_awgn4 = [('A-D', 0.827), ('A-D', 0.373), ('A-D', 0.939), ('A-D', 0.605), ('A-D', 0.312), ('A-D', 0.822)]
    #self.best_sw_patterns_awgn2 = 
    #self.best_sw_patterns_awgn0 = [('A-D', 0.312), ('A-D', 0.456), ('B-C', 0.338)]

    """ viterbi generator polynomial section """

    self.test_viterbi_gp_1 = [(15, 20568, 31282), (12, 289, 1620), (16, 3664, 6646), (13, 5890, 6271), (13, 4441, 1481), (15, 23443, 19340), (15, 29104, 5402)]
    self.test_viterbi_gp_2 = [(19 , 104171 , 43597),(16 , 25867 , 64068),(15 , 22748 , 20515),(18 , 49870 , 25915),(14 , 2670 , 13353),(19 , 267052 , 292632),(14 , 10117 , 10273),(16 , 19094 , 45608),(18 , 147012 , 171229),(17 , 59952 , 35021),(18 , 223244 , 166886),(19 , 165277 , 44828)]
    self.test_viterbi_gp_3 = [(15 , 4293 , 27673),(19 , 75437 , 395530),(13 , 567 , 1044),(18 , 175858 , 114619),(17 , 24678 , 113783),(15 , 14047 , 24322),(11 , 1504 , 330),(17 , 102984 , 108103),(17 , 49292 , 39302),(18 , 26581 , 142760),(11 , 320 , 517),(17 , 11484 , 2257),(13 , 7120 , 4105),(19 , 315807 , 489515),(18 , 16885 , 110965)]

    self.all_viterbi_gps  = [self.test_viterbi_gp_1, self.test_viterbi_gp_2, self.test_viterbi_gp_3]
    self.best_viterbi_gps = [(18 , 147012 , 171229), (15 , 4293 , 27673),(13 , 5890 , 6271),(15 , 14047 , 24322),(11 , 320 , 517)]

    """ initialize the initialization blocks for the different modulations"""
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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.64, 0.8, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.87, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.54, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
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
                                                              'parameters'           : (600, 0.54, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


                               'LB28-51200-512-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 512,
                                                              'symbol_block_size'    : 51200,
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!

                                                             },

                               'LB28-102400-1024-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 1024,
                                                              'symbol_block_size'    : 102400,
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!

                                                             },

                               'LB28-25600-256-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 256,
                                                              'symbol_block_size'    : 25600,
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!

                                                             },

                               # THIS WORKS...4 BITS PER SECOND
                               'LB28-12800-128-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 128,                                                                
                                                              'symbol_block_size'    : 6400,
                                                              'fft_filter'           : (-2, 2, -2, 2),                                                                
                                                              'extrapolate'          : 'yes', 
                                                              'fft_interpolate'      : (-3, 2, -2, 3),

                                                              'disposition_increment' : 5e-2,

                                                             },


                               #THIS WORKS GOOD   8 bits per second
                               'LB28-3200-32-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 32,
                                                              'symbol_block_size'    : 3200,
                                                              'extrapolate'          : 'yes', #no rotation tables yet!!!
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),

                                                             },

                               # THIS ACTUALLY WORKS!!!! 10 bits per second
                               'LB28-1600-16-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 16,
                                                              'symbol_block_size'    : 2560,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'extrapolate'          : 'yes', #no rotation tables yet!!!
                                                             },

                               # THIS PARTIALLY WORKS!!!!
                               'LB28-800-8-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 8,
                                                              'symbol_block_size'    : 2560,
                                                              #'symbol_block_size'    : 1600,

                                                              #'fft_filter'           : (-20, 16, -16, 20),
                                                              #'fft_filter'           : (-8, 8, -8, 8),

                                                              'fft_filter'           : (-6, 6, -6, 6),
                                                              #'fft_filter'           : (-4, 4, -4, 4),
                                                              #'fft_filter'           : (-3, 2, -2, 3),

                                                              'fft_interpolate'      : (-3, 2, -2, 3),

                                                              #'symbol_block_size'    : 800,
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!

                                                             },


                               'LB28-6400-64-2-37-I3E8-FEC':  {'inherit_from'        : 'LB28-6400-64-2-15-I3S3',
                                                              'text_encoder'         : self.mod_2fsk8psk.stringToTripletFEC,
                                                              'carrier_separation'   : 37,
                                                              'FEC'                  : ocn.FEC_VITERBI,
                                                              'fec_params'           : (13 , 5890 , 6271, []),

                                                              #'symbol_block_size'    : 3200,


                                                              #'fft_filter'           : (-1.8, 0.2, -0.2, 1.8),
                                                              #'fft_interpolate'      : (-3, 2, -2, 3),

                                                              #'fec_params'           : (12, 289, 1620, []), #BER 0.0 at 0.98 ebno
                                                              #'fec_params'           : (16, 3664, 6646, []), #BER 0.0 at 0.99                                                               
                                                              #'fec_params'           : (13, 5890, 6271, []), # BER 0.0 at 0.97
                                                              #'fec_params'           : (13, 4441, 1481, []), #BER 0.0 at 0.98
                                                              #'fec_params'           : (15, 23443, 19340, []), # BER 0.0 at 0.97
                                                              #'fec_params'           : (15, 29104, 5402, []), # BER 0.0 at 0.97

                                                              #'fec_params'           : (18, 2214, 1280, []), #BER0.0 at 2.7 not so reliable
                                                              #'fec_params'           : (18, 1146, 2619, []),  #BER 0.0 at 2.7

                                                              #'fec_params'           : (16, 841, 3524, []), #BER 0.0 at 2.7 conSISTENT


                                                              #'fec_params'           : (11, 0o133, 0o171),
                                                              #'fec_params'           : (12, fec_gp[12][1], fec_gp[12][0], []),
                                                              #'fec_params'           : (12, 1301, 2414, []), # BER 0.0 at 2.7 ebno. CONSISTENT!!!!
                                                              #'fec_params'           : (12, 1684, 2225, []), #BER 0.0 at 2.7 ebno

                                                              #'fec_params'           : (16, fec_gp[16][1], fec_gp[16][0], [1,0,1,1]), #2/3
                                                              #custom polynomials that work....
                                                              #'fec_params'           : (16, 12, 167, [1,0,1,1]), #2/3
                                                              #'fec_params'           : (16, 3805, 3370, [1,0,1,1]), #2/3

                                                              #'fec_params'           : (16, 256, 809, [1,1,1,0]), #2/3
                                                              #'fec_params'           : (16, ?, ?, [0,1,1,1]), #2/3
                                                              #'fec_params'           : (16, 256, 809, [1,1,0,1]), #2/3

                                                              #'fec_params'           : (16, 120, 595, [1,0,1,0,1,1,1,0,1,0]), #5/6

                                                              #these puncture codes dont work...
                                                              #'fec_params'           : (16, fec_gp[21][1], fec_gp[21][0], [1,0,1,1,1,0]), #3/4
                                                              #'fec_params'           : (12, fec_gp[12][1], fec_gp[12][0], [1,0,1,0,1,1,1,1,0,1]), #5/6
                                                              #'fec_params'           : (12, fec_gp[12][1], fec_gp[12][0], [1,0,0,0,1,0,1,1,1,1,1,1,0,1]), #7/8


                                                              #'post_extrapolate_calibrate': 'yes',

                                                              'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,


                                                              'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                                                              'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                                                              'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.312),
                                                                                                                              
                                                              'start_seq'            : '2_of_8',
                                                              'phase_align'          : 'start_seq',
                                                              'extrapolate'          : 'yes',
                                                              'extrapolate_seqlen'   : 8,
                                                              'downconvert_shift'    : 0.535,
                                                               
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MODULATION_SPECIFIC,
                                                              #'I3_pulse_shape_index' : 4, # can go to ebno 3.3
                                                              'I3_pulse_shape_index' : 5, # can go to ebno -0.16 to 2.7

                                                              'parameters'           : (1500, 0.763, 0.107, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

                                                              }, 
                                   'LB28-6400-64-2-37-I3E8':  {'inherit_from'        : 'LB28-6400-64-2-15-I3S3',
                                                              'carrier_separation'   : 37,
                                                              'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.312), 
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.311), 
                                                              'start_seq'            : '2_of_8',
                                                              'phase_align'          : 'start_seq',
                                                              'extrapolate'           : 'yes',
                                                              'extrapolate_seqlen'       : 8,
                                                              'downconvert_shift'    : 0.535,
                                                              #'downconvert_shift'    : 0.5,
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MODULATION_SPECIFIC,
                                                              'I3_pulse_shape_index' : 3,
                                                              'parameters'           : (1500, 0.763, 0.107, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                                                              }, 

                                   'LB28-6400-64-2-15-I3E8':  {'inherit_from'        : 'LB28-6400-64-2-15-I3S3',
                                                              'start_seq'            : '2_of_8',
                                                              'phase_align'          : 'start_seq',
                                                              'extrapolate'           : 'yes',
                                                              'extrapolate_seqlen'       : 8,
                                                              'I3_offsets_type'      : ocn.OFFSETS_PATTERN37,
                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN47,
                                                              #('A-D', 0.312)
                                                              #'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.312), 
                                                              }, 


                                   'LB28-6400-64-2-15-I3F':    {'inherit_from'        : 'LB28-6400-64-2-15-I3S3',
                                   #'LB28-6400-64-2-15-I3F':    {'inherit_from'        : 'LB28-6400-64-2-15-I',

                                                              'phase_rotation'       : (1,2),   #AWGN must be set to 1 or lower!!!
                                                              'phase_align'          : 'fixed_rotation', #YES THIS WORKS WHEN AWGN LOW!!!!

                                                              'pulse_start_sigma'    : 3,

                                                              }, 


                                   'LB28-6400-64-2-15-I3S3':    {'inherit_from'        : 'LB28-6400-64-2-15-I',
                                                              'phase_encoding'       : ocn.PHASE_INTRA_TRIPLE,

                                                              #'phase_rotation'       : (1,2),   #AWGN must be set to 1 or lower!!!
                                                              #'phase_align'          : 'fixed_rotation', #YES THIS WORKS WHEN AWGN LOW!!!!

                                                              #'extrapolate'           : 'yes',
                                                              #'extrapolate_seqlen'       : 8,

                                                              'start_seq'            : '2_of_8',
                                                              #'start_seq'            : '2_of_5',
                                                              #'start_seq'            : '2_of_4',
                                                              #'start_seq'            : '2_of_3',
                                                              'phase_align'          : 'start_seq',

                                                              #'doppler_pulse_interpolation' : 'B-Spline',
                                                              #'doppler_pulse_interpolation' : 'Chebyshev',
                                                              'doppler_pulse_interpolation' : 'Pchip',


                                                              'I3_combine'           : ocn.INTRA_COMBINE_TYPE10,
                                                              'I3_extract'           : ocn.INTRA_EXTRACT_TYPE5,
                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN5,
                                                              'I3_offsets_type'      : ocn.OFFSETS_PATTERN37,
                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN19,

                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN5,  # patterns 6, 9, 15, 14, 17  (top 5 - single)
                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN2,  # patterns 11, 27, 8, 6, 3  (top 5 - 3 in a row)
                                                                                                               # best overall: pattern 6

                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN2,  # patterns 15, 17, 35, 5, 4, 11, 27    (top 5 - single)
                                                              #'I3_offsets_type'      : ocn.OFFSETS_PATTERN2,  # patterns 18, 4, 5, 11, 29, 14, 15  (top 5 - 3 in a row)
                                                                                                               # best overall: pattern 5, 4, 11, 17, 27, 14, 15

                                                              #'I3_offsets_type'      : ocn.???????????,  # patterns 17

                                                              #'I3_pulse_type'        : ocn.TRIPLETS_SUPERPOSITION,
                                                              #'I3_pulse_type'        : ocn.TRIPLETS_SEQUENTIAL,

                                                              #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MODULATION_SPECIFIC,
                                                              #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_GENERAL,
                                                              #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_TEST,
                                                              'I3_pulse_shape_index' : 0,
           
                                                              'carrier_separation'   : 37,



                                                              'I3_pulse_alignment'   : ocn.I3_STANDINGWAVE_PULSE_1_OF_3,
                                                              'pulse_detection'      : ocn.PULSE_DETECTION_I3,
                                                              #'pulse_detection'      : ocn.PULSE_DETECTION_NORMAL,

                                                              #'fft_filter'           : (-4, 4, -4, 4),
                                                              #'fft_interpolate'      : (-3, 2, -2, 3),
                                                              #'fft_filter'           : (-1, 1, -1, 1),
                                                              #'fft_interpolate'      : (-2, 2, -2, 2),
                                                              'fft_filter'           : (-1, 1, -1, 1),
                                                              #'fft_interpolate'      : (-1, 1, -1, 1),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              #'fft_interpolate'      : (-7, 1, -1, 7),


                                                              'baseband_conversion'  : 'I3_rel_exp',
                                                              #'parameters'           : (1500, 0.548, 0.695, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'parameters'           : (1500, 0.763, 0.107, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),


                                                              #'parameters'           : (1500, 0.359, 0.715, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (1500, 0.84, 0.60, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (1500, 0.97, 0.481, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (1500, 0.78, 0.60, 10000, 6, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (1500, 0.84, 0.60, 10000, 6, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (1500, 0.84, 0.60, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (2500, 0.84, 0.60, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #'parameters'           : (2500, 0.88, 0.70, 10000, 200, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 0.625, 0.875, 0.09375, 0.78125), 
                                                              }, 




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
                                                              #'phase_encoding'       : ocn.PHASE_INTRA_TRIPLE,
                                                              'fft_filter'           : (-4, 4, -4, 4),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'pulses_per_block'     : 64,
                                                              'process_debug'        : False,
                                                              #'parameters'           : (600, 0.68, 0.89, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                                              'parameters'           : (1500, 0.68, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              #'parameters'           : (600, 0.70, 0.94, 10000, 2, 98) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                                              'parameters'           : (2000, 0.70, 0.94, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.67, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
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
                                                              'parameters'           : (600, 0.8, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.8, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (600, 0.75, 0.65, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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
                                                              'parameters'           : (700, 0.8, 0.6, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) }}  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves



    """ set some default values..."""
    #self.setInitializationBlock('LB28-4-2-40-N')


    """ structure that stores values of the optional params"""
    self.optional_param_values   = {}

    """ defaults are set here"""
    self.optional_param_defaults = {'phase_encoding'       : ocn.PHASE_INTRA_SINGLE,
                                    'phase_rotation'       : (0,0),
                                    'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                    'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.312), 
                                    'start_seq'            : 'none',
                                    'phase_align'          : 'disable',
                                    'I3_pulse_alignment'   : ocn.I3_STANDINGWAVE_PULSE_1_OF_3,
                                    'pulse_detection'      : ocn.PULSE_DETECTION_NORMAL,
                                    'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                    'I3_pulse_shape_index' : 0,
                                    'pulse_start_sigma'    : 7,
                                    'doppler_pulse_interpolation'    : 'Chebyshev',
                                    'extrapolate'          : 'no',
                                    'extrapolate_seqlen'   : 3,
                                    'downconvert_shift'    : 0.5,
                                    'FEC'                  : ocn.FEC_NONE,
                                    'fec_params'           : (), 
                                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                                    'msg_type'             : ocn.MSGTYPE_VARIABLE_LENGTH,
                                    'post_extrapolate_calibrate': 'no',
                                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                                    'disposition_increment' : 1e-1,

                                   }


  def resetOptionalInitParamDefaults(self, mode):
    for param in self.optional_param_defaults:
      self.optional_param_values[param] = self.optional_param_defaults[param]

  def processOptionalInitParams(self, mode):
    for param in self.optional_param_values:
      if param in self.modulation_initialization_block[mode]:
        self.optional_param_values[param] = self.modulation_initialization_block[mode][param]


  """ recursive... multi level inheritance"""
  def processInheritFrom(self, mode):
    self.debug.info_message("processInheritFrom mode: " + str(mode))
    if 'inherit_from' in self.modulation_initialization_block[mode]:
      self.processInheritFrom(self.modulation_initialization_block[mode]['inherit_from'])
    for param in self.modulation_initialization_block[mode]:
      self.optional_param_values[param] = self.modulation_initialization_block[mode][param]

  def getParam(self, mode, param_name):
    if param_name in self.modulation_initialization_block[mode]:
      param_value = self.modulation_initialization_block[mode][param_name]
      self.debug.info_message("param " + str(param_name) + " = " + str(param_value))
      return param_value  #  self.modulation_initialization_block[mode][param_name]
    else:
      param_value = self.optional_param_values[param_name]
      self.debug.info_message("param " + str(param_name) + " = " + str(param_value))
      return param_value  #  self.optional_param_values[param_name]

  def getOptionalParam(self, param):
    return self.optional_param_values[param]


  def setInitializationBlock(self, mode):
    self.debug.info_message("setInitializationBlock")
    try:
      gc.collect()

      self.mode = mode
      self.sequence_start_character_detected_low = False
      self.sequence_start_character_rotation_low = 0
      self.sequence_start_character_detected_high = False
      self.sequence_start_character_rotation_high = 0

      self.has_invalid_decodes = False

      self.resetOptionalInitParamDefaults(mode)
      self.processOptionalInitParams(mode)

      self.debug.info_message("optional_param_values (before inherit): " + str(self.optional_param_values))

      self.processInheritFrom(mode)

      self.debug.info_message("optional_param_values (after inherit): " + str(self.optional_param_values))

      self.info                 = self.getParam(mode, 'info')
      self.encoder_callback     = self.getParam(mode, 'encoder_callback')
      self.decoder_callback     = self.getParam(mode, 'decoder_callback')
      self.text_encoder         = self.getParam(mode, 'text_encoder')
      self.mode_selector        = self.getParam(mode, 'mode_selector')
      self.symbol_block_size    = self.getParam(mode, 'symbol_block_size')
      self.sample_rate          = self.getParam(mode, 'sample_rate')
      self.parameters           = self.getParam(mode, 'parameters')
      self.carrier_separation   = self.getParam(mode, 'carrier_separation')
      self.num_carriers         = self.getParam(mode, 'num_carriers')
      self.pulses_per_block     = self.getParam(mode, 'pulses_per_block')
      self.detector_function    = self.getParam(mode, 'detector_function')
      self.symbol_wave_function = self.getParam(mode, 'symbol_wave_function')
      self.extraction_points    = self.getParam(mode, 'extraction_points')
      self.phase_extraction     = self.getParam(mode, 'phase_extraction')
      self.baseband_conversion  = self.getParam(mode, 'baseband_conversion')
      self.process_debug        = self.getParam(mode, 'process_debug')
      self.fft_filter           = self.getParam(mode, 'fft_filter')
      self.fft_interpolate      = self.getParam(mode, 'fft_interpolate')
      self.modulation_object    = self.getParam(mode, 'modulation_object')
      self.demodulation_object  = self.getParam(mode, 'demodulation_object')
      self.phase_rotation       = self.getParam(mode, 'phase_rotation')
      self.i3_offsets_type      = self.getParam(mode, 'I3_offsets_type')
      self.i3_parameters        = self.getParam(mode, 'I3_parameters')
      self.start_seq            = self.getParam(mode, 'start_seq')
      self.phase_align          = self.getParam(mode, 'phase_align')
      self.i3_pulse_align       = self.getParam(mode, 'I3_pulse_alignment')
      self.pulse_detection      = self.getParam(mode, 'pulse_detection')
      self.I3_pulse_shape_type  = self.getParam(mode, 'I3_pulse_shape_type')
      self.I3_pulse_shape_index = self.getParam(mode, 'I3_pulse_shape_index')
      self.pulse_start_sigma    = self.getParam(mode, 'pulse_start_sigma')
      self.doppler_pulse_interpolation    = self.getParam(mode, 'doppler_pulse_interpolation')
      self.extrapolate          = self.getParam(mode, 'extrapolate')
      self.extrapolate_seqlen   = int(self.getParam(mode, 'extrapolate_seqlen'))
      self.downconvert_shift    = float(self.getParam(mode, 'downconvert_shift'))
      self.FEC                  = self.getParam(mode, 'FEC')
      self.fec_params           = self.getParam(mode, 'fec_params')
      self.msg_sections         = self.getParam(mode, 'msg_sections')
      self.msg_type             = self.getParam(mode, 'msg_type')
      self.post_extrapolate_calibrate = self.getParam(mode, 'post_extrapolate_calibrate')
      self.holographic_decode   = self.getParam(mode, 'holographic_decode')
      self.disposition_increment   = float(self.getParam(mode, 'disposition_increment'))


      self.fec.init_params(self.FEC)

      """ keep track of the chunks being processed """
      self.chunk_num = 0

      self.rotation_tables      = self.opd.readRotationTablesFromFile(mode)

      """
      self.info                 = self.modulation_initialization_block[mode]['info']
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
      """

      blocksize_override_checked = self.form_gui.window['cb_override_blocksize'].get()
      if blocksize_override_checked:
        self.symbol_block_size = int(self.form_gui.window['in_symbol_block_size'].get())

      samplerate_override_checked = self.form_gui.window['cb_enable_sample_rate_override'].get()
      if samplerate_override_checked:
        self.sample_rate = int(self.form_gui.window['in_sample_rate_override'].get())



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

      self.demodulation_object.remainder = np.array([])

      rrc_alpha, rrc_T = self.getPulseShape(mode)

      override_rrc_alpha = self.form_gui.window['cb_override_rrc_alpha'].get()
      override_rrc_T     = self.form_gui.window['cb_override_rrc_t'].get()
      if override_rrc_alpha:
        rrc_alpha = float(self.form_gui.window['in_rrc_alpha'].get())
      if override_rrc_T:
        rrc_T     = float(self.form_gui.window['in_rrc_t'].get())


      if self.pulses_per_block == 1:
        """ calculate RRC coefficients for single pulse"""
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size), rrc_alpha, rrc_T, self.sample_rate)
        self.filtRRC_wave1 = self.filtRRC_coef_main
        self.filtRRC_wave2 = self.filtRRC_coef_main # not required
      elif self.pulses_per_block == 2:
        """ calculate the RRC coefficients for double carrier"""
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/2), rrc_alpha, rrc_T, self.sample_rate)
        self.filtRRC_wave1 = np.append(self.filtRRC_coef_main, np.zeros(int(self.symbol_block_size/2)), )
        self.filtRRC_wave2 = np.append(np.zeros(int(self.symbol_block_size/2)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 4:
        """ calculate the RRC coefficients for quad carrier"""
        divisor = 4
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_fourth_wave = [0] * divisor
        self.filtRRC_fourth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_fourth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_fourth_wave[i] = np.append(self.filtRRC_fourth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_fourth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 8:
        """ calculate the RRC coefficients for eighth carrier"""
        divisor = 8
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_eighth_wave = [0] * divisor
        self.filtRRC_eighth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_eighth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_eighth_wave[i] = np.append(self.filtRRC_eighth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_eighth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 12:
        """ calculate the RRC coefficients for twelfth carrier"""
        divisor = 12
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_twelfth_wave = [0] * divisor
        self.filtRRC_twelfth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_twelfth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_twelfth_wave[i] = np.append(self.filtRRC_twelfth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_twelfth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 16:
        """ calculate the RRC coefficients for sixteenth carrier"""
        divisor = 16
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_sixteenth_wave = [0] * divisor
        self.filtRRC_sixteenth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_sixteenth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_sixteenth_wave[i] = np.append(self.filtRRC_sixteenth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_sixteenth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      elif self.pulses_per_block == 32:
        """ calculate the RRC coefficients for 32nds carrier"""
        divisor = 32
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_thirtysecond_wave = [0] * divisor
        self.filtRRC_thirtysecond_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_thirtysecond_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_thirtysecond_wave[i] = np.append(self.filtRRC_thirtysecond_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_thirtysecond_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 64:
        """ calculate the RRC coefficients for 64ths carrier"""
        divisor = 64
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_sixtyfourth_wave = [0] * divisor
        self.filtRRC_sixtyfourth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_sixtyfourth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_sixtyfourth_wave[i] = np.append(self.filtRRC_sixtyfourth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_sixtyfourth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 128:
        """ calculate the RRC coefficients for 128ths carrier"""
        divisor = 128
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_onehundredtwentyeighth_wave = [0] * divisor
        self.filtRRC_onehundredtwentyeighth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_onehundredtwentyeighth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_onehundredtwentyeighth_wave[i] = np.append(self.filtRRC_onehundredtwentyeighth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_onehundredtwentyeighth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

      elif self.pulses_per_block == 256:
        """ calculate the RRC coefficients for 256ths carrier"""
        divisor = 256
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_twohundredfiftysixth_wave = [0] * divisor
        self.filtRRC_twohundredfiftysixth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_twohundredfiftysixth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_twohundredfiftysixth_wave[i] = np.append(self.filtRRC_twohundredfiftysixth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_twohundredfiftysixth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)
      else:
        """ These modes are only available in C compiled code. Only need to create RRC shape."""
        divisor = int(self.pulses_per_block)
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

      """ initialize the sin cosine optimization lookup tables"""
      self.radTablesInitialize()
      #self.sinRadTest()

    except:
      self.debug.error_message("Exception in setInitializationBlock: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def getPulseShape(self, mode):
    self.debug.info_message("getPulseShape")

    try:
      if self.I3_pulse_shape_type == ocn.PULSE_SHAPE_MANUAL:
        #return (self.parameters[1], self.parameters[2])
        return self.parameters[1], self.parameters[2]
      elif self.I3_pulse_shape_type == ocn.PULSE_SHAPE_MODULATION_SPECIFIC:
        #return self.modulation_specific_pulse_shapes[mode][self.I3_pulse_shape_index]
        doublet = self.modulation_specific_pulse_shapes[mode][self.I3_pulse_shape_index]
        return doublet[0], doublet[1]
      elif self.I3_pulse_shape_type == ocn.PULSE_SHAPE_GENERAL:
        #return self.best_pulse_shapes[self.I3_pulse_shape_index]
        doublet = self.best_pulse_shapes[self.I3_pulse_shape_index]
        return doublet[0], doublet[1]
      elif self.I3_pulse_shape_type == ocn.PULSE_SHAPE_TEST:
        #return self.test_pulse_shapes[self.I3_pulse_shape_index]
        doublet = self.test_pulse_shapes[self.I3_pulse_shape_index]
        return doublet[0], doublet[1]

    except:
      self.debug.error_message("Exception in getPulseShape: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


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
                             'LB28-6400-64-2-15-I3S3'    : self.getPersistentData('LB28-6400-64-2-15-I3S3',   values),
                             'LB28-6400-64-2-15-I3F'    : self.getPersistentData('LB28-6400-64-2-15-I3F',   values),
                             'LB28-6400-64-2-15-I3E8'   : self.getPersistentData('LB28-6400-64-2-15-I3E8',   values),
                             'LB28-6400-64-2-37-I3E8'   : self.getPersistentData('LB28-6400-64-2-37-I3E8',   values),
                             'LB28-6400-64-2-37-I3E8-FEC' : self.getPersistentData('LB28-6400-64-2-37-I3E8-FEC',   values),
                             'LB28-25600-256-2-37-I3E8-FEC' : self.getPersistentData('LB28-25600-256-2-37-I3E8-FEC',   values),
                             'LB28-51200-512-2-37-I3E8-FEC' : self.getPersistentData('LB28-51200-512-2-37-I3E8-FEC',   values),
                             'LB28-102400-1024-2-37-I3E8-FEC' : self.getPersistentData('LB28-102400-1024-2-37-I3E8-FEC',   values),

                             'LB28-12800-128-2-37-I3E8-FEC' : self.getPersistentData('LB28-12800-128-2-37-I3E8-FEC',   values),
                             'LB28-3200-32-2-37-I3E8-FEC' : self.getPersistentData('LB28-3200-32-2-37-I3E8-FEC',   values),
                             'LB28-1600-16-2-37-I3E8-FEC' : self.getPersistentData('LB28-1600-16-2-37-I3E8-FEC',   values),
                             'LB28-800-8-2-37-I3E8-FEC' : self.getPersistentData('LB28-800-8-2-37-I3E8-FEC',   values),

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


  def calcCarrierFrequenciesFromFFT(self, fft_frequency, separation_override):
    self.debug.info_message("calcCarrierFrequenciesFromFFT")
    try:
      rxfrequencydelta = 0.0
      is_rxfrequencydelta_checked = self.form_gui.window['cb_enable_rxfrequencydelta'].get()
      if is_rxfrequencydelta_checked:
        rxfrequencydelta = float(self.form_gui.window['in_rxfrequencydelta'].get())

      enable_separation_override_checked = self.form_gui.window['cb_enable_separation_override'].get()
      self.debug.info_message("separation_override: " + str(separation_override))

      frequency = []
      span = (self.num_carriers-1) * self.carrier_separation
      if span > 0:
        for i in range(0, self.num_carriers):
          temp_freq = fft_frequency + (i * self.carrier_separation) 
          if i > 0:
            if enable_separation_override_checked:
              temp_freq = frequency[i-1] + int(separation_override)
            else:
              temp_freq = frequency[i-1] + span

          if i == 0:
            frequency.append(temp_freq + rxfrequencydelta)
          else:
            frequency.append(temp_freq)
      else:
        frequency.append(center_frequency)

      self.form_gui.window['text_info_freq1'].update(frequency[0])
      self.form_gui.window['text_info_freq2'].update(frequency[1])

      self.debug.info_message("calcCarrierFrequenciesFromFFT. frequencies: " + str(frequency))

      return frequency

    except:
      self.debug.error_message("Exception in calcCarrierFrequenciesFromFFT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calcCarrierFrequencies(self, center_frequency, separation_override):
    self.debug.info_message("calcCarrierFrequencies")
    try:
      def set_sw_values():
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, standingwave_location)
        if   standingwave_pattern == 'A-B':
          self.i3_offsets = phase_list_low[0], phase_list_low[1], phase_list_high[0], phase_list_high[1]
        elif standingwave_pattern == 'A-C':
          self.i3_offsets = phase_list_low[0], phase_list_low[2], phase_list_high[0], phase_list_high[2]
        elif standingwave_pattern == 'A-D':
          self.i3_offsets = phase_list_low[0], phase_list_low[3], phase_list_high[0], phase_list_high[3]
        elif standingwave_pattern == 'A-E':
          self.i3_offsets = phase_list_low[0], phase_list_low[4], phase_list_high[0], phase_list_high[4]
        elif standingwave_pattern == 'B-C':
          self.i3_offsets = phase_list_low[1], phase_list_low[2], phase_list_high[1], phase_list_high[2]
        elif standingwave_pattern == 'B-D':
          self.i3_offsets = phase_list_low[1], phase_list_low[3], phase_list_high[1], phase_list_high[3]
        elif standingwave_pattern == 'B-E':
          self.i3_offsets = phase_list_low[1], phase_list_low[4], phase_list_high[1], phase_list_high[4]
        elif standingwave_pattern == 'C-D':
          self.i3_offsets = phase_list_low[2], phase_list_low[3], phase_list_high[2], phase_list_high[3]
        elif standingwave_pattern == 'C-E':
          self.i3_offsets = phase_list_low[2], phase_list_low[4], phase_list_high[2], phase_list_high[4]
        elif standingwave_pattern == 'D-E':
          self.i3_offsets = phase_list_low[3], phase_list_low[4], phase_list_high[3], phase_list_high[4]
        elif standingwave_pattern == 'A-A':
          self.i3_offsets = phase_list_low[0], phase_list_low[0], phase_list_high[0], phase_list_high[0]
        elif standingwave_pattern == 'B-B':
          self.i3_offsets = phase_list_low[1], phase_list_low[1], phase_list_high[1], phase_list_high[1]
        elif standingwave_pattern == 'C-C':
          self.i3_offsets = phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
        elif standingwave_pattern == 'D-D':
          self.i3_offsets = phase_list_low[3], phase_list_low[3], phase_list_high[3], phase_list_high[3]
        elif standingwave_pattern == 'E-E':
          self.i3_offsets = phase_list_low[4], phase_list_low[4], phase_list_high[4], phase_list_high[4]


      enable_align_checked = self.form_gui.window['cb_enable_align'].get()
      carrier_alignment = self.form_gui.window['option_carrier_alignment'].get()

      rxfrequencydelta = 0.0
      is_rxfrequencydelta_checked = self.form_gui.window['cb_enable_rxfrequencydelta'].get()
      if is_rxfrequencydelta_checked:
        rxfrequencydelta = float(self.form_gui.window['in_rxfrequencydelta'].get())

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
          if i > 0:
            if enable_separation_override_checked:
              temp_freq = frequency[i-1] + int(separation_override)
            else:
              temp_freq = frequency[i-1] + span

          if i == 0:
            frequency.append(temp_freq + rxfrequencydelta)
          else:
            frequency.append(temp_freq)
      else:
        frequency.append(center_frequency)

      self.form_gui.window['text_info_freq1'].update(frequency[0])
      self.form_gui.window['text_info_freq2'].update(frequency[1])

      offsets_override_checked = self.form_gui.window['cb_override_standingwaveoffsets'].get()
      if offsets_override_checked:
        standingwave_location = float(self.form_gui.window['in_standingwavelocation'].get())
        standingwave_pattern = self.form_gui.window['combo_standingwave_pattern'].get()
        set_sw_values()

      else:
        if self.i3_offsets_type == ocn.OFFSETS_MANUAL:
          standingwave_pattern  = self.i3_parameters[3]
          standingwave_location = float(self.i3_parameters[4])
          set_sw_values()

          #self.i3_offsets = [self.i3_parameters[3], self.i3_parameters[4], self.i3_parameters[5], self.i3_parameters[6] ]
        else:
          pattern_override_checked = self.form_gui.window['cb_override_standingwavepattern'].get()
          if pattern_override_checked:
            patterns = {'Pattern 1': ocn.OFFSETS_PATTERN1, 'Pattern 2': ocn.OFFSETS_PATTERN2, 'Pattern 3': ocn.OFFSETS_PATTERN3, 'Pattern 4': ocn.OFFSETS_PATTERN4,
                        'Pattern 5': ocn.OFFSETS_PATTERN5, 'Pattern 6': ocn.OFFSETS_PATTERN6, 'Pattern 7': ocn.OFFSETS_PATTERN7, 'Pattern 8': ocn.OFFSETS_PATTERN8,
                        'Pattern 9': ocn.OFFSETS_PATTERN9, 'Pattern 10': ocn.OFFSETS_PATTERN10, 'Pattern 11': ocn.OFFSETS_PATTERN11, 'Pattern 12': ocn.OFFSETS_PATTERN12,
                        'Pattern 13': ocn.OFFSETS_PATTERN13, 'Pattern 14': ocn.OFFSETS_PATTERN14, 'Pattern 15': ocn.OFFSETS_PATTERN15, 'Pattern 16': ocn.OFFSETS_PATTERN16,
                        'Pattern 17': ocn.OFFSETS_PATTERN17, 'Pattern 18': ocn.OFFSETS_PATTERN18, 'Pattern 19': ocn.OFFSETS_PATTERN19, 'Pattern 20': ocn.OFFSETS_PATTERN20,
                        'Pattern 21': ocn.OFFSETS_PATTERN21, 'Pattern 22': ocn.OFFSETS_PATTERN22, 'Pattern 23': ocn.OFFSETS_PATTERN23, 'Pattern 24': ocn.OFFSETS_PATTERN24,
                        'Pattern 25': ocn.OFFSETS_PATTERN25, 'Pattern 26': ocn.OFFSETS_PATTERN26, 'Pattern 27': ocn.OFFSETS_PATTERN27, 'Pattern 28': ocn.OFFSETS_PATTERN28,
                        'Pattern 29': ocn.OFFSETS_PATTERN29, 'Pattern 30': ocn.OFFSETS_PATTERN30, 'Pattern 31': ocn.OFFSETS_PATTERN31, 'Pattern 32': ocn.OFFSETS_PATTERN32,
                        'Pattern 33': ocn.OFFSETS_PATTERN33, 'Pattern 34': ocn.OFFSETS_PATTERN34, 'Pattern 35': ocn.OFFSETS_PATTERN35, 'Pattern 36': ocn.OFFSETS_PATTERN36,
                        'Pattern 37': ocn.OFFSETS_PATTERN37, 'Pattern 38': ocn.OFFSETS_PATTERN38, 'Pattern 39': ocn.OFFSETS_PATTERN39, 'Pattern 40': ocn.OFFSETS_PATTERN40,
                        'Pattern 41': ocn.OFFSETS_PATTERN41, 'Pattern 42': ocn.OFFSETS_PATTERN42, 'Pattern 43': ocn.OFFSETS_PATTERN43, 'Pattern 44': ocn.OFFSETS_PATTERN44,
                        'Pattern 45': ocn.OFFSETS_PATTERN45, 'Pattern 46': ocn.OFFSETS_PATTERN46, 'Pattern 47': ocn.OFFSETS_PATTERN47, 'Pattern 48': ocn.OFFSETS_PATTERN48,
                        'Pattern 49': ocn.OFFSETS_PATTERN49, 'Pattern 50': ocn.OFFSETS_PATTERN50}

            selected_pattern =  self.form_gui.window['combo_selectstandingwavepattern'].get()
            self.i3_offsets = self.getOffsetsForPattern(patterns[selected_pattern], frequency)
          else:
            self.i3_offsets = self.getOffsetsForPattern(self.i3_offsets_type, frequency)


      self.debug.info_message("calcCarrierFrequencies. frequencies: " + str(frequency))

      return frequency

    except:
      self.debug.error_message("Exception in calcCarrierFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  """
LB28-6400-64-2-15-I3,-3.3688081276276565,-24.768404549646576,0.1419753086419753,1.25,7.5,8.0,C-C,0.576
LB28-6400-64-2-15-I3,-3.385194480383654,-24.7701383461221,0.1419753086419753,1.25,7.5,8.0,C-C,0.576

LB28-6400-64-2-15-I3,-3.390658322207706,-24.77071500793826,0.13580246913580246,1.25,7.5,8.0,C-C,0.441
LB28-6400-64-2-15-I3,-3.402601245196894,-24.771972958226737,0.10185185185185185,1.25,7.5,8.0,C-C,0.441

LB28-6400-64-2-15-I3,-3.399110346168559,-24.771605618914805,0.16049382716049382,1.25,7.5,8.0,B-C,0.533
LB28-6400-64-2-15-I3,-3.3856661544784235,-24.77018815590878,0.13271604938271606,1.25,7.5,8.0,B-C,0.533

LB28-6400-64-2-15-I3,-2.841106467648608,-24.70892392064246,0.10185185185185185,1.25,7.5,7.5,C-E,0.506
LB28-6400-64-2-15-I3,-2.8501448810689416,-24.71000472080637,0.11419753086419752,1.25,7.5,7.5,C-E,0.506
LB28-6400-64-2-15-I3,-2.8524277828350826,-24.710277351208777,0.14506172839506173,1.25,7.5,7.5,C-E,0.506

LB28-6400-64-2-15-I3,-0.9559430363783641,-24.426360151639578,0.09876543209876543,1.25,7.5,6.0,C-C,0.661
LB28-6400-64-2-15-I3,-0.9875609717966416,-24.43218084814434,0.027777777777777776,1.25,7.5,6.0,C-C,0.661
LB28-6400-64-2-15-I3,-0.9672116530154518,-24.42843950682772,0.09567901234567901,1.25,7.5,6.0,C-C,0.661

LB28-6400-64-2-15-I3,-0.9431309765336505,-24.42398942705106,0.07716049382716049,1.25,7.5,6.0,C-C,0.618
LB28-6400-64-2-15-I3,-0.9559387508702288,-24.426359359824218,0.05555555555555555,1.25,7.5,6.0,C-C,0.618
LB28-6400-64-2-15-I3,-0.9745958119028404,-24.429799154553944,0.024691358024691357,1.25,7.5,6.0,C-C,0.618

LB28-6400-64-2-15-I3,-0.9686352381030344,-24.428701811952745,0.033950617283950615,1.25,7.5,6.0,C-C,0.133
LB28-6400-64-2-15-I3,-0.9840877470981899,-24.431543515527476,0.05864197530864197,1.25,7.5,6.0,C-C,0.133

LB28-6400-64-2-15-I3,-0.9820511669305283,-24.431169568351986,0.043209876543209874,1.25,7.5,6.0,C-C,0.471
LB28-6400-64-2-15-I3,-0.9700256657816014,-24.428957924614554,0.06481481481481481,1.25,7.5,6.0,C-C,0.471

LB28-6400-64-2-15-I3,-0.9750060566423036,-24.429874625295326,0.040123456790123455,1.25,7.5,6.0,C-C,0.537
LB28-6400-64-2-15-I3,-0.9809274658394942,-24.430963164612727,0.06172839506172839,1.25,7.5,6.0,C-C,0.537

LB28-6400-64-2-15-I3,-2.8510139163094204,-24.710108520276737,0.1697530864197531,1.25,7.5,7.5,A-E,0.266
LB28-6400-64-2-15-I3,-2.8336903806745872,-24.708035434387636,0.12962962962962962,1.25,7.5,7.5,A-E,0.266



LB28-6400-64-2-15-I3,-2.849538523164944,-24.709932283788635,0.15123456790123457,1.25,7.5,7.5,D-D,0.522
LB28-6400-64-2-15-I3,-2.837694642740801,-24.70851535449473,0.1697530864197531,1.25,7.5,7.5,D-D,0.522

LB28-6400-64-2-15-I3,-2.8320473695617543,-24.707838387624726,0.19753086419753085,1.25,7.5,7.5,C-C,0.261
LB28-6400-64-2-15-I3,-2.8226896816632774,-24.706714695139553,0.1697530864197531,1.25,7.5,7.5,C-C,0.261

LB28-6400-64-2-15-I3,-0.9668543037576628,-24.428373649322435,0.04938271604938271,1.25,7.5,6.0,C-C,0.668
LB28-6400-64-2-15-I3,-0.976676958524719,-24.430181939396853,0.07716049382716049,1.25,7.5,6.0,C-C,0.668

LB28-6400-64-2-15-I3,-0.9856955371214592,-24.431838606454278,0.07716049382716049,1.25,7.5,6.0,C-C,0.992
LB28-6400-64-2-15-I3,-0.9443071336580797,-24.424207353091507,0.06790123456790123,1.25,7.5,6.0,C-C,0.992

LB28-6400-64-2-15-I3,-0.9810918345989177,-24.430993359540913,0.05864197530864197,1.25,7.5,6.0,C-C,0.969
LB28-6400-64-2-15-I3,-0.977450929550439,-24.430324248931225,0.07407407407407407,1.25,7.5,6.0,C-C,0.969

LB28-6400-64-2-15-I3,-0.9805581960423583,-24.43089532471359,0.06790123456790123,1.25,7.5,6.0,B-C,0.024
LB28-6400-64-2-15-I3,-0.9624270747393336,-24.42755728573219,0.07716049382716049,1.25,7.5,6.0,B-C,0.024





  """

  """
  best patterns...
     total best...37, 19,,15,18     50, 12, 25, then 44 then 17
     self.test_sw_patterns = [('A-D', 0.594), ('A-D', 0.657), ('A-D', 0.312), ('A-D', 0.562), ('A-D', 0.605), ('A-D', 0.827), ('A-D', 0.822), ('A-D', 0.373), ('A-D', 0.827)]


     3,4,5,6,8,9,11,14,15,17,18,27,29,35
     self.test_sw_patterns = [('C-C', 0.133),('B-C', 0.338),('C-C', 0.233),('C-C', 0.144),('C-E', 0.506),('C-C', 0.429),('A-D', 0.821),('A-D', 0.196),('A-D', 0.312),('A-D', 0.827),('A-D', 0.562),('A-D', 0.026),('A-D', 0.612),('A-D', 0.616)]

     17,19,30,36,40,41,44,50


 can reuse 20-26  45-49,
  31-34 37-39 42-43

  """
  def getOffsetsForPattern(self, pattern, frequency):
    self.debug.info_message("getOffsetsForPattern")
    try:

      """ 5 patterns with lowest single test result followed by 5 patterns with lowest 3-in-a-row test result"""
      if pattern == ocn.OFFSETS_PATTERN1: # 1
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.358)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN2:   # 2
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.666)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN3:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.133)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN4:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.338)
        return phase_list_low[1], phase_list_low[2], phase_list_high[1], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN5:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.233)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN6:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.144)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN7:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.839)
        return phase_list_low[4], phase_list_low[4], phase_list_high[4], phase_list_high[4]
      elif pattern == ocn.OFFSETS_PATTERN8:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.506)
        return phase_list_low[2], phase_list_low[4], phase_list_high[2], phase_list_high[4]
      elif pattern == ocn.OFFSETS_PATTERN9:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.429)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN10:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.839)
        return phase_list_low[1], phase_list_low[1], phase_list_high[1], phase_list_high[1]

      elif pattern == ocn.OFFSETS_PATTERN11:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.821)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN12:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.827)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN13:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.193)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN14:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.196)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN15:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.312)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN16:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.749)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN17:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.827)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN18:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.562)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN19:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.657)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN20:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.248)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN21:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.883)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN22:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.589)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN23:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.171)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN24:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.943)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN25:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.822)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN26:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.425)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN27:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.026)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]



      elif pattern == ocn.OFFSETS_PATTERN28:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.467)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN29:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.612)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN30:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.616)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN31:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.719)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN32:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.889)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN33:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.256)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN34:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.758)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN35:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.616)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN36:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.527)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN37:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.594)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN38:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.144)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN39:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.551)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN40:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.303)
        return phase_list_low[3], phase_list_low[3], phase_list_high[4], phase_list_high[4]

      elif pattern == ocn.OFFSETS_PATTERN41:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.653)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN42:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.195)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN43:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.865)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN44:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.373)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN45:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.366)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN46:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.313)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN47:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.482)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN48:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.364)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN49:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.698)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN50:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.605)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


    except:
      self.debug.error_message("Exception in getOffsetsForPattern: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calcPhaseAngles(self, frequency, standingwaveoffset):
    self.debug.info_message("calcPhaseAngles")

    try:
      pulse_length_in_samples = self.symbol_block_size / self.pulses_per_block

      def calcPhases(freq, offset_ratio):
        """ calculate the phases for the pulses at fixed pulse distance from pulse C """
        """ pulse separation is equivalent to pulse_length """
        offset_samples = pulse_length_in_samples * offset_ratio
        wavelength_in_samples   = self.sample_rate / freq
        phase_for_pulse_A = (((2*pulse_length_in_samples) - wavelength_in_samples + offset_samples) %  wavelength_in_samples ) / wavelength_in_samples
        phase_for_pulse_B = ((pulse_length_in_samples - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        phase_for_pulse_C = ((0 - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        phase_for_pulse_D = ((-pulse_length_in_samples - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        phase_for_pulse_E = ((-2*pulse_length_in_samples - wavelength_in_samples + offset_samples) %  wavelength_in_samples) / wavelength_in_samples
        self.debug.info_message("phase_for_pulse_A: " + str(phase_for_pulse_A))
        self.debug.info_message("phase_for_pulse_B: " + str(phase_for_pulse_B))
        self.debug.info_message("phase_for_pulse_C: " + str(phase_for_pulse_C))
        self.debug.info_message("phase_for_pulse_D (A): " + str(phase_for_pulse_D))
        self.debug.info_message("phase_for_pulse_E (B): " + str(phase_for_pulse_E))
        return [phase_for_pulse_A, phase_for_pulse_B, phase_for_pulse_C, phase_for_pulse_D, phase_for_pulse_E]

      def calcForEachFreq(offset_ratio):
        """ calculate first frequency """
        #self.debug.info_message("frequency[0]: " + str(frequency[0]))
        phase_list_low  = calcPhases(frequency[0], offset_ratio)
        """ calculate second frequency """
        #self.debug.info_message("frequency[1]: " + str(frequency[1]))
        phase_list_high = calcPhases(frequency[1], offset_ratio)

        return phase_list_low, phase_list_high


      return calcForEachFreq(standingwaveoffset)

    except:
      self.debug.error_message("Exception in calcPhaseAngles: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



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


  def startDecoder(self, mode, window, values):
    self.debug.info_message("startDecoder")

    self.runDecoder = True

    try:
      self.setInitializationBlock(mode)
      self.initInputStream(self.sample_rate, window, values)

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
    #self.debug.info_message("pushInputBuffer")
    try:
      self.inputBufferItemCount += 1
      #self.debug.info_message("pushInputBuffer putting item")
      self.inputBuffer.put(data)
      #self.debug.info_message("pushInputBuffer completed putting item")
      #self.debug.info_message("pushInputBuffer count: " + str(self.inputBufferItemCount))
    except:
      self.debug.error_message("Exception in pushInputBuffer: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

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

  def initInputStream(self, sample_rate, window, values):
    self.debug.info_message("initInputStream" )

    if self.symbol_block_size != self.previousBlocksizeIn or self.inStreamRunning == False:
      self.previousBlocksizeIn = self.symbol_block_size

      if self.inStreamRunning == True:
        self.inStream.stop()
        #self.inStream.stop_stream()

      self.resetInputBuffer()

      #p = pyaudio.PyAudio()
      #for i in range(p.get_device_count()):
      #  device_info = p.get_device_info_by_index(i)
      #  self.debug.info_message("device_info: index(" + str(i) + ") name = " + str(device_info['name']) )
      #instream = p.open(format=pyaudio.paFloat32, channels=1, rate = 48000, input = True, input_device_index=2, stream_callback=self.pa_instream_callback, frames_per_buffer=self.symbol_block_size)
      #instream.start_stream()

      devices = sd.query_devices()
      for device in devices:
        self.debug.info_message("device ID: " + str(device['index']) + " device Name: " + str(device['name'])  )

      #self.inStream = sd.InputStream(samplerate=48000, channels = 1, blocksize=self.symbol_block_size,
      #self.inStream = sd.InputStream(samplerate=16000, channels = 1, blocksize=self.symbol_block_size,
      #self.inStream = sd.InputStream(samplerate=8000, channels = 1, blocksize=self.symbol_block_size,

      self.inStream = sd.InputStream(samplerate=int(self.sample_rate), channels = 1, blocksize=self.symbol_block_size,
                                     dtype=np.float32, callback = self.sd_instream_callback)
      #device_id = 2
      #self.inStream = sd.InputStream(samplerate=48000, channels = 2, device = device_id, blocksize=self.symbol_block_size,
      #                               dtype=np.float32, callback = self.sd_instream_callback)

      self.inStream.start()

      self.inStreamRunning = True

    """ start the decoder thread """
    t1 = threading.Thread(target=self.decoder_thread, args=(window, values, ))
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


  def pa_instream_callback(self, in_data, frame_count, time_info, flag):
    self.debug.info_message("pa_instream_callback")
    self.debug.info_message("frame_count: " + str(frame_count))
    self.debug.info_message("len(in_data): " + str(len(in_data)))
    self.debug.info_message("flag: " + str(flag))
    try:
      if self.runDecoder == True:
        #if in_data.shape[1] == 2:
        #  single_channel_data[:, 0] = in_data[:, 0]
        #  audio_data = np.frombuffer(single_channel_data, dtype=np.float32)
        #  self.pushInputBuffer(audio_data)
        #else:
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.pushInputBuffer(audio_data)
      return (in_data, pyaudio.paContinue)
    except:
      self.debug.error_message("Exception in pa_instream_callback: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  """ This method is the focal point of the detector. Looks for signals then starts putting blocks in the decoder queue"""
  def sd_instream_callback(self, indata, frames, time, status):
    self.debug.info_message("sd_instream_callback. num frames: " + str(frames))
    self.debug.info_message("time_info capture time of first sample: " + str(time.inputBufferAdcTime))

    """ send data to demodulator """
    #"""
    if self.runDecoder == True:
      self.pushInputBuffer(np.array(indata))
      #self.pushInputBuffer(np.array(indata).astype(np.float64))
      #self.pushInputBuffer(indata.astype(np.float64))
    #"""

    """ send data to spectrum display """
    #self.form_gui.spectralDensityQueue.put(indata)
    """
    if self.inStreamRunning == True:
      if self.spectral_density_queue_counter == 0:
        self.spectral_density_block = indata
        self.spectral_density_queue_counter += 1
      elif self.spectral_density_queue_counter > 0:
        #self.debug.info_message("pushing data for spectral density plot")
        self.spectral_density_block = np.append(self.spectral_density_block, indata)
        self.form_gui.spectralDensityQueue.put(self.spectral_density_block)
        self.spectral_density_queue_counter = 0
      else:
        self.spectral_density_block = np.append(self.spectral_density_block, indata)
        self.spectral_density_queue_counter += 1
    """
    return None

  def decoder_thread(self, window, values):
    self.debug.info_message("decoder_thread")

    while self.runDecoder == True:
      num_items = self.getInputBufferItemCount()
      self.debug.info_message("decoder_thread num items: "+ str(num_items))
      if  num_items >= 30:
        self.debug.info_message("we have 30 items in queue...starting decode")
        for i in range(0, 30): 
          block = self.popInputBuffer()
          if i == 0:
            self.debug.info_message("i: " + str(i))
            self.debug.info_message("self.symbol_block_size: " + str(self.symbol_block_size))
            multi_block = np.zeros((30 * self.symbol_block_size,), dtype = np.float64)
            multi_block[0:self.symbol_block_size] = block.reshape(self.symbol_block_size,).astype(np.float64)
          else:
            self.debug.info_message("i: " + str(i))
            multi_block[i*self.symbol_block_size:(i+1) * self.symbol_block_size] = block.reshape(self.symbol_block_size,).astype(np.float64)


        multi_block = multi_block * 1
        #self.modulation_object.writeFileWav2("TEST_AUDIO.wav", multi_block)
        self.modulation_object.writeFileWav("TEST_AUDIO.wav", multi_block)

        save_sampled_signal_checked = self.form_gui.window['cb_savesampledsignal'].get()
        if save_sampled_signal_checked:
          sampled_signal_name = self.form_gui.window['in_sampledsignalname'].get()
          self.modulation_object.writeFileWav(sampled_signal_name, multi_block)


        # test increase amplitude...
        #multi_block = 0.0001 * multi_block * (2**15 - 1) / np.max(np.abs(multi_block)) 
        multi_block = 0.001 * multi_block * (2**15 - 1) / np.max(np.abs(multi_block)) 

        center_frequency = values['slider_frequency']
        separation_override = values['slider_carrier_separation']

        #self.debug.info_message("strongest frequency is: " + str(self.modulation_object.getStrongestFrequency(multi_block, 1160, 1450)))
        self.debug.info_message("strongest frequency is: " + str(self.modulation_object.getStrongestFrequency(multi_block, 1350, 1450)))
        self.debug.info_message("strongest frequencies over range is: " + str(self.modulation_object.getStrongestFrequencies(multi_block, 20, 1160, 1450)))

        fft_frequency = self.modulation_object.getStrongestFrequency(multi_block, 1160, 1450)
        #fft_frequency = 1280 #1300.166666
        fft_frequency = 1382 #1300.166666
        self.form_gui.window['text_info_fftfreq'].update(fft_frequency)

        #frequency = self.calcCarrierFrequencies(center_frequency, separation_override)
        frequency = self.calcCarrierFrequenciesFromFFT(fft_frequency, separation_override)


        #self.osmod.modulation_object.writeFileWav(mode + ".wav", data2)


        self.decoder_callback(multi_block, frequency)

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

    

