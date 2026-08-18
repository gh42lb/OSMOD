#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
#import sounddevice as sd
import numpy as np
#import matplotlib.pyplot as plt
import threading
import sys
import gc
import pyaudio
import ctypes
import pkgutil
import os
import platform
import osmod_net_gui
import getopt
import osmod_net_events
import json

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
from scipy import signal as scipy_signal

from scipy.signal import hilbert

from osmod_dictionary import PersistentData
from osmod_analysis import OsmodAnalysis
from osmod_detector import OsmodDetector
from osmod_simulations import OsmodSimulator
from osmod_interpolation import OsmodInterpolator
from osmod_test import OsmodTest
from osmod_fec import OsmodFEC
from osmod_prod_params import OsmodProdParams
from osmod_sonic import OsmodSonic

#from osmod_net_gui import MainNetWindow
from osmod_net_client import OSMOD_Net


class osModem(object):

  mod_psk   = None
  demod_psk = None

  mod_2fsk8psk   = None
  demod_2fsk8psk = None
  mod_2fsk4psk   = None
  demod_2fsk4psk = None

  mod_fsk   = None
  demod_fsk = None

  decoderRunning = False
  encoderRunning = False

  fec_params = (400, 361, 2, 20, 500)

  previousBlocksizeIn  = 0
  previousBlocksizeOut = 0
  inStreamRunning      = False
  outStreamRunning     = False
  signal_squelch_value = 0.0  # 0.25
  bias_filter_value    = 1.3
  input_gain = 1.0
  output_gain = 0.1

  slider_awgn = 0.0
  slider_amplitude = 1.0
  slider_carrier_separation = 37

  all_messages = {}
  messages_by_frequency = {}
  messages_by_callsign = {}

  solve_data_dict = {}

  test_mode_enabled = True

  received_data_table = None
  received_data_table_callsign = None

  spectral_density_queue_counter = 0
  spectral_density_block = None

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)

  """ Transmit times:
                           BEACON                                           TIMED GENERAL MESSAGE
      LB28-1600-I3-FEC   30 characters     15 seconds     15           60 characters          30 seconds

      LB28-1600-I3       68 characters     15 seconds     20          136 characters          40 seconds 

      LB28-3200-I3-FEC   62 characters                    25          124 characters          50 seconds

      LB28-3200-I3       58 characters     25 seconds     30          116 characters          50 seconds

      LB28-6400-I3-FEC   52 characters                    35          104 characters          70 seconds

      LB28-6400-I3       48 characters     40 seconds     40           96 characters           80 seconds

      LB28-12800-I3-FEC  42 characters                    45           84 characters           90 seconds

      LB28-12800-I3      38 characters     1 minute                    76 characters           2 minutes

      LB28-25600-I3      28 characters     1 minute 30 seconds        56 characters           3 minutes


message types...

Beacon General       - BG(<Locator>)
Beacon Net           - BN(<Locator>, <Time UTC>, <Freq>, <Mode>)
Beacon Text Msg      - BT(<Locator>, <Text Msg>)
Beacon CQCQ          - BC(<Locator>, <Freq>, <Mode>)
Beacon Alert (RED)   - BR(<Locator>, <Message>)
Beacon Alert (GREEN) - BA(<Locator>, <Message>)

message formats...

28 characters...
<rotation sequence. 8 chars><message 10 chars><end of message 1 char><callsign 6 chrs><space 1 char><checksum 2 chars>

28 characters...

  """

  def __init__(self, form_gui):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")

    for module in pkgutil.iter_modules():
      self.debug.info_message("module: " + str(module[1]))


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
    self.center_frequency = 1400
    self.symbols = 32
    self.bandwidth = 1000
    self.bits_per_symbol = int(np.log2(self.symbols))

    self.test_counter = 0

    self.aperture = 1.5

    self.mode = None

    """ frequency separation between tones, in Hz (baud rate) """
    self.freq_sep = self.bandwidth/self.symbols
    self.time_sep = int(np.ceil(self.sample_rate/self.freq_sep))

    self.core_utils = ModemCoreUtils(self)

    self.mod_2fsk8psk   = mod_2FSK8PSK(self)
    self.demod_2fsk8psk = demod_2FSK8PSK(self)

    #self.mod_2fsk4psk   = mod_2FSK4PSK(self)
    #self.demod_2fsk4psk = demod_2FSK4PSK(self)

    self.prodparams   = OsmodProdParams(self)

    self.sonic = OsmodSonic(self)

    #self.net = OSMOD_Net()
    self.osmod_net_dispatcher = None
    self.osmod_net = None
    self.osmod_net_main()

    """ start the decoder thread """
    #self.t1_decoder = threading.Thread(target=self.decodeProcessing, args=(window, values, ))
    #self.t1_decoder.start()
    self.t1_decoder = None
    self.exit_decoder_processing = False


    self.dataQueue = Queue()
    self.inputBuffer = Queue()

    self.two_times_pi = 2 * np.pi

    self.timer_dict_when = {}
    self.timer_dict_elapsed = {}
    self.timer_last_name = ''

    self.dict_rcvd = {}
    self.dict_rcvd_callsign = {}

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
    self.modulation_specific_pulse_shapes['LB28-2560-8-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-320-8-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-400-8-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-320-8-2-37-I3E8-FEC-FDM'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-400-8-2-37-I3E8-FEC-FDM'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]

    self.modulation_specific_pulse_shapes['LB28-25600-256-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-51200-512-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]
    self.modulation_specific_pulse_shapes['LB28-102400-1024-2-37-I3E8-FEC'] = [(0.735 , 0.221), (0.804 , 0.25), (0.595 , 0.255), (0.638 , 0.318), (0.104 , 0.187), (0.604 , 0.295)]

    #self.test_pulse_shapes = [(0.215 , 0.091), (0.763, 0.107), (0.992 , 0.276),(0.849 , 0.898),(0.586 , 0.78),(0.612 , 0.353),(0.63 , 0.206),(0.638 , 0.318)]

    #self.test_pulse_shapes = [(0.763, 0.107), (0.992 , 0.276),(0.849 , 0.898),(0.586 , 0.78),(0.612 , 0.353),(0.63 , 0.206),(0.638 , 0.318), (0.515 , 0.035),(0.662 , 0.228),(0.937 , 0.172),(0.523 , 0.068),(0.585 , 0.056),(0.308 , 0.024),(0.966 , 0.157),(0.215 , 0.091),(0.708 , 0.104),(0.92 , 0.025),(0.735 , 0.221),(0.161 , 0.209),(0.128 , 0.188),(0.096 , 0.157),(0.881 , 0.266),(0.783 , 0.282),(0.941 , 0.254),(0.778 , 0.177),(0.403 , 0.294),(0.68 , 0.096),(0.579 , 0.248)]

    #self.test_pulse_shapes = [(0.595 , 0.255),(0.727 , 0.225),(0.491 , 0.29),(0.945 , 0.346),(0.954 , 0.289),(0.979 , 0.34),(0.657 , 0.173),(0.183 , 0.148),(0.552 , 0.288),(0.569 , 0.191),(0.692 , 0.241),(0.138 , 0.176),(0.285 , 0.268),(0.904 , 0.305),(0.412 , 0.224),(0.785 , 0.323),(0.658 , 0.241),(0.737 , 0.238),(0.336 , 0.223),(0.475 , 0.242),(0.604 , 0.295),(0.824 , 0.34),(0.519 , 0.177),(0.707 , 0.244),(0.557 , 0.252),(0.804 , 0.25),(0.601 , 0.175),(0.638 , 0.32),(0.833 , 0.309),(0.104 , 0.187),(0.887 , 0.322),(0.722 , 0.288),(0.352 , 0.165),(0.367 , 0.25)]

    #self.test_pulse_shapes = [(0.323 , 0.054),(0.115 , 0.028),(0.07 , 0.025),(0.402 , 0.048),(0.3 , 0.085),(0.708 , 0.076),(0.745 , 0.093),(0.051 , 0.009)]

    #self.test_pulse_shapes = [(0.065 , 0.458),(0.261 , 0.74),(0.227 , 0.965),(0.892 , 0.977),(0.518 , 0.826),(0.876 , 0.425),(0.163 , 0.918),(0.623 , 0.919),(0.22 , 0.453),(0.388 , 0.551),(0.159 , 0.732),(0.595 , 0.777),(0.06 , 0.924),(0.876 , 0.956),(0.975 , 0.853),(0.516 , 0.863),(0.336 , 0.908),(0.787 , 0.628),(0.902 , 0.389),(0.319 , 0.946),(0.468 , 0.512),(0.985 , 0.781),(0.615 , 0.531),(0.077 , 0.731),(0.816 , 0.978),(0.564 , 0.406),(0.354 , 0.863),(0.371 , 0.678),(0.305 , 0.775),(0.021 , 0.954),(0.512 , 0.466),(0.914 , 0.479)]

    self.test_pulse_shapes = [(0.26 , 0.627),(0.868 , 0.567),(0.024 , 0.214),(0.611 , 0.82),(0.799 , 0.395),(0.1 , 0.575),(0.62 , 0.911),(0.796 , 0.561),(0.455 , 0.505),(0.936 , 0.909),(0.112 , 0.627),(0.497 , 0.603),(0.849 , 0.805),(0.64 , 0.68),(0.465 , 0.842),(0.352 , 0.458),(0.493 , 0.966),(0.311 , 0.461),(0.982 , 0.373),(0.202 , 0.825),(0.116 , 0.331),(0.369 , 0.639),(0.742 , 0.271),(0.014 , 0.525),(0.881 , 0.736),(0.44 , 0.155),(0.112 , 0.785),(0.457 , 0.338),(0.381 , 0.758),(0.371 , 0.865),(0.629 , 0.986),(0.897 , 0.77),(0.041 , 0.997),(0.273 , 0.24),(0.48 , 0.683),(0.032 , 0.428),(0.094 , 0.095),(0.054 , 0.827),(0.549 , 0.271),(0.349 , 0.887),(0.646 , 0.35),(0.548 , 0.797),(0.014 , 0.458),(0.035 , 0.831),(0.463 , 0.933),(0.059 , 0.723),(0.537 , 0.398),(0.2 , 0.735),(0.007 , 0.237),(0.347 , 0.296),(0.128 , 0.998)]


    self.best_pulse_shapes = [(0.104 , 0.187), (0.336 , 0.223), (0.804 , 0.25), (0.722 , 0.288), (0.595 , 0.255), (0.945 , 0.346), (0.737 , 0.238), (0.638 , 0.318), (0.662 , 0.228), (0.161 , 0.209), (0.612 , 0.353), (0.783 , 0.282), (0.735 , 0.221), (0.215 , 0.091)]

    self.pulse_shapes_1 =[(0.661 , 0.063),(0.646 , 0.039),(0.392 , 0.093),(0.123 , 0.058),(0.344 , 0.098)]


    #self.all_pulse_shapes = [self.test_pulse_shapes, self.best_pulse_shapes]

    self.all_pulse_shapes = [self.pulse_shapes_1]

    #[,(0.519 , 0.177),(0.569 , 0.191),(0.945 , 0.346),(0.785 , 0.323),(0.887 , 0.322),(0.824 , 0.34),(0.104 , 0.187),(0.183 , 0.148),(0.336 , 0.223),(0.475 , 0.242),(0.737 , 0.238),(0.804 , 0.25),(0.722 , 0.288),(0.904 , 0.305),(0.601 , 0.175),(0.595 , 0.255),(0.833 , 0.309),(0.557 , 0.252),(0.707 , 0.244),(0.979 , 0.34),(0.954 , 0.289),(0.138 , 0.176),(0.285 , 0.268),(0.604 , 0.295),(0.367 , 0.25),(0.657 , 0.173),(0.692 , 0.241),(0.491 , 0.29),(0.552 , 0.288)]

    self.test_sw_patterns_1  = [('A-D', 0.594), ('A-D', 0.657), ('A-D', 0.312), ('A-D', 0.562), ('A-D', 0.605), ('A-D', 0.827), ('A-D', 0.822), ('A-D', 0.373), ('A-D', 0.827)]
    self.test_sw_patterns_2  = [('C-C', 0.133),('B-C', 0.338),('C-C', 0.233),('C-C', 0.144),('C-E', 0.506),('C-C', 0.429),('A-D', 0.821),('A-D', 0.196),('A-D', 0.312),('A-D', 0.827),('A-D', 0.562),('A-D', 0.026),('A-D', 0.612),('A-D', 0.616)]
    self.test_sw_patterns_3  = [('A-D', 0.768),('A-D', 0.656),('A-D', 0.456),('A-D', 0.913),('A-D', 0.385),('A-D', 0.113),('A-D', 0.577),('A-D', 0.426),('A-D', 0.493),('A-D', 0.675),('A-D', 0.747),('A-D', 0.882),('A-D', 0.636),('B-C', 0.581),('A-C', 0.843),('A-D', 0.417),('B-C', 0.314),('A-D', 0.933),('A-D', 0.565),('A-D', 0.678),('A-D', 0.501),('A-D', 0.621),('A-D', 0.492),('A-D', 0.659),('A-D', 0.825),('A-D', 0.939)]
    self.test_sw_patterns_4  = [('B-B', 0.791),('C-C', 0.462),('B-D', 0.875),('B-C', 0.435),('A-E', 0.327),('D-E', 0.742),('E-E', 0.212),('B-C', 0.324),('A-C', 0.685),('C-C', 0.1),('A-D', 0.201),('A-D', 0.61),('C-C', 0.148),('B-E', 0.752),('B-D', 0.38),('D-D', 0.811),('A-E', 0.049),('A-C', 0.43),('B-E', 0.921),('B-B', 0.88),('C-E', 0.777),('B-E', 0.561),('A-A', 0.288),('B-C', 0.266),('C-E', 0.788),('C-E', 0.251),('D-D', 0.977),('A-D', 0.034),('A-E', 0.653),('A-D', 0.541),('A-C', 0.797),('C-C', 0.696),('C-C', 0.323),('A-C', 0.053),('D-D', 0.884),('C-C', 0.195),('A-B', 0.287),('C-D', 0.748),('A-D', 0.656),('B-C', 0.041),('A-B', 0.286)]
    self.test_sw_patterns_5  = [('A-B', 0.32),('A-A', 0.114),('A-A', 0.577),('C-E', 0.397),('A-E', 0.196),('B-B', 0.72),('A-C', 0.061),('D-D', 0.028),('D-E', 0.935),('C-E', 0.126),('A-D', 0.369),('C-E', 0.398),('E-E', 0.516),('A-C', 0.701),('E-E', 0.99),('D-D', 0.702),('B-B', 0.012)]


    self.test_sw_patterns_6  = [('E-E', 0.139),('B-D', 0.977),('B-B', 0.974),('A-D', 0.474),('A-A', 0.822),('A-E', 0.15),('C-E', 0.584),('B-C', 0.113),('A-D', 0.986),('C-E', 0.806),('D-E', 0.531),('A-D', 0.972),('A-D', 0.62),('B-D', 0.068),('C-C', 0.152),('C-E', 0.579),('D-E', 0.763),('D-E', 0.418),('A-D', 0.473),('B-B', 0.748),('B-D', 0.109),('E-E', 0.078),('A-D', 0.337),('B-D', 0.24),('B-E', 0.713),('A-E', 0.012),('A-B', 0.655),('A-D', 0.031),('B-C', 0.801),('A-B', 0.367),('B-D', 0.645),('B-D', 0.243),('A-C', 0.204),('C-E', 0.234),('A-C', 0.627),('A-C', 0.57),('B-D', 0.172),('A-C', 0.838)]

    self.test_sw_patterns_7  = [('A-E', 0.022),('B-B', 0.691),('A-B', 0.998),('B-D', 0.452)]

    self.test_sw_patterns_8  = [('A-C', 0.507),('C-C', 0.331),('E-E', 0.197),('A-A', 0.483),('A-E', 0.039),('A-C', 0.491),('B-D', 0.631),('B-B', 0.031),('B-E', 0.342),('D-D', 0.198),('A-C', 0.871),('C-E', 0.505),('E-E', 0.261),('D-D', 0.913),('E-E', 0.822),('A-E', 0.686),('B-C', 0.399),('B-D', 0.722),('A-C', 0.895),('C-D', 0.999),('A-D', 0.496),('A-A', 0.138),('A-A', 0.367),('D-D', 0.264),('B-B', 0.841),('A-C', 0.092),('A-C', 0.49),('B-B', 0.851),('B-C', 0.465),('D-E', 0.02),('B-E', 0.681),('C-D', 0.119),('B-E', 0.633),('E-E', 0.774),('A-C', 0.928),('C-E', 0.92),('B-B', 0.489),('B-C', 0.525),('C-D', 0.008),('C-D', 0.405),('A-E', 0.033),('B-B', 0.225),('C-C', 0.074),('B-C', 0.565),('D-E', 0.53),('A-B', 0.597),('C-D', 0.992),('A-E', 0.428),('C-E', 0.229),('C-E', 0.538),('C-C', 0.093),('D-E', 0.614),('C-D', 0.513),('C-E', 0.971),('D-D', 0.745),('A-B', 0.602),('A-A', 0.855),('C-C', 0.266),('C-C', 0.157),('A-A', 0.071),('E-E', 0.489),('D-E', 0.535),('A-C', 0.34),('E-E', 0.213),('D-D', 0.492),('A-D', 0.452),('E-E', 0.246),('B-D', 0.52),('A-A', 0.075),('A-E', 0.593),('B-E', 0.024),('A-A', 0.372),('B-C', 0.531),('A-D', 0.949),('B-B', 0.522),('B-D', 0.682),('B-C', 0.441),('C-D', 0.934),('E-E', 0.248),('A-E', 0.856),('C-E', 0.947)]
    self.test_sw_patterns_9  = [('B-C', 0.029),('B-E', 0.854),('B-E', 0.96), ('C-E', 0.954),('A-C', 0.775),('C-E', 0.635),('D-E', 0.032),('A-A', 0.601),('A-B', 0.798),('C-E', 0.828),('B-D', 0.969),('A-E', 0.914)],

    self.test_sw_patterns_10  = [('A-E', 0.856),('A-C', 0.895),('A-C', 0.871),('A-B', 0.602),('B-B', 0.225),('B-D', 0.631),('E-E', 0.489)]

    self.test_sw_patterns_11  = [('B-C', 0.48),('A-C', 0.212),('C-C', 0.194),('B-D', 0.176),('B-B', 0.187),('A-C', 0.144),('B-B', 0.938),('C-C', 0.287),('A-B', 0.922),('C-C', 0.921),('A-D', 0.48),('A-E', 0.903),('C-C', 0.865),('D-E', 0.912),('D-D', 0.695),('A-B', 0.649),('B-D', 0.525)]

    self.test_sw_patterns_12  =  [('A-C', 0.546),('C-E', 0.43),('A-D', 0.397),('C-C', 0.091),('A-A', 0.828),('B-C', 0.782),('A-A', 0.701),('A-B', 0.135),('B-B', 0.806),('A-B', 0.255),('A-A', 0.175),('D-E', 0.2),('B-B', 0.636),('A-B', 0.971),('B-D', 0.395),('A-E', 0.745),('A-C', 0.659),('B-D', 0.805),('B-C', 0.287),('C-E', 0.763),('B-B', 0.852),('A-A', 0.116),('A-A', 0.413),('B-B', 0.401),('B-D', 0.405)]
    self.test_sw_patterns_13  =  [('C-E', 0.43),('D-E', 0.719),('D-E', 0.781),('D-D', 0.601),('A-A', 0.701),('A-D', 0.647),('D-E', 0.56),('A-B', 0.255),('A-A', 0.175),('D-E', 0.601),('D-E', 0.2),('A-C', 0.146),('A-C', 0.659),('A-B', 0.429),('C-D', 0.29),('A-A', 0.321),('B-D', 0.405)]

    self.test_sw_patterns_14  =  [('B-B', 0.31),('A-D', 0.83),('C-E', 0.37),('D-D', 0.274),('A-B', 0.425),('A-D', 0.492),('A-E', 0.173),('C-E', 0.34),('D-D', 0.011),('C-E', 0.261),('C-E', 0.254),('A-C', 0.506),('B-E', 0.313),('B-B', 0.215),('A-D', 0.369),('C-D', 0.208),('D-D', 0.517),('C-E', 0.747),('B-E', 0.571),('C-E', 0.413)]
    self.test_sw_patterns_15  =  [('B-B', 0.973),('A-B', 0.253),('A-E', 0.173),('B-B', 0.328),('C-E', 0.254),('B-E', 0.313),('A-D', 0.369),('D-D', 0.517),('A-B', 0.795)]

    self.test_sw_patterns_16  =  [('B-D', 0.047),('A-E', 0.631),('B-D', 0.346),('B-D', 0.18),('E-E', 0.227),('A-E', 0.681),('B-D', 0.9),('D-E', 0.82),('D-D', 0.923),('C-C', 0.972),('C-D', 0.199),('E-E', 0.173),('B-B', 0.146),('B-B', 0.879),('E-E', 0.461),('A-E', 0.064),('A-B', 0.496),('B-B', 0.078),('B-C', 0.968),('C-E', 0.373),('C-D', 0.49),('B-D', 0.269)]
    self.test_sw_patterns_17  =  [('B-E', 0.684),('E-E', 0.571),('C-C', 0.972),('A-B', 0.907),('B-C', 0.655),('A-C', 0.576),('C-C', 0.486),('D-D', 0.539),('C-E', 0.373),('C-D', 0.49)]

    self.test_sw_patterns_18  =  [('D-E', 0.136),('D-E', 0.519),('A-C', 0.685),('B-C', 0.861),('A-E', 0.943),('D-E', 0.601),('A-C', 0.288),('A-E', 0.628),('A-E', 0.575),('C-E', 0.847),('B-E', 0.197),('A-E', 0.034),('C-C', 0.647),('A-B', 0.83),('B-D', 0.072),('C-D', 0.743),('A-D', 0.516),('B-E', 0.459),('E-E', 0.637),('A-A', 0.46),('B-B', 0.139),('B-B', 0.571),('B-D', 0.312),('C-C', 0.654),('A-C', 0.414),('E-E', 0.889),('A-A', 0.568),('C-C', 0.391),('B-D', 0.741),('C-E', 0.39),('D-D', 0.526),('B-D', 0.085),('C-C', 0.061),('B-D', 0.386)]
    self.test_sw_patterns_19  =  [('A-E', 0.24),('C-D', 0.806),('B-B', 0.799),('C-E', 0.835),('C-C', 0.654),('A-E', 0.448)]

    self.test_sw_patterns_20  =   [('A-D', 0.515),('B-B', 0.484),('C-E', 0.369),('B-D', 0.504),('B-E', 0.003),('C-E', 0.479),('C-E', 0.947),('C-C', 0.986),('B-D', 0.412),('C-C', 0.805),('C-E', 0.263),('A-C', 0.274),('C-C', 0.719),('A-C', 0.718),('B-B', 0.272),('B-D', 0.309),('E-E', 0.458),('A-A', 0.6),('B-E', 0.375),('A-D', 0.203),('E-E', 0.144)]
    self.test_sw_patterns_21  =   [('A-B', 0.939),('A-E', 0.46),('A-D', 0.995),('C-C', 0.57),('A-C', 0.718),('A-C', 0.388)]
    #self.test_sw_patterns = [self.test_sw_patterns_1, self.test_sw_patterns_2, self.test_sw_patterns_3]
    #self.test_sw_patterns = [self.test_sw_patterns_8,self.test_sw_patterns_9]
    self.test_sw_patterns = [self.test_sw_patterns_20, self.test_sw_patterns_21]

    #self.best_sw_patterns_awgn8 = [('C-C', 0.144),('C-C', 0.429),('A-D', 0.493),('A-D', 0.562),('A-D', 0.768), ('A-D', 0.312), ('A-D', 0.417)]
    #self.best_sw_patterns_awgn6 = [('A-D', 0.417), ('C-C', 0.429), ('A-D', 0.822), ('B-C', 0.338), ('A-D', 0.312)]
    #self.best_sw_patterns_awgn4 = [('A-D', 0.827), ('A-D', 0.373), ('A-D', 0.939), ('A-D', 0.605), ('A-D', 0.312), ('A-D', 0.822)]
    #self.best_sw_patterns_awgn2 = 
    #self.best_sw_patterns_awgn0 = [('A-D', 0.312), ('A-D', 0.456), ('B-C', 0.338)]

    """ viterbi generator polynomial section """

    self.test_viterbi_gp_1 = [(15, 20568, 31282), (12, 289, 1620), (16, 3664, 6646), (13, 5890, 6271), (13, 4441, 1481), (15, 23443, 19340), (15, 29104, 5402)]
    self.test_viterbi_gp_2 = [(19 , 104171 , 43597),(16 , 25867 , 64068),(15 , 22748 , 20515),(18 , 49870 , 25915),(14 , 2670 , 13353),(19 , 267052 , 292632),(14 , 10117 , 10273),(16 , 19094 , 45608),(18 , 147012 , 171229),(17 , 59952 , 35021),(18 , 223244 , 166886),(19 , 165277 , 44828)]
    self.test_viterbi_gp_3 = [(15 , 4293 , 27673),(19 , 75437 , 395530),(13 , 567 , 1044),(18 , 175858 , 114619),(17 , 24678 , 113783),(15 , 14047 , 24322),(11 , 1504 , 330),(17 , 102984 , 108103),(17 , 49292 , 39302),(18 , 26581 , 142760),(11 , 320 , 517),(17 , 11484 , 2257),(13 , 7120 , 4105),(19 , 315807 , 489515),(18 , 16885 , 110965)]
    self.test_viterbi_gp_4 = [(14 , 12874 , 2088),(19 , 487833 , 272009),(18 , 138813 , 136592),(17 , 56883 , 48672),(15 , 32454 , 29350),(17 , 109434 , 66931),(13 , 6375 , 5183),(19 , 454203 , 467489),(15 , 11481 , 14041),(17 , 125755 , 97716),(13 , 5426 , 1463),(16 , 53269 , 27876),(18 , 117917 , 109517),(18 , 21198 , 252632),(17 , 58879 , 96029),(14 , 4112 , 11103),(18 , 203771 , 210048),(15 , 10070 , 13902),(16 , 37139 , 63104),(17 , 62779 , 86412),(14 , 8074 , 15694),(14 , 2897 , 70),(12 , 3541 , 1599),(11 , 736 , 452),(17 , 59265 , 11061),(17 , 128295 , 5901),(19 , 316991 , 234689),(16 , 20506 , 54096),(15 , 13684 , 2676),(13 , 8154 , 7138),(12 , 2495 , 3051),(19 , 76197 , 499063),(13 , 1394 , 4569),(12 , 3972 , 388),(15 , 11382 , 26842),(19 , 94049 , 382551)]

    self.test_viterbi_gp_5  = [(14, 9725, 8427, [1,1,1,1,0,1,0,0,1,0,0,0]), (17, 92104, 28536, [1,1,1,1,0,0,0,0,1,0,0,1]), (19, 125643, 152972, [1,1,1,1,0,1,0,0,1,0,1,0]), (18, 16519, 138370, [1,1,1,1,0,0,0,0,1,0,1,0]), (14, 1432, 528, [1,1,1,1,0,0,0,0,1,0,0,1]), (17, 68853, 40425, [1,1,1,1,0,0,0,0,1,0,0,1]), (17, 22831, 5325, [1,1,1,1,0,1,0,0,0,0,1,0])]
    self.test_viterbi_gp_6  = [(16 , 52915 , 12072,  [1,1,1,1,0,0,0,0,0,1,0,1]),(18 , 214645 , 70304,  [1,1,1,1,0,0,0,0,0,0,1,1]),(17 , 37298 , 26770,  [1,1,1,1,0,0,0,1,0,0,1,0]),(11 , 512 , 1570,  [1,0,1,0,0,0,1,0,1,0,1,1])]
    self.test_viterbi_gp_7  = [(19 , 481747 , 86084,  [1,1,1,1,0,0,0,0,0,1,0,1]),(18 , 245535 , 156150,  [1,1,1,1,1,0,0,0,1,0,0,0]),(18 , 163543 , 62859,  [1,1,1,1,1,1,0,0,0,0,0,0]),(18 , 214645 , 70304,  [1,1,1,1,0,0,0,0,0,0,1,1]),(18 , 186220 , 217112,  [1,1,1,1,0,1,0,0,0,0,0,1])]
    self.test_viterbi_gp_8  = [(14 , 13841 , 9738 , [1,1,1,1,0,0,1,0,0,1,0,0]),(15 , 28995 , 30779 , [1,1,1,1,1,0,0,0,1,0,0,0]),(17 , 108863 , 129336 , [1,1,1,1,0,0,0,1,0,1,0,0]),(15 , 10578 , 10911 , [1,1,1,1,0,0,1,0,0,0,1,0])]
    self.test_viterbi_gp_9  = [(11 , 1360 , 1 , [0,1,0,1,0,0,1,1,1,0,1,0]),(11 , 1775 , 1808 , [1,1,1,1,0,1,1,0,0,0,0,0]),(11 , 752 , 483 , [1,1,1,1,1,0,0,0,0,0,0,1])]
    self.test_viterbi_gp_10 = [(11 , 1775 , 1808 , [1,1,1,1,0,1,1,0,0,0,0,0]),(11 , 885 , 26 , [1,1,0,1,0,0,1,1,0,0,0,1]),(19 , 168183 , 20892 , [1,1,1,1,1,0,0,0,1,0,0,0]),(11 , 752 , 483 , [1,1,1,1,1,0,0,0,0,0,0,1])]

    self.test_viterbi_gp_11 = [(13 , 5248 , 4576 , [1,1,1,0]),(12 , 16 , 2212 , [1,1,0,1]),(11 , 6 , 675 , [1,0,1,1]),(13 , 2419 , 669 , [1,1,1,0])]
    self.test_viterbi_gp_12 = [(12 , 1423 , 16 , [1,1,0,1]), (11, 861, 2, [0,1,1,1]), (19, 93986, 444204, [0,1,1,1]), (11 , 505 , 551 , [0,1,1,1]), (13 , 4322 , 1875 , [0,1,1,1])]

    self.all_viterbi_gps  = [self.test_viterbi_gp_11, self.test_viterbi_gp_12]



    #self.all_viterbi_gps  = [self.test_viterbi_gp_1, self.test_viterbi_gp_2, self.test_viterbi_gp_3]
    self.best_viterbi_gps = [(18 , 147012 , 171229), (15 , 4293 , 27673),(13 , 5890 , 6271),(15 , 14047 , 24322),(11 , 320 , 517)]

    """ downconvert shift values section"""
    #self.test_dcs_values = [(0.765),(0.654),(0.734),(0.665),(0.653),(0.757),(0.784),(0.562),(0.597),(0.603),(0.957),(0.964),(0.658),(0.814),(0.697),(0.892),(0.723),(0.614),(0.7),(0.857),(0.822),(0.744),(0.891),(0.684),(0.835),(0.802),(0.774),(0.601),(0.51),(0.762),(0.994),(0.907),(0.933),(0.735),(0.84),(0.925),(0.803),(0.795),(0.689),(0.982),(0.641),(0.918),(0.879),(0.713),(0.897),(0.76),(0.672),(0.751),(0.647),(0.792),(0.704),(0.956),(0.722),(0.769),(0.574),(0.557),(0.997),(0.758),(0.905),(0.927),(0.602),(0.8),(0.752),(0.924),(0.998),(0.932),(0.794),(0.797),(0.992),(0.606),(0.604),(0.731),(0.727),(0.98),(0.558),(0.763),(0.586),(0.903),(0.659),(0.657),(0.613),(0.669),(0.62),(0.73),(0.981),(0.645),(0.746),(0.505),(0.809),(0.871)]

    #self.test_dcs_values = [(0.898),(0.798),(0.399),(0.025),(0.199),(0.643),(0.137),(0.073),(0.205),(0.251),(0.753),(0.082),(0.818),(0.267),(0.347),(0.399),(0.84),(0.898)]

    #self.test_dcs_values = [(0.374),(0.509),(0.204),(0.691),(0.345),(0.944),(0.464),(0.26),(0.722),(0.823),(0.46),(0.961),(0.178),(0.129),(0.944),(0.561),(0.594),(0.036),(0.484),(0.731),(0.15),(0.872),(0.711),(0.722),(0.062),(0.793),(0.132),(0.582)]

    #self.test_dcs_values = [(0.499),(0.007),(0.044),(0.277),(0.876),(0.081),(0.573),(0.986),(0.969),(0.238),(0.007),(0.092),(0.31),(0.014),(0.662),(0.837),(0.556),(0.673),(0.428),(0.558),(0.072),(0.22),(0.504),(0.906),(0.861),(0.797),(0.091),(0.45),(0.258),(0.639),(0.969)]

    self.test_dcs_values = [(0.243),(0.392),(0.141),(0.209),(0.174),(0.23),(0.132),(0.755),(0.673),(0.379),(0.357),(0.468),(0.325),(0.681),(0.975),(0.85),(0.601),(0.6),(0.472),(0.99),(0.502),(0.314),(0.392),(0.141),(0.174),(0.147),(0.375),(0.755),(0.896),(0.384),(0.443),(0.527),(0.037),(0.638),(0.132),(0.229),(0.415),(0.23),(0.508),(0.966),(0.357),(0.654),(0.257),(0.79),(0.131),(0.588),(0.378),(0.629),(0.883),(0.318),(0.111),(0.954),(0.009),(0.461),(0.645),(0.993),(0.785),(0.105),(0.302),(0.404),(0.553),(0.02),(0.209),(0.546),(0.186),(0.533),(0.273),(0.671),(0.91),(0.03),(0.534),(0.493),(0.969),(0.655),(0.831),(0.227),(0.173),(0.583)]


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
                                                              #'parameters'           : (600, 0.54, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                                              'parameters'           : (1500, 0.435, 0.579, 10000, 2, 98, 0.403, 0.21, 0.828, 0.025) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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
                                                              #'parameters'           : (600, 0.54, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                                              'parameters'           : (1500, 0.986, 0.015, 10000, 2, 98, 0.403, 0.21, 0.828, 0.025) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves


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



                               # THIS WORKS!!!! 10 bits per second
                               'LB28-2560-8-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 8,
                                                              'symbol_block_size'    : 2560,
                                                              'fft_filter'           : (-6, 6, -6, 6),
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                                              'parameters'           : (1500, 0.215, 0.091, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'persistent_search'    : (1, 0.95, -0.005, "yes"), #hi range, lo range, inc, scan entire range
                                                             },


                               #  80 bits per second
                             'LB28-320-8-2-37-I3E8-FEC-FDM': {'inherit_from'        : 'LB28-320-8-2-37-I3E8-FEC',
                                                              'FDM'                 : "yes",
                                                              #'FDM_parameters'      : (2, 93), #frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.171,

                                                              #'FDM_parameters'      : (2, 94), #frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.907,

                                                              #'FDM_parameters'      : (2, 93.25), #frequency division mutiplexing: multiplier, separation
                                                              'downconvert_shift'    : 0.133,

                                                              'FDM_parameters'      : [2, 85.865], #frequency division mutiplexing: multiplier, separation


                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.246), 
                                                              #'fft_filter'           : (-12, 12, -12, 12),

                                                              #'parameters'           : (1500, 0.069, 0.033, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

                                                             },


                               # THIS  WORKS!!!! 80 bits per second
                               'LB28-320-8-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 8,
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'persistent_search'    : (1, 0.95, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                                                              'symbol_block_size'    : 320,

                                                              #'fft_filter'           : (-20, 16, -16, 20),
                                                              #'downconvert_shift'    : 0.514,
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-C', 0.701), 

                                                              #4.35 EBN0
                                                              #'fft_filter'           : (-16, 12, -12, 16),
                                                              #'downconvert_shift'    : 0.898,
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'B-B', 0.928), 

                                                              'fft_filter'           : (-12, 12, -12, 12),
                                                              'downconvert_shift'    : 0.898,
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'B-B', 0.928), 
                                                              #3.8 EBN0
                                                              #'parameters'           : (1500, 0.661, 0.063, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #2.6 EBN0
                                                              #'parameters'           : (1500, 0.69, 0.046, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              #1.7 EBN0
                                                              'parameters'           : (1500, 0.069, 0.033, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),


                                                              'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                                              'I3_combine'           : ocn.INTRA_COMBINE_TYPE6,

                                                              #'carrier_separation'   : 59,   or 75 with .932 downconvert shift

                                                              #'fec_params'           : (15 , 13684 , 2676, [1,0,1,0] ),
                                                             },



                               #  80 bits per second X 4....YES THIS WORKS!!!!
                             'LB28-400-8-2-37-I3E8-FEC-FDM': {'inherit_from'        : 'LB28-400-8-2-37-I3E8-FEC',
                                                              'FDM'                 : "yes",
                                                              #'downconvert_shift'    : 0.814,
                                                              #'FDM_parameters'      : (2, 82.544), #frequency division mutiplexing: multiplier, separation
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-B', 0.649), 

                                                              #'FDM_parameters'      : [4, 82.544, 160.073], #149.295 frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.598,

                                                              #'FDM_parameters'      : (2, 84.243), #frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.709,
                                                              #'FDM_parameters'      : (2, 77.934), #frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.633,
                                                              #'FDM_parameters'      : (2, 95.172), #frequency division mutiplexing: multiplier, separation
                                                              #'downconvert_shift'    : 0.835,


                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'E-E', 0.861), 
                                                              'FDM_parameters'      : [2, 76.607], #frequency division mutiplexing: multiplier, separation
                                                              'downconvert_shift'    : 0.656,


                                                              #'sample_rate'          : 48000,


                                                             },


                               # THIS WORKS!!!! 64.4 bits per second
                               'LB28-400-8-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 8,
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'persistent_search'    : (1, 0.95, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                                                              'symbol_block_size'    : 400,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'downconvert_shift'    : 0.514,
                                                              'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-C', 0.701), 
                                                              'extrapolate'          : 'no', #no rotation tables yet!!!
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                                              'parameters'           : (1500, 0.661, 0.063, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'I3_combine'           : ocn.INTRA_COMBINE_TYPE6,

                                                             },


                               # THIS WORKS!!!! 32 bits per second
                               'LB28-800-8-2-37-I3E8-FEC': {'inherit_from'        : 'LB28-6400-64-2-37-I3E8-FEC',
                                                              'pulses_per_block'     : 8,
                                                              'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'persistent_search'    : (1, 0.95, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                                                              'symbol_block_size'    : 800,
                                                              #'symbol_block_size'    : 320,
                                                              'fft_filter'           : (-20, 16, -16, 20),
                                                              'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'C-E', 0.776), 
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'B-E', 0.921), 
                                                              #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-D', 0.61), 
                                                              'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-A', 0.003), 

                                                              'extrapolate'          : 'no', #no rotation tables yet!!!
                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                                                              #'parameters'           : (1500, 0.215, 0.091, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'parameters'           : (1500, 0.115, 0.028, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

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

                                                              #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,


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

                                                              'tx_filter'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 48, 5, 50),


                                                              #'I3_combine'           : ocn.INTRA_COMBINE_TYPE5,
                                                              'I3_combine'           : ocn.INTRA_COMBINE_TYPE9,
                                                              'I3_extract'           : ocn.INTRA_EXTRACT_TYPE4,
                                                              'doppler_pulse_interpolation' : 'Chebyshev',
                                                              #'doppler_pulse_interpolation' : 'Cubic-Spline',
                                                              #'doppler_pulse_interpolation' : 'Pchip',

                                                              'doppler_adjust'           : ocn.DOPPLER_ADJUST_NONE,

                                                              #'disposition_increment' : 3e-1,

                                                              #'rotation_increments'  : 20,

                                                              #'persistent_search'    : (1, 0.95, -0.002, "yes"), #hi range, lo range, inc, scan entire range

                                                              #'fft_filter'           : (-1, 1, -1, 1),
                                                              'fft_filter'           : (-2, 2, -2, 2),
                                                              #'fft_filter'           : (-3, 2, -2, 3),
                                                              #'fft_interpolate'      : (-3, 2, -2, 3),
                                                              'fft_interpolate'      : (-2, 2, -2, 2),
                                                              #'fft_interpolate'      : (-3, 3, -3, 3),

                                                              #PATTERN 19 PATTERN 27 PATTERN 36 

                                                              'start_seq'            : '2_of_8',
                                                              'phase_align'          : 'start_seq',
                                                              #'extrapolate'           : 'yes',
                                                              'extrapolate'           : 'no',
                                                              'extrapolate_seqlen'       : 8,
                                                              #'downconvert_shift'    : 0.535,
                                                              #'downconvert_shift'    : 0.533,
                                                              #'downconvert_shift'    : 0.505,   # use this to generate rotation tables. this works great for 8k simulation
                                                              #'downconvert_shift'    : 0.530,  # this works great for 8k tx with 8k rx and 48k tx with 8k rx
                                                              #'downconvert_shift'    : 0.55,  # this works great for 8k tx with 8k rx and 48k tx with 8k rx
                                                              'downconvert_shift'    : 0.53,  # this works great for 8k tx with 8k rx and 48k tx with 8k rx
                                                              #'downconvert_shift'    : 0.851,  #  this is for best audio decode at 1480hz using rta built from 0.530 at 8000hz sample rate
                                                              #'downconvert_shift'    : 0.502,  # this works great for 8k simulation and 48k simulation
                                                              #'downconvert_shift'    : 0.5,
                                                              #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MODULATION_SPECIFIC,

                                                              'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,

                                                              'I3_pulse_shape_index' : 3,
                                                              #'parameters'           : (1500, 0.763, 0.107, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                                                              'parameters'           : (1500, 0.822, 0.997, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

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


                                                              'I3_combine'           : ocn.INTRA_COMBINE_TYPE9,
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
                                                              #'parameters'           : (1500, 0.68, 0.89, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves
                                                              'parameters'           : (1500, 0.76, 0.191, 10000, 2, 98, 0.403, 0.21, 0.828, 0.025) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

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

    """ default to test block """
    self.initBlock = self.getInitializationBlock()


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
                                    'pulse_train_sigma'    : 5.0,
                                    'pulse_start_envelope_sigma'    : 7,
                                    'doppler_pulse_interpolation'    : 'Chebyshev',
                                    'doppler_adjust'           : ocn.DOPPLER_ADJUST_NONE,
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
                                    'persistent_search'    : (0.89, 0.5, -0.1, "no"),
                                    'I3_combine'           : ocn.INTRA_COMBINE_TYPE9,
                                    'FDM_parameters'       : [0, 0],
                                    'FDM'                  : "no",
                                    'rotation_increments'  : 40,
                                    'rx_filter'            : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
                                    'tx_filter'            : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
                                    'rx_filter2'           : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
                                    'tx_filter2'           : (ocn.FILTER_NONE, ocn.FILTER_NONE, 0, 0, 0),  #type, width, repeats, order
                                    'resample_params'      : [ocn.RESAMPLE_UNAVAILABLE, 0, 0, 0], # available, low freq, hi freq
                                    'resample_params_48k'  : [ocn.RESAMPLE_UNAVAILABLE, 0, 0, 0], # available, low freq, hi freq
                                    'dcs_type'             : ocn.DCS_GENERAL,
                                    'dcs_by_frequency'     : {},
                                    'pattern_by_msglen'    : {},


                                   }


  def getTxMessageLength(self, message_text):
    rotation_sequence_length = 8
    return len(message_text) + rotation_sequence_length


  def setSliderAwgn(self, new_value):
    self.slider_awgn = new_value

  def getSliderAwgn(self):
    return self.slider_awgn

  def setSliderAmplitude(self, new_value):
    self.slider_amplitude = new_value

  def getSliderAmplitude(self):
    return self.slider_amplitude

  def setSliderCarrierSeparation(self, new_value):
    self.slider_carrier_separation = new_value

  def getSliderCarrierSeparation(self):
    return self.slider_carrier_separation


  def getTestModeEnabled(self):
    return self.test_mode_enabled

  def setTestModeEnabled(self, enabled):
    self.test_mode_enabled = enabled


  def setAperture(self, aperture):
    self.aperture = aperture

  def getAperture(self):
    return self.aperture

  def setBiasFilterValue(self, bias):
    self.bias_filter_value = bias

  def getBiasFilterValue(self):
    return self.bias_filter_value

  def setInputGain(self, gain):
    self.input_gain = gain

  def getInputGain(self):
    return self.input_gain

  def setOutputGain(self, gain):
    self.output_gain = gain

  def getOutputGain(self):
    return self.output_gain

  def getDownconvertShift(self):
    if self.dcs_type == ocn.DCS_GENERAL:
      return self.downconvert_shift
    elif self.dcs_type == ocn.DCS_FREQUENCY_SPECIFIC:
      key_name = str(self.center_frequency)
      if key_name in self.dcs_by_frequency:
        return self.dcs_by_frequency[key_name]
      else:
        return self.downconvert_shift

  def getRxSampleRate(self):
    use_hifi_rx = self.form_gui.window['cb_enable_hifi_input_sampling'].get()
    if use_hifi_rx:
      return self.rx_sample_rate * 6  # 8000 * 6  #self.sample_rate * 6
    else:
      return self.rx_sample_rate  #8000 # self.sample_rate

  def getRxSymbolBlockSize(self):
    use_hifi_rx = self.form_gui.window['cb_enable_hifi_input_sampling'].get()
    if use_hifi_rx:
      return self.rx_symbol_block_size * 6  #3200 * 6 #self.symbol_block_size * 6
    else:
      return self.rx_symbol_block_size  # 3200 # self.symbol_block_size

  def getTxSampleRate(self):
    use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
    if use_hifi_tx:
      return self.tx_sample_rate * 6  #8000 * 6 #self.sample_rate * 6
    else:
      return self.tx_sample_rate #8000 #self.sample_rate

  def getTxSymbolBlockSize(self):
    use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
    if use_hifi_tx:
      return self.tx_symbol_block_size * 6 #3200 * 6 #self.symbol_block_size * 6
    else:
      return self.tx_symbol_block_size # 3200 #self.symbol_block_size

  def getTxResampleParams(self):
    use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
    if use_hifi_tx:
      return self.resample_params_48k
    else:
      return self.resample_params

  def getRealMode(self, values, form_gui):
    if values['cb_use_prod_modes'] == True:
      form_gui.osmod.useProdMode()
      mode = values['combo_main_modem_prod_modes']
    else:
      form_gui.osmod.useTestMode()
      mode = values['combo_main_modem_modes']
    return mode


  def getSignalSquelch(self):
    return self.signal_squelch_value

  def setSignalSquelch(self, newvalue):
    self.signal_squelch_value = newvalue

  def getInitializationBlock(self):
    return self.modulation_initialization_block

  def getInitBlockParam(self, mode, param):
    #return self.modulation_initialization_block[mode][param]
    return self.initBlock[mode][param]

  def getInitBlockMode(self, mode):
    #return self.modulation_initialization_block[mode]
    return self.initBlock[mode]

  def useTestMode(self): 
    self.debug.info_message("switching to test mode initialization block")
    self.initBlock = self.getInitializationBlock()

  def useProdMode(self): 
    self.debug.info_message("switching to prod mode initialization block")
    self.initBlock = self.prodparams.getInitializationBlock()

  def setCenterFrequency(self, centfreq):
    self.center_frequency = centfreq

  def getCenterFrequency(self):
    return self.center_frequency

  def resetOptionalInitParamDefaults(self, mode):
    self.debug.info_message("resetOptionalInitParamDefaults")
    for param in self.optional_param_defaults:
      self.optional_param_values[param] = self.optional_param_defaults[param]

  def processOptionalInitParams(self, mode):
    self.debug.info_message("processOptionalInitParams")
    for param in self.optional_param_values:
      if param in self.getInitBlockMode(mode):
        self.optional_param_values[param] = self.getInitBlockParam(mode, param)


  """ recursive... multi level inheritance"""
  def processInheritFrom(self, mode):
    self.debug.info_message("processInheritFrom mode: " + str(mode))
    if 'inherit_from' in self.getInitBlockMode(mode):
      self.processInheritFrom(self.getInitBlockParam(mode, 'inherit_from'))
    for param in self.getInitBlockMode(mode):
      self.optional_param_values[param] = self.getInitBlockParam(mode, param)

  def getParam(self, mode, param_name):
    if param_name in self.getInitBlockMode(mode):
      param_value = self.getInitBlockParam(mode, param_name)
      #self.debug.info_message("param " + str(param_name) + " = " + str(param_value))
      sys.stdout.write("                    \'"  + str(param_name) + "\' : " + str(param_value) + "\n")

      return param_value  #  self.getInitBlockParam(mode, param_name]
    else:
      param_value = self.optional_param_values[param_name]
      #self.debug.info_message("param " + str(param_name) + " = " + str(param_value))
      sys.stdout.write("                    \'"  + str(param_name) + "\' : " + str(param_value) + "\n")
      return param_value  #  self.optional_param_values[param_name]

  def getOptionalParam(self, param):
    return self.optional_param_values[param]


  class C_mode(ctypes.Structure):
    _fields_ = [   ("mode_name",           ctypes.c_char_p),
                   ("symbol_block_size",   ctypes.c_int),
                   ("symbols_per_block",   ctypes.c_int),
                   ("pulses_per_block",    ctypes.c_int),
                   ("sample_rate",         ctypes.c_int),
                   ("num_carriers",        ctypes.c_int),
                   ("carrier_separation",  ctypes.c_int),
                   ("baseband_conversion", ctypes.c_char_p),
                   ("fft_filter",          ctypes.POINTER(ctypes.c_float)),
                   ("fft_intrpolate",      ctypes.POINTER(ctypes.c_float)),
                   ("downconvert_shift",   ctypes.c_int)  ]

  def C_setCurrentMode(self):
    try:
      #c_current_mode = 

      c_int_array = ptoc_int_array(python_list)

      self.osmod.compiled_lib.find_mode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
      self.osmod.compiled_lib.find_mode.restype = C_mode
      mode = self.osmod.compiled_lib.find_mode(c_int_array, len(python_list))

      mode.x=10
      mode.y=20

    except:
      self.debug.error_message("Exception in C_setCurrentMode: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def resetDecoder(self):
    self.debug.info_message("resetDecoder")
    try:
      self.demod_2fsk8psk.remainder = np.array([])

      """ reset rotation angles """
      self.detector.rotation_angles[0] = 0.0
      self.detector.rotation_angles[1] = 0.0

    except:
      self.debug.error_message("Exception in resetDecoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def setInitializationBlockSR(self, mode, sample_rate, symbol_block_size):
    try:
      self.debug.info_message("setInitializationBlock")
      self.setInitializationBlockCommon(mode, sample_rate, symbol_block_size, True)
    except:
      self.debug.error_message("Exception in setInitializationBlockSR: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def setInitializationBlock(self, mode):
    try:
      self.debug.info_message("setInitializationBlock")
      self.setInitializationBlockCommon(mode, 0, 0, False)
    except:
      self.debug.error_message("Exception in setInitializationBlock: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def setInitializationBlockCommon(self, mode, sample_rate, symbol_block_size, override_sr_and_sbs):
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

      sys.stdout.write("\n")
      sys.stdout.write("\n")
      sys.stdout.write("          \'" + str(mode) + "\' :{ \n")

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
      self.pulse_start_envelope_sigma    = self.getParam(mode, 'pulse_start_envelope_sigma')
      self.pulse_train_sigma    = self.getParam(mode, 'pulse_train_sigma')
      self.doppler_pulse_interpolation    = self.getParam(mode, 'doppler_pulse_interpolation')
      self.doppler_adjust       = self.getParam(mode, 'doppler_adjust')
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
      self.persistent_search   = self.getParam(mode, 'persistent_search')
      self.I3_combine          = self.getParam(mode, 'I3_combine')
      self.FDM_parameters      = self.getParam(mode, 'FDM_parameters')
      self.FDM                 = self.getParam(mode, 'FDM')
      self.rotation_increments = self.getParam(mode, 'rotation_increments')
      self.rx_filter           = self.getParam(mode, 'rx_filter')
      self.tx_filter           = self.getParam(mode, 'tx_filter')
      self.rx_filter2          = self.getParam(mode, 'rx_filter2')
      self.tx_filter2          = self.getParam(mode, 'tx_filter2')
      self.resample_params     = self.getParam(mode, 'resample_params')
      self.resample_params_48k = self.getParam(mode, 'resample_params_48k')
      self.dcs_type            = self.getParam(mode, 'dcs_type')
      self.dcs_by_frequency    = self.getParam(mode, 'dcs_by_frequency')
      self.pattern_by_msglen   = self.getParam(mode, 'pattern_by_msglen')



      # these used for hi-hi settings
      self.rx_sample_rate = self.sample_rate
      self.tx_sample_rate = self.sample_rate
      self.rx_symbol_block_size = self.symbol_block_size
      self.tx_symbol_block_size = self.symbol_block_size


      override_extrapolate = self.form_gui.window['cb_override_extrapolate'].get()
      if override_extrapolate:
        self.extrapolate          = "yes"


      sys.stdout.write("          }, \n")
      sys.stdout.write("\n")
      sys.stdout.write("\n")


      if override_sr_and_sbs:
        self.sample_rate = sample_rate
        self.symbol_block_size = symbol_block_size


      fdmsep_override_checked = self.form_gui.window['cb_overridefdmseparation'].get()
      if fdmsep_override_checked:
        self.debug.info_message("FDM_parameters: " + str(self.FDM_parameters))
        pair_level = self.form_gui.window['combo_fdmpairlevel'].get()
        fdm_separation = float(self.form_gui.window['in_fdmseparation'].get())
        if pair_level == "Scale 1":
          self.FDM_parameters[1] = fdm_separation
        elif pair_level == "Scale 2":
          self.FDM_parameters[2] = fdm_separation
        elif pair_level == "Scale 3":
          self.FDM_parameters[3] = fdm_separation


      combine_override_checked = self.form_gui.window['cb_overridecombine'].get()
      if combine_override_checked:
        selected_type = self.form_gui.window['combo_intra_combine_type'].get()

        if selected_type == 'Type 1':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE1
        elif selected_type == 'Type 2':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE2
        elif selected_type == 'Type 3':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE3
        elif selected_type == 'Type 4':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE4
        elif selected_type == 'Type 5':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE5
        elif selected_type == 'Type 6':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE6
        elif selected_type == 'Type 7':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE7
        elif selected_type == 'Type 8':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE8
        elif selected_type == 'Type 9':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE9
        elif selected_type == 'Type 10':
          self.I3_combine = ocn.INTRA_COMBINE_TYPE10


      self.fec.init_params(self.FEC)

      """ keep track of the chunks being processed """
      self.chunk_num = 0

      self.rotation_tables      = self.opd.readRotationTablesFromFile(mode)

      """
      self.info                 = self.getInitBlockParam(mode, 'info')
      self.encoder_callback     = self.getInitBlockParam(mode, 'encoder_callback')
      self.decoder_callback     = self.getInitBlockParam(mode, 'decoder_callback')
      self.text_encoder         = self.getInitBlockParam(mode, 'text_encoder')
      self.mode_selector        = self.getInitBlockParam(mode, 'mode_selector')
      self.symbol_block_size    = self.getInitBlockParam(mode, 'symbol_block_size')
      self.sample_rate          = self.getInitBlockParam(mode, 'sample_rate')
      self.parameters           = self.getInitBlockParam(mode, 'parameters')
      self.carrier_separation   = self.getInitBlockParam(mode, 'carrier_separation')
      self.num_carriers         = self.getInitBlockParam(mode, 'num_carriers')
      self.pulses_per_block     = self.getInitBlockParam(mode, 'pulses_per_block')
      self.detector_function    = self.getInitBlockParam(mode, 'detector_function')
      self.symbol_wave_function = self.getInitBlockParam(mode, 'symbol_wave_function')
      self.extraction_points    = self.getInitBlockParam(mode, 'extraction_points')
      self.phase_extraction     = self.getInitBlockParam(mode, 'phase_extraction')
      self.baseband_conversion  = self.getInitBlockParam(mode, 'baseband_conversion')
      self.process_debug        = self.getInitBlockParam(mode, 'process_debug')
      self.fft_filter           = self.getInitBlockParam(mode, 'fft_filter')
      self.fft_interpolate      = self.getInitBlockParam(mode, 'fft_interpolate')
      self.modulation_object    = self.getInitBlockParam(mode, 'modulation_object')
      self.demodulation_object  = self.getInitBlockParam(mode, 'demodulation_object')
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

        self.debug.info_message("self.filtRRC_coef_pre: " + str(self.filtRRC_coef_pre))
        self.debug.info_message("self.filtRRC_coef_main: " + str(self.filtRRC_coef_main))
        self.debug.info_message("self.filtRRC_coef_post: " + str(self.filtRRC_coef_post))

      elif self.pulses_per_block == 512:
        """ calculate the RRC coefficients for 512ths carrier"""
        divisor = 512
        self.filtRRC_coef_pre, self.filtRRC_coef_main, self.filtRRC_coef_post = self.demod_2fsk8psk.filterSpanRRC( int(self.symbol_block_size/divisor), rrc_alpha, rrc_T, self.sample_rate)

        self.filtRRC_fivehundredtwelfth_wave = [0] * divisor
        self.filtRRC_fivehundredtwelfth_wave[0] = np.append(self.filtRRC_coef_main, np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)) )
        for i in range(1,divisor-1):
          self.filtRRC_fivehundredtwelfth_wave[i] = np.append(np.zeros(int((self.symbol_block_size*i)/divisor)), self.filtRRC_coef_main)
          self.filtRRC_fivehundredtwelfth_wave[i] = np.append(self.filtRRC_fivehundredtwelfth_wave[i], np.zeros(int((self.symbol_block_size*(divisor-i-1))/divisor)))
        self.filtRRC_fivehundredtwelfth_wave[divisor-1] = np.append(np.zeros(int((self.symbol_block_size*(divisor-1))/divisor)), self.filtRRC_coef_main)

        self.debug.info_message("self.filtRRC_coef_pre: " + str(self.filtRRC_coef_pre))
        self.debug.info_message("self.filtRRC_coef_main: " + str(self.filtRRC_coef_main))
        self.debug.info_message("self.filtRRC_coef_post: " + str(self.filtRRC_coef_post))

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
                             'LB28-2560-8-2-37-I3E8-FEC' : self.getPersistentData('LB28-2560-8-2-37-I3E8-FEC',   values),
                             'LB28-320-8-2-37-I3E8-FEC' : self.getPersistentData('LB28-320-8-2-37-I3E8-FEC',   values),
                             'LB28-400-8-2-37-I3E8-FEC' : self.getPersistentData('LB28-400-8-2-37-I3E8-FEC',   values),
                             'LB28-320-8-2-37-I3E8-FEC-FDM' : self.getPersistentData('LB28-320-8-2-37-I3E8-FEC-FDM',   values),
                             'LB28-400-8-2-37-I3E8-FEC-FDM' : self.getPersistentData('LB28-400-8-2-37-I3E8-FEC-FDM',   values),

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
            self.getInitBlockParam(mode, 'symbol_block_size'),
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
    form_gui.window['in_symbol_block_size'].update(self.getInitBlockParam(mode, 'symbol_block_size'))
    form_gui.window['combo_chunk_options'].update('30')
    form_gui.window['cb_enable_align'].update(True)
    form_gui.window['option_carrier_alignment'].update('100')
    form_gui.window['cb_enable_separation_override'].update(True)
    form_gui.window['cb_display_phases'].update(False)
    form_gui.window['option_chart_options'].update('Both')
    form_gui.window['cb_override_rrc_alpha'].update(False)
    form_gui.window['in_rrc_alpha'].update(self.getInitBlockParam(mode, 'parameters')[1])
    form_gui.window['cb_override_rrc_t'].update(False)
    form_gui.window['in_rrc_t'].update(self.getInitBlockParam(mode, 'parameters')[2])
    form_gui.window['btn_slider_awgn'].update(8)
    form_gui.window['slider_amplitude'].update(1.0)
    form_gui.window['slider_carrier_separation'].update(self.getInitBlockParam(mode, 'carrier_separation'))

    self.setFromCodeDefaults(mode)


  def setFromCodeDefaults(self, mode):

    updated_data = (
            True,
            '3:peter piper',
            False,
            self.getInitBlockParam(mode, 'symbol_block_size'),
            '30',
            True,
            '100',
            True,
            False,
            'Both',
            False,
            self.getInitBlockParam(mode, 'parameters')[1],
            False,
            self.getInitBlockParam(mode, 'parameters')[2],
            '8.0',
            '1.0',
            self.getInitBlockParam(mode, 'carrier_separation')
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

  """
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

      self.setStandingWaveValues(frequency)


      self.debug.info_message("calcCarrierFrequenciesFromFFT. frequencies: " + str(frequency))

      return frequency

    except:
      self.debug.error_message("Exception in calcCarrierFrequenciesFromFFT: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
  """


  def calcCarrierFrequenciesSR(self, center_frequency, separation_override, sample_rate):
    self.debug.info_message("calcCarrierFrequenciesSR")
    #sample_rate = self.sample_rate
    return self.calcCarrierFrequenciesCommon(center_frequency, separation_override, sample_rate)


  def calcCarrierFrequencies(self, center_frequency, separation_override):
    self.debug.info_message("calcCarrierFrequencies")
    sample_rate = self.sample_rate
    return self.calcCarrierFrequenciesCommon(center_frequency, separation_override, sample_rate)

  def calcCarrierFrequenciesCommon(self, center_frequency, separation_override, sample_rate):
    self.debug.info_message("calcCarrierFrequenciesCommon")
    try:
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


      self.setStandingWaveValues(frequency, sample_rate)


      self.debug.info_message("calcCarrierFrequencies. frequencies: " + str(frequency))

      return frequency

    except:
      self.debug.error_message("Exception in calcCarrierFrequencies: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def setStandingWaveValues(self, frequency, sample_rate):
    self.debug.info_message("calcCarrierFrequencies")

    try:

      def set_sw_values():
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, standingwave_location, sample_rate)
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


      offsets_override_checked = self.form_gui.window['cb_override_standingwaveoffsets'].get()
      if offsets_override_checked:
        standingwave_location = float(self.form_gui.window['in_standingwavelocation'].get())
        standingwave_pattern = self.form_gui.window['combo_standingwave_pattern'].get()
        set_sw_values()

      else:
        if self.i3_offsets_type == ocn.OFFSETS_MANUAL:
          self.debug.info_message("processing OFFSETS_MANUAL")
          standingwave_pattern  = self.i3_parameters[3]
          standingwave_location = float(self.i3_parameters[4])
          self.debug.info_message("standingwave_pattern: " + str(standingwave_pattern))
          self.debug.info_message("standingwave_pattern: " + str(standingwave_location))
          set_sw_values()

          #self.i3_offsets = [self.i3_parameters[3], self.i3_parameters[4], self.i3_parameters[5], self.i3_parameters[6] ]
        elif self.i3_offsets_type == ocn.OFFSETS_MSGLEN_SPECIFIC:
          self.debug.info_message("processing pattern as OFFSETS_MSGLEN_SPECIFIC")
          max_message_length = int(self.form_gui.window['combo_max_message_length'].get())
          key = str(max_message_length)
          if key in self.pattern_by_msglen:
            self.debug.info_message("located pattern for msglen")
            standingwave_pattern  = self.pattern_by_msglen[key][0]
            standingwave_location = float(self.pattern_by_msglen[key][1])
            self.debug.info_message("standingwave_pattern: " + str(standingwave_pattern))
            self.debug.info_message("standingwave_pattern: " + str(standingwave_location))
            set_sw_values()
          else:
            #default back to manual setting
            self.debug.info_message("unable to locate pattern for msglen. defaulting to OFFSETS_MANUAL")
            standingwave_pattern  = self.i3_parameters[3]
            standingwave_location = float(self.i3_parameters[4])
            self.debug.info_message("standingwave_pattern: " + str(standingwave_pattern))
            self.debug.info_message("standingwave_pattern: " + str(standingwave_location))
            set_sw_values()


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

    except:
      self.debug.error_message("Exception in setStandingWaveValues: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


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
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.358, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN2:   # 2
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.666, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN3:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.133, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN4:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.338, sample_rate)
        return phase_list_low[1], phase_list_low[2], phase_list_high[1], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN5:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.233, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN6:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.144, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN7:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.839, sample_rate)
        return phase_list_low[4], phase_list_low[4], phase_list_high[4], phase_list_high[4]
      elif pattern == ocn.OFFSETS_PATTERN8:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.506, sample_rate)
        return phase_list_low[2], phase_list_low[4], phase_list_high[2], phase_list_high[4]
      elif pattern == ocn.OFFSETS_PATTERN9:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.429, sample_rate)
        return phase_list_low[2], phase_list_low[2], phase_list_high[2], phase_list_high[2]
      elif pattern == ocn.OFFSETS_PATTERN10:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.839, sample_rate)
        return phase_list_low[1], phase_list_low[1], phase_list_high[1], phase_list_high[1]

      elif pattern == ocn.OFFSETS_PATTERN11:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.821, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN12:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.827, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN13:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.193, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN14:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.196, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN15:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.312, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN16:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.749, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN17:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.827, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN18:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.562, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN19:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.657, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN20:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.248, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN21:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.883, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN22:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.589, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN23:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.171, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN24:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.943, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN25:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.822, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN26:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.425, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN27:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.026, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]



      elif pattern == ocn.OFFSETS_PATTERN28:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.467, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN29:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.612, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN30:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.616, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN31:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.719, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN32:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.889, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN33:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.256, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN34:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.758, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN35:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.616, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN36:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.527, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN37:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.594, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN38:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.144, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN39:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.551, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN40:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.303, sample_rate)
        return phase_list_low[3], phase_list_low[3], phase_list_high[4], phase_list_high[4]

      elif pattern == ocn.OFFSETS_PATTERN41:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.653, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN42:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.195, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN43:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.865, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN44:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.373, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]

      elif pattern == ocn.OFFSETS_PATTERN45:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.366, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN46:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.313, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN47:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.482, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN48:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.364, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]
      elif pattern == ocn.OFFSETS_PATTERN49:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.698, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


      elif pattern == ocn.OFFSETS_PATTERN50:
        phase_list_low, phase_list_high = self.calcPhaseAngles(frequency, 0.605, sample_rate)
        return phase_list_low[0], phase_list_low[0], phase_list_high[3], phase_list_high[3]


    except:
      self.debug.error_message("Exception in getOffsetsForPattern: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def calcPhaseAngles(self, frequency, standingwaveoffset, sample_rate):
    self.debug.info_message("calcPhaseAngles")

    try:
      pulse_length_in_samples = self.symbol_block_size / self.pulses_per_block

      def calcPhases(freq, offset_ratio):
        """ calculate the phases for the pulses at fixed pulse distance from pulse C """
        """ pulse separation is equivalent to pulse_length """
        offset_samples = pulse_length_in_samples * offset_ratio
        #wavelength_in_samples   = self.sample_rate / freq
        wavelength_in_samples   = sample_rate / freq
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
    self.debug.info_message("name: " + str(name))
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

     
  def startEncoder(self, values, text, mode, use_existing_txblocks, txblocks):
    self.debug.info_message("startEncoder")

    self.encoderRunning = True

    try:
      self.useProdMode()
      mode = values['combo_main_modem_prod_modes']

      self.setInitializationBlock(mode)

      self.debug.info_message("encoding text: " + str(text))

      if use_existing_txblocks == False:
        noise = values['btn_slider_awgn']
        text_num = values['combo_text_options'].split(':')[0]
        amplitude = values['slider_amplitude']
        carrier_separation_override = values['slider_carrier_separation']

        use_preset_message = self.form_gui.window['cb_use_preset_message'].get()
        if use_preset_message:
          txblocks = self.createTxBlocks(mode, values, noise, text_num, carrier_separation_override, amplitude, True, "")
        else:
          send_text = self.form_gui.window['ml_txrx_sendtext'].get()
          txblocks = self.createTxBlocks(mode, values, noise, text_num, carrier_separation_override, amplitude, False, send_text)

      #test1 = np.max(np.abs(txblocks))
      #test2 = txblocks * (2**15 - 1)
      txblocks = txblocks * (2**3 - 1) / np.max(np.abs(txblocks))
      txblocks = txblocks.astype(np.float32)

      txblocks = txblocks * self.getOutputGain()

      self.debug.info_message("getOutputGain() : " + str(self.getOutputGain()))

      self.resetDataQueue()

      use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
      if use_hifi_tx == False:
        local_sample_rate = self.sample_rate
      else:
        local_sample_rate = 48000

      #gc.disable()

      for location in range(0, len(txblocks) , local_sample_rate):
        block_item = txblocks[location:location+local_sample_rate]
        if len(block_item) < local_sample_rate:
          new_item =  np.zeros((local_sample_rate,), dtype = np.float32)
          new_item[0:len(block_item)] = block_item
          block_item = new_item
        self.pushDataQueue(block_item)

      self.debug.info_message("local_sample_rate: " + str(local_sample_rate))
      self.debug.info_message("self.sample_rate: " + str(self.sample_rate))

      self.initOutputStream(values, local_sample_rate)

    except:
      self.debug.error_message("Exception in startEncoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def startDecoder(self, mode, window, values):
    self.debug.info_message("startDecoder")

    self.decoderRunning = True

    try:
      self.useProdMode()
      mode = values['combo_main_modem_prod_modes']

      self.setInitializationBlock(mode)

      use_hifi_rx = self.form_gui.window['cb_enable_hifi_input_sampling'].get()
      if use_hifi_rx == False:
        local_sample_rate = self.sample_rate #8000
      else:
        local_sample_rate = 48000


      #self.initInputStream(self.sample_rate, window, values)
      self.initInputStream(local_sample_rate, window, values)

    except:
      self.debug.error_message("Exception in startDecoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def resetAll(self):
    self.debug.info_message("resetAll")
    self.opd = PersistentData(self)
    self.analysis     = OsmodAnalysis(self)
    self.detector     = OsmodDetector(self)
    self.simulator    = OsmodSimulator(self)
    self.interpolator = OsmodInterpolator(self)
    self.test         = OsmodTest(self, form_gui.window)
    self.fec          = OsmodFEC(self, form_gui.window)
    self.core_utils   = ModemCoreUtils(self)
    self.mod_2fsk8psk   = mod_2FSK8PSK(self)
    self.demod_2fsk8psk = demod_2FSK8PSK(self)
    self.prodparams   = OsmodProdParams(self)


  def stopEncoder(self):
    self.debug.info_message("stopEncoder")
    self.encoderRunning = False

    if self.form_gui.window['cb_continuous_decode'].get() == False:
      self.stopOutputStream()
    else:
      self.outStream.stop()



  def stopInputStream(self):
    self.debug.info_message("stopInputStream")
    # leave the in stream running
    if self.inStreamRunning == True:
      self.inStream.stop()
      self.inStreamRunning = False
      self.inStream.close()
      self.inStream = None
      #gc.enable()
      #gc.collect()


  def stopOutputStream(self):
    self.debug.info_message("stopOutputStream")
    try:
      if self.outStreamRunning == True:
        self.outStreamRunning = False
        self.outStream.stop()
        self.outStream.close()
        self.outStream = None
        #gc.enable()
        #gc.collect()
      #else:
      #  self.outStream.stop()

    except:
      self.debug.error_message("Exception in stopEncoder: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def stopDecoder(self):
    self.debug.info_message("stopDecoder")
    self.decoderRunning = False
    if self.form_gui.window['cb_continuous_decode'].get() == False:
      self.stopInputStream()

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

  def resetDataQueue(self):
    self.dataQueue = Queue()


  def get_sd_blocksize(self):
    #return self.symbol_block_size
    return self.sample_rate

  def set_sd_blocksize_tx(self):
    use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
    if use_hifi_tx:
      self.sd_blocksize_tx = self.sample_rate * 6
    else:
      self.sd_blocksize_tx = self.sample_rate


  def get_sd_blocksize_tx(self):
    return self.sd_blocksize_tx
    """
    use_hifi_tx = self.form_gui.window['cb_enable_hifi_output_sampling'].get()
    if use_hifi_tx:
      return self.sample_rate * 6
    else:
      return self.sample_rate
    """


  def set_sd_blocksize_rx(self):
    use_hifi_rx = self.form_gui.window['cb_enable_hifi_input_sampling'].get()
    if use_hifi_rx:
      self.sd_blocksize_rx = self.sample_rate * 6
    else:
      self.sd_blocksize_rx = self.sample_rate

  def get_sd_blocksize_rx(self):
    return self.sd_blocksize_rx


  def set_symbol_blocksize_rx(self):
    use_hifi_rx = self.form_gui.window['cb_enable_hifi_input_sampling'].get()
    if use_hifi_rx:
      self.symbol_blocksize_rx = self.symbol_block_size * 6
    else:
      self.symbol_blocksize_rx = self.symbol_block_size

  def get_symbol_blocksize_rx(self):
    return self.symbol_blocksize_rx



  def isDataQueueEmpty(self):
    return self.dataQueue.empty()


  def openConstellationPlot(self):    

    m = 16
    constellation = exp(1j * arange(0, 2 * pi, 2 * pi / m))

    plt.scatter(constellation.real, constellation.imag)

    plt.title('Constellation')
    plt.grid()
    plt.show()

    


  def createTxBlocks(self, mode, values, noise_mode, text_num, carrier_separation_override, amplitude, use_preset, custom_message):

    self.debug.info_message("createTxBlocks")

    try:
      self.startTimer('init')

      """ initialize the block"""
      #self.setInitializationBlock(mode)
      self.setInitializationBlockSR(mode, self.getTxSampleRate(), self.getTxSymbolBlockSize())

      """ figure out the carrier frequencies"""
      #center_frequency = values['slider_frequency']

      #frequency = self.calcCarrierFrequencies(center_frequency, carrier_separation_override)
      frequency = self.calcCarrierFrequenciesSR(self.center_frequency, carrier_separation_override, self.getTxSampleRate())


      self.debug.info_message("center frequency: " + str(self.center_frequency))
      self.debug.info_message("carrier frequencies: " + str(frequency))

      """ convert text to bits"""
      text_examples = [0] * 17
      text_examples[0]  = " cq wh6ggo "
      text_examples[1]  = " cqcqcqcqcqcq wh6ggo "
      text_examples[2]  = " cqcqcqcqcqcqcqcqcqcqcq wh6ggo "
      text_examples[3]  = " peter piper picked a peck of pickled peppercorn "
      text_examples[4]  = "jack be nimble jack be quick jack jump over the candlestick"
      text_examples[5]  = "row row row your boat gently down the stream merrily merrily merrily merrily life is but a dream"
      text_examples[6]  = "hickory dickory dock the mouse ran up the clock the clock struch one the mouse ran down hickory dicory dock"
      text_examples[7]  = "its raining its pouring the old man is snoring he bumped his head and went to bed and he couldnt get up in the morning"
      text_examples[8]  = "jack and jill went up the hill to fetch a pail of water jack fell down and broke his crown and jill came tumbling after"
      text_examples[9]  = "humpty dumpty dat on a wall humpty dumpty had a great fall all the kings forses and all the kings men coudnt put humpty together again"
      text_examples[10]  = "a wise old owl sat in an oak the more he heard the less he spoke the less he spoke the more he heard why arent we all like that wise old bird"
      text_examples[11]  = "hey diddle diddle the cat and the fiddle the cow jumped over the moon the little dog laughed to see such fun and the dish ran away with the spoon"
      text_examples[12]  = "baa baa black sheep have you any wool yes sir yes sir three bags full one for the master and one for the dame and one for the little boy who lives down the lane"
      text_examples[13] = "twinkle twinkle little bat how i wonder what youre at up above the world you fly like a tea tray in the sky twinkle twinkle little bat how i wonder what youre at"
      text_examples[14] = "i can read on a boat i can read with a goat i can read on a train i can read in the rain i can read with a fox i can read in a box i can read with a mouse i can read in a house i can read here or there i can read anywhere"
      text_examples[15] = "the queen of hearts she made some tarts all on a summers day the knave of hearts he stole the tarts and took them clean away the king of hearts called for the tarts and beat the knave full sore the knave of hearts brought back the tarts and vowed hed steal no more"
      text_examples[16] = "The Queen Of Hearts SHE MADE SOME TARTS all on a summer's day. The Knave Of Hearts HE STOLE THE TARTS and took them clean away. The King Of Hearts CALLED FOR THE TARTS and beat the knave full sore. The Knave Of Hearts BROUGTH BACK THE TARTS and vowed he'd steal no more."

      if use_preset:
        message_text = text_examples[int(text_num)]
      else:
        #custom_message = custom_message.lower()
        #message_text = "                                                e"
        #message_text = custom_message +  message_text[len(custom_message):]    #   [0:len(custom_message)] = custom_message
        message_text = custom_message 


      self.debug.info_message("message_text: " + str(message_text))


      max_message_length = int(self.form_gui.window['combo_max_message_length'].get())
      truncate_to_max_msglength = self.form_gui.window['cb_truncate_to_max_msglength'].get()

      """ add the callsign to the start of the message """
      message_text = self.modulation_object.addCallsignSOM_WithColon(message_text)

      """ translate ASCII to base 64 (excluding rotation sequence and padding character)"""
      self.debug.info_message("translating text: " + str(message_text))
      message_text = self.modulation_object.translateOutbound(message_text)

      """ insert CRC codes to protect message fragments """
      is_crc_enabled = self.form_gui.window['cb_enable_crc'].get()
      if is_crc_enabled:
        message_text = self.modulation_object.protectMessage(message_text, truncate_to_max_msglength, 8, max_message_length)


      """ add start sequence character and trailing space """
      if self.start_seq == '2_of_3':
        text = '   ' + message_text + ' '
      elif self.start_seq == '2_of_4' or self.start_seq == '3_of_4':
        text = '    ' + message_text + ' '
      elif self.start_seq == '2_of_5' or self.start_seq == '3_of_5' or self.start_seq == '4_of_5':
        text = '     ' + message_text + ' '
      elif self.start_seq == '2_of_6':
        text = '      ' + message_text + ' '
      elif self.start_seq == '2_of_7':
        text = '       ' + message_text + ' '
      elif self.start_seq == '2_of_8':
        text = '        ' + message_text + ' '
      else:
        text = '   ' + message_text + ' '


      #tx_message_length = len(text)
      #self.form_gui.window['combo_max_message_length'].update(value = str(tx_message_length))


      #max_message_length = int(self.form_gui.window['combo_max_message_length'].get())
      #truncate_to_max_msglength = self.form_gui.window['cb_truncate_to_max_msglength'].get()
      #if truncate_to_max_msglength:
      #  text = text[:max_message_length]

      if truncate_to_max_msglength:
        text_len = len(text)
        if text_len > max_message_length:
          text = text[:max_message_length]
        elif text_len <  max_message_length:
          text = text + (' ' * (max_message_length - text_len))

      self.debug.info_message("encoding text: " + str(text))

      #self.debug.info_message("encoding text: " + str(text))

      bit_groups, sent_bitstring, binary_array_pre_fec = self.text_encoder(text)
      #data2 = self.modulation_object.modulate(frequency, bit_groups)
      data2 = self.modulation_object.modulate(frequency, bit_groups, self.getTxSampleRate(), self.getTxSymbolBlockSize())

      """ filter the output signal """
      data2 = self.modulation_object.apply_filterSR(data2, self.tx_filter, self.center_frequency, self.getTxSampleRate())
      data2 = self.modulation_object.apply_filterSR(data2, self.tx_filter2, self.center_frequency, self.getTxSampleRate())


      """ write to file """
      self.debug.info_message("size of signal data: " + str(len(data2)))
      #self.modulation_object.writeFileWav(mode + ".wav", data2)
      #self.modulation_object.writeFileWavSR(mode + ".wav", data2, self.getTxSampleRate())
      self.modulation_object.writeFileWavSR(mode + ".wav", data2, self.getTxSampleRate())

      """ read file """
      use_audio_sample = self.form_gui.window['cb_test_routine_use_audio_sample'].get()
      if use_audio_sample:
        audio_sample_name = self.form_gui.window['combo_audio_sample_name'].get()
        audio_array = self.modulation_object.readFileWav(audio_sample_name) 
      else:
        audio_array = self.modulation_object.readFileWav(mode + ".wav") 

      self.debug.info_message("audio data type: " + str(audio_array.dtype))
      self.debug.info_message("demodulating")
      total_audio_length = len(audio_array)


      """ normalize magnitude """
      #audio_array = audio_array * (2**3 - 1) / np.max(np.abs(audio_array[8000:-8000]))


      """ add noise for testing..."""
      noise_free_signal = audio_array*0.00001 * float(amplitude)   

      self.debug.info_message("noise mode: " + str(noise_mode))
      value = float(noise_mode)

      audio_array = noise_free_signal
      if self.form_gui.window['cb_enable_awgn'].get():
        audio_array = self.modulation_object.addAWGN(audio_array, value, frequency)
      if self.form_gui.window['cb_enable_timing_noise'].get():
        audio_array = self.modulation_object.addTimingNoise(audio_array)
      if self.form_gui.window['cb_enable_phase_noise'].get():
        audio_array = self.modulation_object.addPhaseNoise2(audio_array)

      self.debug.info_message("size of noise data: " + str(len(audio_array)))
      #self.modulation_object.writeFileWavSR(mode + "_with_noise.wav", audio_array, self.getTxSampleRate())
      self.modulation_object.writeFileWavSR(mode + "_with_noise.wav", audio_array, self.getTxSampleRate())

      audio_array_with_unfiltered_noise = audio_array.copy()




      #return noise_free_signal * output_signal_gain
      return noise_free_signal

    except:
      self.debug.error_message("Exception in createTxBlocks: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def osmod_net_main(self):
    net = None
    debug = db.Debug(cn.DEBUG_INFO)

    """
    if (platform.system() == 'Windows'):
      appdata_folder = os.getenv('LOCALAPPDATA') 
      hrrm_appdata_folder = appdata_folder + '\HRRM'
      if(not os.path.exists(hrrm_appdata_folder)):
        os.chdir(appdata_folder)
        os.mkdir('HRRM')
        os.chdir(hrrm_appdata_folder)
        os.mkdir('received_images')
        os.mkdir('received_files')
        os.mkdir('hrrm_files')
      else:
        os.chdir(hrrm_appdata_folder)
    else:
      appdata_folder = os.getenv('HOME') 
      hrrm_appdata_folder = appdata_folder + '/.HRRM'
      if(not os.path.exists(hrrm_appdata_folder)):
        os.chdir(appdata_folder)
        os.mkdir('.HRRM')
        os.chdir(hrrm_appdata_folder)
        os.mkdir('received_images')
        os.mkdir('received_files')
        os.mkdir('hrrm_files')
      else:
        os.chdir(hrrm_appdata_folder)
    """

    """
    if nothing is specified for edition, profile string defaults to the day of the week in local time
    edition=day_zulu      set the profile string to today based on zulu time
    edition=time_of_day   set profile to morning, noon, afternoon, evening, nighttime
    edition=time_of_day_zulu   set profile to morning, noon, afternoon, evening, nighttime  based on zulu time
    interface = netcontrol
    interface = participant
    simulate
    edit
    combo_tks =
    combo_aloha =
    debug_level =
    """

    """ set the default values for command line parameters"""
    operating_mode  = cn.NETCONTROL
    #operating_mode  = cn.PARTICIPANT

    #simulation_mode = False
    simulation_mode = True
    edit_mode       = False
    combo_list_1    = 'Report,Good Report,Great Report,Signal Report,Great Question,Good Idea,Good Comment'.split(',')
    combo_list_2    = 'Great Evening,Good Evening,Great Rest of the Day,Good morning'.split(',')
    net_data_file   = "osmod_net_save_data.txt"


    client_read_details = True
    group = ""
    frequency = ""
    counter_value = 200
    show_counter=True
    delay_send = 25
    freq_from_osmod=False
    offsets_list = '1337,700,870,1140,1210,750,920,1190,1260,800,970,1240'
    update_freq_on_qsy = False
    from_plan = True
    main_offset = 1000
    visuals = 'background:LightGray,main:SeaGreen1,side:LightBlue1,flash1:red,flash2:blue'
    visuals = 'background:gray,main:turquoise1,side:DarkOliveGreen1,flash1:red2,flash2:green1'
    override_visuals = False
    side_main_offset_boundary = 700
    view = osmod_net_gui.MainNetWindow()
    js=view.readDictFromFile(net_data_file)
    self.osmod_net_view = view

    (opts, args) = getopt.getopt(sys.argv[1:], "h:i:n:p:t:a:e:s:r:g:f:c:d:o:b:v:m:u",
      ["help", "interface=", "net_file=", "profile=", "combo_tks=", "combo_aloha=", "edit", "simulate", "client_read_details", "group=", "frequency=", "counter=", "delay=", "offsets=", "boundary=", "visual=", "main_offset=", "update_freq_on_qsy"])
    rosterFile, macroFile = None, None
    for option, argval in opts:
      if (option in ("-h", "--help")):
        debug.info_message("main. usage")
        usage()

      elif (option in ("-i", "--interface")):
        debug.info_message("interface = " + argval)
        if(argval == "netcontrol"):
          operating_mode = cn.NETCONTROL
        else:  
          operating_mode = cn.PARTICIPANT
			
      elif (option in ("-n", "--net_file")):
        debug.info_message("net file = " + argval)
        net_data_file = argval
        
      elif (option in ("-p", "--profile")):
        debug.info_message("profile = " + argval)

      elif (option in ("-t", "--combo_tks")):
        debug.info_message("combo_tks = " + argval)
        combo_list_1    = argval.split(',')
        
      elif (option in ("-a", "--combo_aloha")):
        debug.info_message("combo_aloha = " + argval)
        combo_list_2    = argval.split(',')
        
      elif (option in ("-e", "--edit")):
        debug.info_message("edit mode")
        edit_mode = True

      elif (option in ("-s", "--simulate")):
        debug.info_message("simulate mode")
        simulation_mode = True

      elif (option in ("-g", "--group")):
        debug.info_message("group = " + argval)
        group = argval

      elif (option in ("-o", "--offsets")):
        debug.info_message("offsets = " + argval)
        offsets_list = argval

      elif (option in ("-b", "--boundary")):
        debug.info_message("boundary = " + argval)
        side_main_offset_boundary = int(argval)

      elif (option in ("-v", "--visual")):
        override_visuals = True
        debug.info_message("visual = " + argval)
        visuals = argval

      elif (option in ("-u", "--update_freq_on_qsy")):
        debug.info_message("update frequency field on qsy ")
        update_freq_on_qsy = True

      elif (option in ("-f", "--frequency")):
        debug.info_message("frequency = " + argval)
        frequency = argval

      elif (option in ("-r", "--client_read_details")):
        debug.info_message("read client details from file")
        client_read_details = True

      elif (option in ("-d", "--delay")):
        debug.info_message("set delay send")
        delay_send = int(argval)

      elif (option in ("-m", "--main_offset")):
        debug.info_message("set main offset")
        if(argval == "from_file"):
          main_offset = js.get("params").get("MainOffset")
          from_plan = False
        elif(argval == "from_plan"):
          from_plan = True
        else:
          main_offset = int(argval)
          from_plan = False

      elif (option in ("-c", "--counter")):
        debug.info_message("set counter value: "+ argval)
        if(argval == "off"):
          show_counter=False
        else:
          counter_value = int(argval)


    if(from_plan == True):
      main_offset = int(offsets_list.split(",")[0])

    if(override_visuals == False):
      if(operating_mode == cn.NETCONTROL):
        visuals = 'background:gray,main:turquoise1,side:DarkOliveGreen1,flash1:red2,flash2:green1'
      elif(operating_mode == cn.PARTICIPANT):
        visuals = 'background:indigo,main:SeaGreen1,side:LightBlue1,flash1:red,flash2:blue'
 
    view.setDelayValue(delay_send)



    self.osmod_net_layout = view.createClientWindow(js, operating_mode, simulation_mode, edit_mode, combo_list_1, combo_list_2, client_read_details, group, frequency, counter_value, show_counter, visuals, offsets_list, main_offset)
    #window = view.createClientWindow(js, operating_mode, simulation_mode, edit_mode, combo_list_1, combo_list_2, client_read_details, group, frequency, counter_value, show_counter, visuals, offsets_list, main_offset)
    window = None


    net = OSMOD_Net(debug, view, window, self)
    self.osmod_net = net
    view.osmod_net = net

    #view.window = window

    net.setManualGroup(group)
    net.setManualFrequency(frequency)
    net.setFreqFromOSMOD(freq_from_osmod)
    net.setSideMainOffsetBoundary(side_main_offset_boundary)
    net.setOffsetsList(offsets_list)
    net.setUpdateFreqOnQsy(update_freq_on_qsy)

    net.timeout = counter_value
    net.max_timeout = counter_value
   
    """ set the corresponding variables"""
    flashstate = js.get("params").get("FlashBtn")
    if(flashstate):
      net.setFlashingState(True)

    autocheckin = js.get("params").get("AutoCheckin")
    if(autocheckin):
      net.setAutoCheckin(True)

    net.setOperatingMode(operating_mode)

    """ only reload the roster data for net control view else create empty roster """	  
    if(operating_mode == cn.NETCONTROL):
      net.roster = js.get("roster")
    elif(operating_mode == cn.PARTICIPANT):
      net.roster = []

    """ now add the call sign lookups data object  """
    net.setKnownCalls(js.get("callsigns") )
    net.setNcsData(js)

    """ create the main gui controls event handler """
    dispatcher = osmod_net_events.ControlsProc(view, net, window)
    self.osmod_net_dispatcher = dispatcher

    """ create a separate thread to handle incoming messages """

    #osmod_client.connect(server)
    #t1 = threading.Thread(target=osmod_client.run, args=())
    #t1.start()
    #osmod_client.setCallback(net.my_new_callback)

    error_displayed = False

    """
    for x in range(10):
      if(osmod_client.isConnected()==False ):
        if(error_displayed == False):
          error_displayed = True
    if(osmod_client.isConnected()==True ):
      net.getStationCall()
      net.getDialAndOffset()
      view.run(osmod_client, net, dispatcher)
    else:
      osmod_client.stopThreads()
    """



  #""" osmod net mthods..."""
  #def sendMsg(self, *args, **kwargs):
  #def sendMsg(self, msg_type, message):
  #  self.debug.info_message("osmod_main::sendMsg")

  def sendMsg(self, *args, **kwargs):
    sys.stdout.write("osmod_main::sendMsg\n")
    self.debug.info_message("args: " +str(args))
    self.debug.info_message("kwargs: " +str(kwargs))

    #station_callsign = self.osmod_net.getStationCallSign()
    station_callsign = self.form_gui.window['input_ncs'].get()

    if True: #self.connected:
      params = kwargs.get('params', {})
      if '_ID' not in params:
        params['_ID'] = '{}'.format(int(time.time()*1000))
        kwargs['params'] = params
      message = self.to_message(*args, **kwargs)
      try:
        """ remember to send the newline at the end :) """
        self.debug.info_message("sending message: " +str(message))
        #self.sock.send((message + '\n').encode()) 

        #kernel_action = ocn.KERNEL_TX_NOW
        #self.sonic.pushKernelQueue(kernel_action)
        #self.sonic.send(self.form_gui.window, None, self.form_gui)


        dict_obj = json.loads(message)
        self.debug.info_message("dict_obj: " +str(dict_obj))
        if dict_obj['type'] == 'TX.SEND_MESSAGE':
          self.debug.info_message("TX.SEND_MESSAGE: " +str(message))
          #message_text = "        " + dict_obj['value']
          #message_text = station_callsign +": " + dict_obj['value']
          message_text = ": " + dict_obj['value']

          self.form_gui.window['ml_txrx_sendtext'].update(message_text)
          self.form_gui.window['cb_use_preset_message'].update(False)
          tx_message_length = self.getTxMessageLength(message_text)

          #self.form_gui.window['combo_max_message_length'].update(value = str(tx_message_length))
          self.form_gui.window['cb_truncate_to_max_msglength'].update(False)

          kernel_action = ocn.KERNEL_TXRX_NOW
          self.sonic.pushKernelQueue(kernel_action)
          self.sonic.send_threaded(self.form_gui.window, None, self.form_gui)
        if dict_obj['type'] == 'TX.SET_TEXT':
          self.debug.info_message("TX.SET_TEXT: " +str(message))

      except:
        sys.stdout.write("EXCEPT IN sendMsg\n")
        #self.close()



  def from_message(self, content):
    try:
      return json.loads(content)
    except ValueError:
      return {}

  def to_message(self, typ, value='', params=None):
    if params is None:
      params = {}
    return json.dumps({'type': typ, 'value': value, 'params': params})



  def stripEndOfMessage(self, message):
    sys.stdout.write("osmod_main::stripEndOfMessage\n")
    self.debug.info_message("message: " +str(message))

    retstring = ''
    try:    
      eom = u'♢'.encode('utf-8')
      retstring = message.split(eom, 1)[0]
    except:    
      sys.stdout.write("EXCEPTION\n")

    return retstring



  """
  get the contents of a named parameter from the return string
  """
  def getNetParam(self, dict_obj, paramname):
    sys.stdout.write("osmod_main::getNetParam\n")
    subdict  = dict_obj.get('params')
    param_value = subdict.get(paramname)
    return(str(param_value))



  """
  return the value of json string item
  """
  def getValue(self, dict_obj, objname):
    sys.stdout.write("osmod_main::getValue\n")
    value  = dict_obj.get(objname)
    return (value.encode('utf-8'))

  """
  test if message contains missing frame unicode character(s)
  """
  def areFramesMissing(self, message):
    sys.stdout.write("osmod_main::areFramesMissing\n")
    
    frame_missing = u'……'.encode('utf-8')

    count = start = 0
    flag = True
    while flag:
      a = message.find(frame_missing, start)
      if a == -1:
        flag = False
      else:
        count += 1
        start = a+1
    return (count)


  """
  test if message contains text
  unicode encode necessary for correct functioning otherwise throws unicode exception
  """
  def isTextInMessage(self, text, message):
    sys.stdout.write("osmod_main::isTextInMessage\n")
    try:
      newtext = text.encode('utf-8')
      if newtext in message.encode('utf-8'):
        return (1)
      else:
        return (0)
    except:    
      sys.stdout.write("EXCEPTION\n")



  def osmodNetCallback(self, message):
    sys.stdout.write("osmodNetCallback\n")
    try:
      formatted_nssage = '{"params":{"DIAL":7078000,"FREQ":7080341,"OFFSET":890,"SNR":-5,"SPEED":4,"TDRIFT":-0.5,"UTC":1654371086715,"_ID":-1},"type":"RX.DIRECTED","value":"' + message + '"}\n'

      self.osmod_net.my_new_callback(formatted_nssage, cn.RCV, "NOT USED", "NOT USED")
    except:
      self.debug.error_message("Exception in osmodNetCallback: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def displayReceivedMessage(self, message, update, erase):
    self.debug.info_message("displayReceivedMessage")
    try:
        """ erase existing display info """
        bypass_display_during_test = self.form_gui.window['cb_bypass_display_during_test'].get()

        if erase and bypass_display_during_test == False:
          self.form_gui.window['ml_txrx_recvtext'].update('')

        message_struct = self.processFragmentedMessage(message)
        fragments = message_struct['fragments']
        pass_fail = message_struct['pass_fail']

        if len(fragments) > 1:
          with_crc = True
          if pass_fail[0] == 'p':
            sender_callsign = fragments[0][4:].split(' ')[0].split(':')[0]
            sender_callsign = sender_callsign.upper()
            self.debug.info_message("sender_callsign: " + str(sender_callsign))
          else:
            sender_callsign = ''
        else:
          with_crc = False
          sender_callsign = ''

        reconstituted_message = message_struct['reconstituted_message']
        original_message = self.modulation_object.translateInbound(reconstituted_message)
        #original_message = reconstituted_message #self.modulation_object.translateInbound(message_struct['reconstituted_message'])

        xref = self.createMessageXref(reconstituted_message)

        last_position = 0
        #for count in range(0, len(xref)):
        for count in range(0, len(fragments)):
          if with_crc:
            part_message = original_message[last_position:xref[count]]
          else:
            part_message = original_message


          if bypass_display_during_test == False:
            if pass_fail[count] == 'p':
              self.form_gui.window['ml_txrx_recvtext'].print(str(part_message), end="", text_color='green', background_color = 'white')
            elif pass_fail[count] == 'f':
              self.form_gui.window['ml_txrx_recvtext'].print(str(part_message), end="", text_color='red', background_color = 'white')
            elif pass_fail[count] == 'u':
              self.form_gui.window['ml_txrx_recvtext'].print(str(part_message), end="", text_color='purple', background_color = 'white')

          #self.form_gui.window['ml_txrx_recvtext'].print(str(part_message), end="", text_color='orange', background_color = 'white')
          last_position = xref[count]

        if bypass_display_during_test == False:
          self.form_gui.window['ml_txrx_recvtext'].print("\n\n", end="", text_color='orange', background_color = 'white')


        #self.form_gui.window['ml_txrx_recvtext'].print(str(original_message) + "\n\n", end="", text_color='blue', background_color = 'white')



        for frag_count in range(0, len(fragments)):
          if with_crc:
            temp_fragment = fragments[frag_count]
            fragment = temp_fragment[2:len(temp_fragment)-2]
          else:
            fragment = fragments[frag_count]

          if bypass_display_during_test == False:
            if pass_fail[frag_count] == 'p':
              self.form_gui.window['ml_txrx_recvtext'].print(str(fragment), end="", text_color='green', background_color = 'white')
            elif pass_fail[frag_count] == 'f':
              self.form_gui.window['ml_txrx_recvtext'].print(str(fragment), end="", text_color='red', background_color = 'white')
            elif pass_fail[frag_count] == 'u':
              self.form_gui.window['ml_txrx_recvtext'].print(str(fragment), end="", text_color='purple', background_color = 'white')

        if bypass_display_during_test == False:
          self.form_gui.window['ml_txrx_recvtext'].print("\n", end="", text_color='orange', background_color = 'white')

        if update:
          timestamp = self.modulation_object.appendTableRow(original_message, sender_callsign)

          if self.form_gui.window['cb_use_prod_modes'].get() == True:
            mode = self.form_gui.window['combo_main_modem_prod_modes'].get()
          else:
            mode = self.form_gui.window['combo_main_modem_modes'].get()
          callsign = "FIXME_TEST"
          #mode = "FIXME_MODE"


          #self.storeMessageInMemory(self.getCenterFrequency(), mode, callsign, timestamp, original_message)
          self.storeMessageInMemory(self.getCenterFrequency(), mode, sender_callsign, timestamp, message)


          net_enabled = self.form_gui.window['cb_enable_osmod_net'].get()
          #net_enabled = False:
          if net_enabled:
            self.osmodNetCallback(original_message)


    except:
      self.debug.error_message("Exception in displayReceivedMessage: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  """ cross reference to map crc verified sections of text to translated text """
  def createMessageXref(self, untranslated_message):
    self.debug.info_message("createMessageXref")
    try:
      xref = []

      self.debug.info_message("untranslated_message: " + str(untranslated_message))

      two_chars = ['==', '=a', '=b', '=c', '=d', '=f', '=g', '=p', '=q', '=r', '=s', '=t', '=u', '=v', '//']

      crc_fragment_size = int(self.form_gui.window['in_crc_fragment_size'].get())

      rolling_counter = 0
      tranlated_msg_location = 0
      untranlated_msg_location = 0
      untranslated_msg_len = len(untranslated_message)
      while untranlated_msg_location < untranslated_msg_len:
        if untranslated_message[untranlated_msg_location] == '/':
          if untranlated_msg_location + 1 < untranslated_msg_len and untranslated_message[untranlated_msg_location + 1].isdigit():
            if untranlated_msg_location + 2 < untranslated_msg_len and untranslated_message[untranlated_msg_location + 2].isdigit():
              if untranlated_msg_location + 3 < untranslated_msg_len and untranslated_message[untranlated_msg_location + 3].isdigit():
                if untranlated_msg_location + 4 < untranslated_msg_len and untranslated_message[untranlated_msg_location + 4].isdigit():
                  # best guess
                  rle_length = int(untranslated_message[untranlated_msg_location + 1:untranlated_msg_location + 4 ])
                  tranlated_msg_location = tranlated_msg_location + 5 + rle_length
                  untranlated_msg_location = untranlated_msg_location + 5
                else:
                  rle_length = int(untranslated_message[untranlated_msg_location + 1:untranlated_msg_location + 4 ])
                  tranlated_msg_location = tranlated_msg_location + 5 + rle_length
                  untranlated_msg_location = untranlated_msg_location + 5
              else:
                rle_length = int(untranslated_message[untranlated_msg_location + 1:untranlated_msg_location + 3 ])
                tranlated_msg_location = tranlated_msg_location + 4 + rle_length
                untranlated_msg_location = untranlated_msg_location + 4
            else:
              rle_length = int(untranslated_message[untranlated_msg_location + 1])
              tranlated_msg_location = tranlated_msg_location + 3 + rle_length
              untranlated_msg_location = untranlated_msg_location + 3
          else:
            tranlated_msg_location = tranlated_msg_location + 1
            untranlated_msg_location = untranlated_msg_location + 2
        elif untranslated_message[untranlated_msg_location] == '=':
          self.debug.info_message("located =")
          #self.debug.info_message("located: " + str(untranslated_message[untranlated_msg_location:untranlated_msg_location + 2]))
          if untranlated_msg_location + 2 < untranslated_msg_len and untranslated_message[untranlated_msg_location:untranlated_msg_location + 2] in two_chars:
            self.debug.info_message("located = two_chars")
            tranlated_msg_location = tranlated_msg_location + 1
            untranlated_msg_location = untranlated_msg_location + 2
          else:
            self.debug.info_message("located = no_chars")
            tranlated_msg_location = tranlated_msg_location + 0
            untranlated_msg_location = untranlated_msg_location + 2
        else:
          tranlated_msg_location = tranlated_msg_location + 1
          untranlated_msg_location = untranlated_msg_location + 1

        if untranlated_msg_location - rolling_counter >= crc_fragment_size:
          xref.append(tranlated_msg_location)
          rolling_counter = rolling_counter + crc_fragment_size

      xref.append(tranlated_msg_location)

      self.debug.info_message("xref: " + str(xref))

      return xref

    except:
      self.debug.error_message("Exception in createMessageXref: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def processFragmentedMessage(self, message):
    self.debug.info_message("processFragmentedMessage")
    try:
      num_rotation_chars = 8

      crc_fragment_size = int(self.form_gui.window['in_crc_fragment_size'].get())
      total_crc_fragment_size = crc_fragment_size + 4

      if '|' not in message.rstrip('|'):
        self.debug.info_message("no crc delimeter found in message")
        return {"num_fragments": 1, "fragments": [message], "pass_fail": ['u'], "reconstituted_message": message}

      sequence_identifier = "0123456789abcdefghijklmnopqrstuvwxyz"
      align_counter = {}
      delimiter_indexes = []

      align_counter["max"] = 0
      align_counter["max_index"] = 0
      for index in range(0, len(message)):
        if message[index] == '|':
          delimiter_indexes.append(index)

      for counter in range(0, len(delimiter_indexes)):
        location = delimiter_indexes[counter] % total_crc_fragment_size
        self.debug.info_message("location: " + str(location))
        if location in align_counter:
          align_counter[location] = align_counter[location] + 1
        else:
          align_counter[location] = 1

        if align_counter[location] > int(align_counter["max"]):
          align_counter["max"] = align_counter[location]
          align_counter["max_index"] = location
          self.debug.info_message("align_counter[max]: " + str(align_counter["max"]))
          self.debug.info_message("align_counter[max_index]: " + str(align_counter["max_index"]))

      offset = align_counter["max_index"]
      num_fragments = (len(message) - num_rotation_chars) // total_crc_fragment_size

      #remainder = (len(message) - num_rotation_chars) - (num_fragments * total_crc_fragment_size)
      temp_strings = message[num_rotation_chars + 2 + (num_fragments * total_crc_fragment_size): ].split('|',1)
      self.debug.info_message("temp_strings: " + str(temp_strings))
      remainder = len(temp_strings[0]) 
      self.debug.info_message("remainder: " + str(remainder))

      fragments = []
      pass_fail = []
      for counter in range(0, num_fragments):
        location = offset + (counter * total_crc_fragment_size)
        fragment = '|' + sequence_identifier[counter] + message[location + 2:location + total_crc_fragment_size]
        fragments.append(fragment)

        checksum = self.modulation_object.calcFragmentCRC(fragment[0:total_crc_fragment_size - 2])
        if checksum == fragment[total_crc_fragment_size - 2:total_crc_fragment_size]:
          pass_fail.append('p')
        else:
          pass_fail.append('f')

        self.debug.info_message("fragment: " + str(fragment))

      self.debug.info_message("processing remainder...")

      if remainder > 0:
        location = offset + (num_fragments * total_crc_fragment_size)
        #fragment = '|' + sequence_identifier[num_fragments] + message[location + 2:location + remainder]
        remainder_location = num_rotation_chars + 2 + (num_fragments * total_crc_fragment_size)
        fragment = '|' + sequence_identifier[num_fragments] + message[remainder_location:remainder_location + remainder]
        fragments.append(fragment)

        checksum = self.modulation_object.calcFragmentCRC(fragment[0:len(fragment) - 2])
        if checksum == fragment[len(fragment) - 2:len(fragment)]:
          pass_fail.append('p')
        else:
          pass_fail.append('f')

        self.debug.info_message("fragment: " + str(fragment))

      self.debug.info_message("align_counter: " + str(align_counter))
      self.debug.info_message("fragments: " + str(fragments))
      self.debug.info_message("pass_fail: " + str(pass_fail))

      reconstituted_message = ''
      for frag_count in range(0, len(fragments)):
        fragment = fragments[frag_count]
        reconstituted_message = reconstituted_message + fragment[2:len(fragment)-2]

      self.debug.info_message("reconstituted_message: " + str(reconstituted_message))

      return {"num_fragments": len(fragments), "fragments": fragments, "pass_fail": pass_fail, "raw_message": message, "reconstituted_message": reconstituted_message}

    except:
      self.debug.error_message("Exception in processFragmentedMessage: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def storeMessageInMemory(self, frequency, mode, callsign, timestamp, message):
    sys.stdout.write("storeMessageInMemory\n")
    
    self.debug.info_message("frequency: " + str(frequency))
    self.debug.info_message("callsign: " + str(callsign))

    try:
      msg_id = 1
      if "message_counter" not in self.all_messages:
        self.all_messages['message_counter'] = 1
        msg_id = 1
      else:
        max_message_number = self.all_messages['message_counter']
        self.all_messages['message_counter'] = max_message_number + 1
        msg_id = max_message_number + 1
 
      self.all_messages[str(msg_id)] = message
      self.all_messages[str(msg_id) + "_Timestamp"] = timestamp

      frequency_mode = str(frequency) + "_" + str(mode)
      if frequency_mode in self.messages_by_frequency:
        table_messages = self.messages_by_frequency[frequency_mode]
        table_messages.append(msg_id)
        self.messages_by_frequency[frequency_mode] = table_messages
      else:
        self.messages_by_frequency[frequency_mode] = [msg_id]

      if callsign in self.messages_by_callsign:
        table_messages = self.messages_by_callsign[callsign]
        table_messages.append(msg_id)
        #self.messages_by_callsign[str(callsign)] = self.messages_by_callsign[str(callsign)].append(str(msg_id))
        self.messages_by_callsign[callsign] = table_messages
      else:
        self.messages_by_callsign[callsign] = [msg_id]

      self.debug.info_message("all_messages: " + str(self.all_messages))
      self.debug.info_message("messages_by_frequency: " + str(self.messages_by_frequency))
      self.debug.info_message("messages_by_callsign: " + str(self.messages_by_callsign))

    except:
      self.debug.error_message("Exception in storeMessageInMemory: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def writeModemSettingsToFile(self, window, filename):
    self.debug.info_message("writeModemSettingsToFile")

    try:	  	  
      """ individual fields first """	  
      dict_data = { 'StationCallsign'    : window['in_station_callsign'].get().strip(),
                    'MainGroup'          : window['in_group'].get().strip(),
                    'LocatorGridSquare'  : window['in_locator_grid_square'].get().strip(),
                    'InputDevice'        : window['combo_main_modem_input_device'].get(),
                    'OutputDevice'       : window['combo_main_modem_output_device'].get(),
                    'ExtrapolateMode'    : window['combo_extrapolate_option'].get() }

      self.dict_data = dict_data

      with open(filename, 'w') as convert_file:
        convert_file.write(json.dumps(dict_data))

    except:
      self.debug.error_message("Exception in writeModemSettingsToFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return(dict_data)


  def readModemSettingsFromFile(self, window, filename):
    self.debug.info_message("readModemSettingsFromFile")
    try:
      """ set defaults """
      dict_data = { 'StationCallsign'    : '<YOUR CALLSIGN HERE>',
                    'MainGroup'          : '<GROUP NAME>',
                    'LocatorGridSquare'  : '<GRID SQUARE>',
                    'InputDevice'        : '',
                    'OutputDevice'       : '' ,
                    'ExtrapolateMode'    : 'Single' }

      with open(filename) as f:
        data = f.read()
  
      """ reconstructing the data as a dictionary """
      loaded_dict_data = json.loads(data)

      """ overwrite values if exist in file otherwise use default setting."""
      for param_name, param_value in loaded_dict_data.items():
        dict_data[param_name] = param_value

    except:
      self.debug.info_message("creating settings data")

    window['in_station_callsign'].update(dict_data['StationCallsign'])
    window['in_group'].update(dict_data['MainGroup']),
    window['in_locator_grid_square'].update(dict_data['LocatorGridSquare']),
    if dict_data['InputDevice'] != '':
      window['combo_main_modem_input_device'].update(dict_data['InputDevice']),
    if dict_data['OutputDevice'] != '':
      window['combo_main_modem_output_device'].update(dict_data['OutputDevice'])
    window['combo_extrapolate_option'].update(dict_data['ExtrapolateMode']),

    self.dict_data = dict_data   
    return(dict_data)



  def loadSolveData(self, mode, filename):
    self.debug.info_message("loadSolveData")

    try:

      def saveDictData(filename, dict_data):
        self.debug.info_message("saveDictData")

        try:
          self.solve_data_dict[mode] = dict_data
          with open(filename, 'w') as convert_file:
            convert_file.write(json.dumps(self.solve_data_dict))
        except:
          self.debug.error_message("Exception in saveDictData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


      def loadDictData(filename):
        self.debug.info_message("loadDictData")

        try:
          with open(filename) as f:
            data = f.read() 
          dict_data = json.loads(data)

          self.solve_data_dict = dict_data
          if mode in dict_data:
            return dict_data[mode]
          else:
            return {}

        except:
          self.debug.error_message("no file in loadDictData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
          return {}


      def format_string_fft(random_1, random_2):
        return '-' + str(random_1) + ',' + str(random_2) + ',-' + str(random_2) + ',' + str(random_1) , ''

      def unformat_string_fft(formatted_string):
        split_values = formatted_string.split('_')
        value_1 = abs(float(split_values[0]))
        value_2 = abs(float(split_values[1]))
        return value_1, value_2

      def unformat_string_standing_wave(formatted_string):
        self.debug.info_message("unformat_string_standing_wave")
        split_values = formatted_string.split('_')
        value_1 = split_values[0]                  # DATA_SW_PATTERN_TYPE
        value_2 = abs(float(split_values[1]))      # DATA_SW_LOCATION
        self.debug.info_message("value_1: " + str(value_1))
        self.debug.info_message("value_2: " + str(value_2))
        return value_1, value_2

      def format_string_standing_wave_from_char(random_1, random_2):
        self.debug.info_message("format_string_standing_wave")
        self.debug.info_message("random_1: " + str(random_1))
        self.debug.info_message("random_2: " + str(random_2))
        return str(random_1), str(random_2)

      def unformat_string_dcs(formatted_string):
        self.debug.info_message("unformat_string_dcs")
        value_1 = abs(float(formatted_string))
        return value_1, 0.0

      def format_string_dcs(random_1, random_2):
        self.debug.info_message("format_string_dcs")
        return random_1, random_2

      def unformat_string_rrc_alpha_t(formatted_string):
        self.debug.info_message("unformat_string_rrc_alpha_t")
        split_values = formatted_string.split('_')
        value_1 = abs(float(split_values[0]))
        value_2 = abs(float(split_values[1]))
        return value_1, value_2

      def format_string_rrc_alpha_t(random_1, random_2):
        self.debug.info_message("format_string_rrc_alpha_t")
        return random_1, random_2


      dict_best_so_far = loadDictData(filename)
      if "DATA_FFT_FILTER" in dict_best_so_far:
        value_1, value_2 = unformat_string_fft(dict_best_so_far["DATA_FFT_FILTER"])
        new_string_1, new_string_2 = format_string_fft(value_1, value_2)
        self.form_gui.window['in_fft_filter'].update(new_string_1)
        self.form_gui.window['cb_override_fft_filter'].update(True)
      if "DATA_FFT_INTERPOLATE" in dict_best_so_far:
        value_1, value_2 = unformat_string_fft(dict_best_so_far["DATA_FFT_INTERPOLATE"])
        new_string_1, new_string_2 = format_string_fft(value_1, value_2)
        self.form_gui.window['in_fft_interpolate'].update(new_string_1)
        self.form_gui.window['cb_override_fft_interpolate'].update(True)
      if "DATA_PATTERN" in dict_best_so_far:
        value_1, value_2 = unformat_string_standing_wave(dict_best_so_far["DATA_PATTERN"])
        new_string_1, new_string_2 = format_string_standing_wave_from_char(value_1, value_2)
        self.form_gui.window['combo_standingwave_pattern'].update(new_string_1)
        self.form_gui.window['in_standingwavelocation'].update(new_string_2)
        self.form_gui.window['cb_override_standingwaveoffsets'].update(True)
      if "DATA_RRC_ALPHA_T" in dict_best_so_far:
        value_1, value_2 = unformat_string_rrc_alpha_t(dict_best_so_far["DATA_RRC_ALPHA_T"])
        new_string_1, new_string_2 = format_string_rrc_alpha_t(value_1, value_2)
        self.form_gui.window['in_rrc_alpha'].update(new_string_1)
        self.form_gui.window['in_rrc_t'].update(new_string_2)
        self.form_gui.window['cb_override_rrc_alpha'].update(True)
        self.form_gui.window['cb_override_rrc_t'].update(True)
      if "DATA_DCS" in dict_best_so_far:
        value_1, value_2 = unformat_string_dcs(dict_best_so_far["DATA_DCS"])
        new_string_1, new_string_2 = format_string_dcs(value_1, value_2)
        self.form_gui.window['in_downconvertshift'].update(new_string_1)
        self.form_gui.window['cb_overridedownconvertshift'].update(True)

      return dict_best_so_far

    except:
      self.debug.error_message("Exception in loadSolveData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
