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
import ctypes

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

class OsmodProdParams(object):

  #mod_psk   = None
  #demod_psk = None

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)

  def __init__(self, osmod):  
  #def __init__(self):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod

    """ initialize the initialization blocks for the different modulations"""
    """ prod modes use simplified naming. full test name in the info comments"""
    self.prodmode_initialization_block = { 

        'LB28-51200-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '0.15625 characters per second, 0.9375 baud (bits per second). -31.7dB SNR approx',
                    'symbol_block_size'     : 51200,
                    'pulses_per_block'      : 512,   #1024,
                    #'fft_filter'            : (-0.5, 0.5, -0.5, 0.5),
                    #'fft_interpolate'       : (-1, 1, -1, 1),
                    #'fft_filter'            : (-1, 1, -1, 1),
                    #'fft_interpolate'       : (-0.75, 0.3, -0.3, 0.75),
                    #'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    #'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    'fft_filter'            : (-0.8, 0.8, -0.8, 0.8),
                    'fft_interpolate'       : (-0.8, 0.8, -0.8, 0.8),
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'D-E', 0.013),
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'E-E', 0.267),
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-B', 0.34),
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'B-E', 0.567),
                    #'I3_combine'            : ocn.INTRA_COMBINE_TYPE3,
                    #'downconvert_shift'     : 0.6,
                    #'parameters'            : (1700, 0.822, 0.997, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
        }, 

        'LB28-25600-I3E' :{ 
                    'inherit_from'          : 'LB28-25600-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-25600-I3' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : '0.3125 characters per second, 1.875 baud (bits per second). -30.6 dB SNR',
                    'symbol_block_size'     : 25600,
                    'pulses_per_block'      : 256,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                    #'fft_filter'           : (-1, 1, -1, 1),
                    #'fft_interpolate'      : (-1, 1, -1, 1),
                    #'fft_filter'           : (-1.5, 1.5, -1.5, 1.5),
                    #'fft_interpolate'      : (-1.5, 1.5, -1.5, 1.5),
                    'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    #'fft_filter'           : (-1.3, 1.3, -1.3, 1.3),
                    #'fft_interpolate'      : (-1.3, 1.3, -1.3, 1.3),
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.125, 19.014, 0], # available, low freq, hi freq

        }, 


        'LB28-12800-I3E' :{ 
                    'inherit_from'          : 'LB28-12800-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-12800-I3' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : '0.625 characters per second, 3.75 baud (bits per second). -28 dB SNR',
                    'symbol_block_size'     : 12800,
                    'pulses_per_block'      : 128,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,
                    #'downconvert_shift'     : 0.241,
                    #'downconvert_shift'     : 0.267,
                    'downconvert_shift'     : 0.374,
                    #'parameters'            : (1500, 0.392, 0.953, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.704, 0.982, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'fft_filter'           : (-1, 1, -1, 1),
                    'fft_interpolate'      : (-1, 1, -1, 1),

                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, 1381.875, 1418.653, 0], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.125, 18.653, 0], # available, low freq, hi freq

        }, 


        #'LB28-6400-I3-B' :{ 
        #            'inherit_from'        : 'LB28-I3-BASE',
        #            'info'                  : 'based on nearest test mode LB28-6400-128-2-15-I - 1.25 characters per second, 7.5 baud (bits per second)',
        #            'symbol_block_size'     : 6400,
        #            'pulses_per_block'      : 128,
        #            'symbol_wave_function'  : self.osmod.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,
        #}, 


        'LB28-6400-I3-DP' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'doppler_adjust'        : ocn.DOPPLER_ADJUST_ALL,
        }, 


        'LB28-6400-I3E' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-6400-I3' :{ 

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '1.25 characters per second, 7.5 baud (bits per second)',
                    'symbol_block_size'     : 6400,
                    'pulses_per_block'      : 64,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.5, 18.556, 250], # available, low freq relative center, hi freq relative center

                    #'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3E-FEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FEC',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3-FEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,

                    #'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                    #'I3_parameters'        : (0.99, 0.99, 2e-3, 'B-B', 0.943),
                    'extrapolate_seqlen'   : 8,
                    #'downconvert_shift'    : 0.535,
                    #'parameters'           : (1500, 0.763, 0.107, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

        }, 


        'LB28-3200-I3E' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3' :{ 

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '2.5 characters per second, 15 baud (bits per second)',
                    'symbol_block_size'     : 3200,
                    'pulses_per_block'      : 32,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.thirtyseconds_symbol_wave_function,
                    'fft_filter'            : (-4, 4, -4, 4),
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    #'extrapolate'           : 'yes',
                    'downconvert_shift'     : 0.422,
                    #'parameters'            : (1500, 0.939, 0.96, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-D', 0.312),
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, 1382.5, 1417.112, 0], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.5, 17.112, 0], # available, low freq, hi freq

        }, 


        """
        'LB28-SPECIAL' :{  # to be used for mode selector. this uses 125 * 64 as block size
                    'inherit_from'          : 'LB28-I3-BASE',
                    'pulses_per_block'     : 100,
                    #'symbol_block_size'    : 7360,
                    'symbol_block_size'    : 10000,
                    'fft_filter'           : (-1, 1, -1, 1),
                    'fft_interpolate'      : (-1, 1, -1, 1),
                    #'symbol_block_size'    : 8320,
                    'extrapolate'          : 'no', 
                    'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-C', 0.128),
                    'downconvert_shift'    : 0.13,
                    'parameters'           : (1500, 0.378, 0.909, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
        }, 
        """

        'LB28-1600-I3E-FEC' :{ 
                    'inherit_from'          : 'LB28-1600-I3-FEC',
                    'extrapolate'           : 'yes',

        }, 


        'LB28-1600-I3-FEC' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                    #'I3_parameters'        : (0.99, 0.99, 2e-3, 'E-E', 0.232),
                    #'I3_parameters'        : (0.99, 0.99, 2e-3, 'D-D', 0.003),
                    'I3_parameters'        : (0.99, 0.99, 2e-3, 'E-E', 0.115),

                    'extrapolate_seqlen'   : 8,
                    #'extrapolate'           : 'no',
                    'downconvert_shift'    : 0.535,
                    #'parameters'           : (1500, 0.099, 0.152, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'           : (1500, 0.982, 0.168, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

        }, 


        'LB28-1600-I3E' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-1600-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '5.0 characters per second, 30.0 baud (bits per second)',
                    'symbol_block_size'     : 1600,
                    'pulses_per_block'      : 16,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixteenths_symbol_wave_function,
                    'fft_filter'            : (-4, 4, -4, 4),
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-C', 0.029),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-E', 0.854),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-E', 0.96),
                    'downconvert_shift'     : 0.945,
                    'parameters'            : (1500, 0.734, 0.76, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'persistent_search'     : (1, 0.95, -0.005, "yes"), #hi range, lo range, inc, scan entire range
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -15, 14.224, 0], # available, low freq, hi freq
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),

        }, 


        'LB28-800-I3-FEC' :{ 
                    'inherit_from'         : 'LB28-800-I3',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),
                    #'downconvert_shift'    : 0.535,

                     #'I3_parameters'        : (0.99, 0.99, 2e-3, 'D-D', 0.003),
                     #'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-A', 0.216),
                     #'I3_parameters'        : (0.99, 0.99, 2e-3, 'E-E', 0.017),
                     'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-E', 0.205),
                    'extrapolate_seqlen'   : 8,
                    #'extrapolate'           : 'yes',
                    'extrapolate'           : 'no',
                    'downconvert_shift'    : 0.535,
                    'parameters'           : (1500, 0.982, 0.168, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'           : (1500, 0.428, 0.836, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'           : (1500, 0.093, 0.62, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

        }, 


        'LB28-800-I3-HFM' :{  # solve for lowest Pattern first, then solve for downconvert and separation together
                    'inherit_from'         : 'LB28-800-I3',
                    'info'                  : 'Holographic Frequency Multiplexing  10.0 characters per second, 60.0 baud (bits per second) per hologram',
                    'FDM'                  : "yes",
                    'downconvert_shift'    : 0.019,
                    #'downconvert_shift'    : 0.04,
                    'FDM_parameters'       : [2, 95.768], #frequency division mutiplexing: multiplier, separation
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 200, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 200, 5, 50),
                    'tx_filter'             : (ocn.FILTER_NONE, ocn.FILTER_NONE, 200, 5, 50),
                    'rx_filter'             : (ocn.FILTER_NONE, ocn.FILTER_NONE, 200, 5, 50),
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'B-D', 0.178), 
        }, 


        'LB28-800-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.eighths_symbol_wave_function,
                    'symbol_block_size'     : 800,
                    'pulses_per_block'      : 8,
                    'fft_filter'            : (-20, 16, -16, 20),
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),

                    'persistent_search'     : (1, 0.95, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-A', 0.003), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-C', 0.85), 
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'B-B', 0.461), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'D-D', 0.361), 

                    'parameters'            : (1500, 0.115, 0.028, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

        }, 


        'LB28-400-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '10.0 characters per second, 60.0 baud (bits per second)',
                    #'pulses_per_block'      : 8,
                    'pulses_per_block'      : 6,
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    'fft_filter'            : (-20, 16, -16, 20),
                    #'fft_filter'            : (-25, 14, -14, 25),
                    #'fft_filter'            : (-18, 18, -18, 18),
                    #'fft_filter'            : (-20, 20, -20, 20),
                    #'fft_filter'            : (-25, 25, -25, 25),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    'persistent_search'     : (1, 0.95, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.93, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                    'symbol_block_size'     : 400,
                    'downconvert_shift'     : 0.514,
                    #'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-C', 0.701), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-D', 0.418), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-A', 0.008), 
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'C-C', 0.387), 
                    'extrapolate'           : 'no', #no rotation tables yet!!!
                    #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                    'parameters'            : (1500, 0.661, 0.063, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'I3_combine'           : ocn.INTRA_COMBINE_TYPE6,

        }, 



        'LB28-I3-BASE' :{ 


                    'encoder_callback'      : self.osmod.mod_2fsk8psk.encoder_8psk_callback,
                    'decoder_callback'      : self.osmod.demod_2fsk8psk.demodulate_2fsk_8psk,
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTriplet,
                    'text_decoder'          : self.osmod.demod_2fsk8psk.displayTextResults,

                    'mode_selector'         : ocn.OSMOD_MODEM_8FSK,
                    'sample_rate'           : 8000,
                    'parameters'            : (1500, 0.822, 0.997, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'carrier_separation'    : 37,
                    'num_carriers'          : 2,
                    'detector_function'     : 'mode',
                    'symbols_per_block'     : 1,  # per carrier!
                    'phase_encoding'        : ocn.PHASE_INTRA_TRIPLE,
                    'doppler_adjust'        : ocn.DOPPLER_ADJUST_NONE,
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 48, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 48, 5, 50),

                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                    'modulation_object'     : self.osmod.mod_2fsk8psk,
                    'demodulation_object'   : self.osmod.demod_2fsk8psk,

                    'extraction_points'     : (0.25, 0.75),
                    'phase_extraction'      : ocn.EXTRACT_INTERPOLATE,
                    'baseband_conversion'   : 'I3_rel_exp',
                    'process_debug'         : False,
                    'fft_filter'            : (-2, 2, -2, 2),
                    'fft_interpolate'       : (-2, 2, -2, 2),

                    'I3_combine'            : ocn.INTRA_COMBINE_TYPE9,
                    'I3_extract'            : ocn.INTRA_EXTRACT_TYPE4,
                    'I3_pulse_shape_type'   : ocn.PULSE_SHAPE_MANUAL,
                    'I3_pulse_shape_index'  : 3,
                    'I3_pulse_alignment'    : ocn.I3_STANDINGWAVE_PULSE_1_OF_3,
                    'I3_offsets_type'       : ocn.OFFSETS_MANUAL,
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-D', 0.312),

                    'pulse_detection'       : ocn.PULSE_DETECTION_I3,

                    'start_seq'             : '2_of_8',
                    'phase_align'           : 'start_seq',
                    'doppler_pulse_interpolation' : 'Chebyshev',
                    'extrapolate'           : 'no',
                    'extrapolate_seqlen'    : 8,
                    'downconvert_shift'     : 0.53,
        }, 

                 'MODE2' :{ 

                    'info'                  : '1.25 characters per second, 7.5 baud (bits per second)',

                 }}

  def getInitializationBlock(self):
    return self.prodmode_initialization_block
