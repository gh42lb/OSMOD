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

        #""" I modes """
        # LB28-25600-512-2-15-I
        # LB28-16-2-15-I
        # LB28-16-2-10-I
        # LB28-3200-32-2-15-I
        # LB28-32-2-10-I
        # LB28-6400-64-2-15-I
        # LB28-320-8-2-50-N
        # LB28-240-2-2-100-N. DONE

        # LB28-256-2-15-I  
        # LB28-256-2-10-I   
        # LB28-64-2-15-I    
        # LB28-6400-64-2-15-I 
        # LB28-64-2-10-I   
        # LB28-25600-256-2-15-I   
        # LB28-25600-512-2-15-I   
        #"""

        'LB28-25600-256-2-15-I':  {'encoder_callback'     : self.osmod.mod_2fsk8psk.encoder_8psk_callback,
                    'decoder_callback'     : self.osmod.demod_2fsk8psk.demodulate_2fsk_8psk,
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTriplet,
                    'text_decoder'         : self.osmod.demod_2fsk8psk.displayTextResults,
                    'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                    'info'                 : '0.15625 characters per second, 0.9375 baud (bits per second)',
                    'symbol_block_size'    : 25600,
                    'symbols_per_block'    : 1,  # per carrier!
                    'symbol_wave_function' : self.osmod.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                    'modulation_object'    : self.osmod.mod_2fsk8psk,
                    'demodulation_object'  : self.osmod.demod_2fsk8psk,
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

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 17, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 17, 5, 50),

                    'parameters'           : (1500, 0.986, 0.015, 10000, 2, 98, 0.403, 0.21, 0.828, 0.025) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

        'LB28-64-2-10-I':    {'encoder_callback'     : self.osmod.mod_2fsk8psk.encoder_8psk_callback,
                    'decoder_callback'     : self.osmod.demod_2fsk8psk.demodulate_2fsk_8psk,
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTriplet,
                    'text_decoder'         : self.osmod.demod_2fsk8psk.displayTextResults,
                    'mode_selector'        : ocn.OSMOD_MODEM_8FSK,
                    'info'                 : '0.625 characters per second, 3.75 baud (bits per second)',
                    'symbol_block_size'    : 12800,
                    'symbols_per_block'    : 1,  # per carrier!
                    'symbol_wave_function' : self.osmod.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                    'modulation_object'    : self.osmod.mod_2fsk8psk,
                    'demodulation_object'  : self.osmod.demod_2fsk8psk,
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

                    #Filter carriers
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 10, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 10, 5, 50),

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),

                    'parameters'           : (600, 0.70, 0.9, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) },  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

        'LB28-240-N' :{ 
                    'info'                 : 'mode - based on LB28-240-2-2-100-N lo carrier must be on 100Hz boundary.  33.333 characters per second, 200 baud (bits per second).',
                    'encoder_callback'     : self.osmod.mod_2fsk8psk.encoder_8psk_callback,
                    'decoder_callback'     : self.osmod.demod_2fsk8psk.demodulate_2fsk_8psk,
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTriplet,
                    'text_decoder'         : self.osmod.demod_2fsk8psk.displayTextResults,
                    'mode_selector'        : ocn.OSMOD_MODEM_8PSK,
                    'info'                 : '33.33 characters per second, 200 baud (bits per second)',
                    'symbol_block_size'    : 240,
                    'symbol_wave_function' : self.osmod.mod_2fsk8psk.halves_symbol_wave_function,
                    'modulation_object'    : self.osmod.mod_2fsk8psk,
                    'demodulation_object'  : self.osmod.demod_2fsk8psk,
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
                    'parameters'           : (700, 0.8, 0.6, 10000, 2, 98, 0.7072, 0.1, 0.1414, 0.01) ,  #magic number for phase value extraction, RRC_1, RRC_2, baseband, normalization value. extract phase num waves

        }, 
        

        'LB28-51200-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '0.15625 characters per second, 0.9375 baud (bits per second).',
                    'symbol_block_size'     : 51200,
                    'pulses_per_block'      : 512,   #1024,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.fivehundredtwelfths_symbol_wave_function,
                    #'fft_filter'            : (-0.5, 0.5, -0.5, 0.5),
                    #'fft_interpolate'       : (-1, 1, -1, 1),
                    #'fft_filter'            : (-1, 1, -1, 1),
                    #'fft_interpolate'       : (-0.75, 0.3, -0.3, 0.75),
                    #'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    #'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    'fft_filter'            : (-0.8, 0.8, -0.8, 0.8),
                    'fft_interpolate'       : (-0.8, 0.8, -0.8, 0.8),
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'D-E', 0.013),
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'E-E', 0.267),
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-D', 0.619),
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


        'LB28-25600-I3-N12' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : 'Narrow 12 Hz - 0.3125 characters per second, 1.875 baud (bits per second).',
                    'symbol_block_size'     : 25600,
                    'pulses_per_block'      : 256,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,
                    #'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    'fft_filter'           : (-0.6, 0.6, -0.6, 0.6),
                    'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    #'fft_interpolate'      : (-2.9, 1.6, -1.6, 2.9),
                    #'fft_interpolate'      : (-0.1, 4.4, -4.4, 0.1),

                    'carrier_separation'    : 10,
                    #'carrier_separation'    : 7,

                    #'parameters'            : (1500, 0.547, 0.584, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.216, 0.925, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.162, 0.746, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.001, 0.908, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #params for 15 Hz wide signal
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.552), # 15 Hz
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.434), # 9 Hz

                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.92), # 12 Hz
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.324), # 12 Hz
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'E-E', 0.177), # 12 Hz
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-B', 0.173), # 12 Hz
                    #'downconvert_shift'     : 0.159,
                    #'downconvert_shift'     : 0.333,
                    #'downconvert_shift'     : 0.413,
                    #'downconvert_shift'     : 0.323,
                    'downconvert_shift'     : 0.175,
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 12, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 12, 5, 50),


                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.12438208730191,  18.74388701266639, 257.8458], # available, low freq relative center, hi freq relative center

        }, 



        'LB28-25600-I3-N6' :{ 

                    'inherit_from'        : 'LB28-25600-I3',
                    'info'                  : 'Narrow 6 Hz - 0.3125 characters per second, 1.875 baud (bits per second).',
                    'fft_filter'           : (-0.6, 2.4, -2.4, 0.6), # 2 of 6 - 1 of 10 @ 18 - 3 of 6 - 2 of 10
                    'fft_interpolate'      : (-0.6, 1.0, -1.0, 0.6), # 3 of 6 - 3 of 10
                    'carrier_separation'    : 4,  #2 of 10 - 3 of 10 - 3 of 10
                    'parameters'            : (1500, 0.368, 0.918, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 2 of 6 - 6 of 12
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.684), # 4 of 6
                    'downconvert_shift'     : 0.461, # 4 of 6 - 4 of 10 @18
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),

        }, 


        'LB28-25600-I3' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : '0.3125 characters per second, 1.875 baud (bits per second).',
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

                    # params for 48 Hz wide signal
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.41),
                    'parameters'            : (1500, 0.216, 0.925, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.125, 19.014, 0], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.12438208730191,  18.74388701266639, 257.8458], # available, low freq relative center, hi freq relative center

        }, 


        'LB28-12800-I3E' :{ 
                    'inherit_from'          : 'LB28-12800-I3',
                    'extrapolate'           : 'yes',
        }, 



        'LB28-12800-I3-N6' :{ 
                    'inherit_from'        : 'LB28-12800-I3',
                    'info'                  : 'Narrow 6 Hz - 0.625 characters per second, 3.75 baud (bits per second).',
                    'carrier_separation'    : 4,
                    #'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    #'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    'parameters'            : (1500, 0.35, 0.78, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.172), # 3 of 6
                    'downconvert_shift'     : 0.784,

        }, 




        'LB28-12800-I3-FC10-FEC' :{ 
                    'inherit_from'          : 'LB28-12800-I3-FC10',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 0.625 characters per second, 3.75 baud (bits per second). ',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'   : 8,

        }, 


        'LB28-12800-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-12800-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 0.625 characters per second, 3.75 baud (bits per second). ',
                    'carrier_separation'    : 10,
                    'fft_filter'            : (-1.23, 2.25, -2.25, 1.23),
                    'fft_interpolate'      : (-1.0, 0.1, -0.1, 1.0),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.973),
                    'downconvert_shift'     : 0.722,
                    'parameters'            : (1500, 0.704, 0.982, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50), 
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),

        }, 


        'LB28-12800-I3-GC32' :{ 
                    'inherit_from'          : 'LB28-12800-I3',
                    'info'                  : 'Filtered Carriers - 0.625 characters per second, 3.75 baud (bits per second).',
                    'fft_filter'           : (-1.27, 0.93, -0.93, 1.27),
                    'fft_interpolate'      : (-1.34, 1.95, -1.95, 1.34),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.405),
                    'downconvert_shift'     : 0.399,

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 32, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 32, 5, 50),

        }, 



        'LB28-12800-I3' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : '0.625 characters per second, 3.75 baud (bits per second).',
                    'symbol_block_size'     : 12800,
                    'pulses_per_block'      : 128,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,
                    #'downconvert_shift'     : 0.241,
                    #'downconvert_shift'     : 0.267,
                    'downconvert_shift'     : 0.374,
                    #'parameters'            : (1500, 0.392, 0.953, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.704, 0.982, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.413),
                    'fft_filter'           : (-1, 1, -1, 1),
                    'fft_interpolate'      : (-1, 1, -1, 1),

                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, 1381.875, 1418.653, 0], # available, low freq, hi freq
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.125, 18.653, 0], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.123760536729833, 18.737747429275032, 257.8458], # available, low freq relative center, hi freq relative center

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



        'LB28-6400-I3-FC10-FEC' :{ 
                    'inherit_from'          : 'LB28-6400-I3-FC10',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 1.25 characters per second, 7.5 baud (bits per second). ',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'   : 8,

        }, 


        'LB28-6400-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 1.25 characters per second, 7.5 baud (bits per second). ',
                    'carrier_separation'    : 10,
                    'fft_filter'            : (-1.51, 4.46, -4.46, 1.51),
                    'fft_interpolate'       : (-3.18, 02.45, -2.45, 3.18),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-C', 0.655),
                    'downconvert_shift'     : 0.014,
                    'parameters'            : (1500, 0.27, 1.0, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50), 
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),

        }, 


        'LB28-6400-I3' :{ 

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '1.25 characters per second, 7.5 baud (bits per second)',
                    'symbol_block_size'     : 6400,
                    'pulses_per_block'      : 64,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixtyfourths_symbol_wave_function,
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.5, 18.556, 257.8458], # available, low freq relative center, hi freq relative center
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.497683197851984, 18.725503490166602, 257.8458], # available, low freq relative center, hi freq relative center


                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-D', 0.312),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-E', 0.372),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.544),

                    #'downconvert_shift'    : 0.52,
                    #'parameters'            : (1500, 0.567, 0.899, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.716, 0.979, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #'parameters'            : (1500, 0.046, 382, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.941, 0.883, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.333, 0.52, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.27, 1.0, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),   #3 of 3 almost perfect!!!


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



        'LB28-3200-I3-FC10-FEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC10',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'   : 8,

                    #'extrapolate'           : 'yes',

        }, 




        'LB28-3200-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'carrier_separation'    : 10,
                    #'extrapolate'           : 'yes',


                    #'fft_filter'            : (-4.29, 4.65, -4.65, 4.29),
                    #'fft_filter'            : (-3.84, 4.36, -4.36, 3.84),
                    #'fft_filter'            : (-3.1, 4.4, -4.4, 3.1),
                    'fft_filter'            : (-2.97, 5.25, -5.25, 2.97),
                    #'fft_interpolate'       : (-4.98, 3.76, -3.76, 4.98), 
                    #'fft_interpolate'       : (-3.91, 5.9, -5.9, 3.91),      
                    'fft_interpolate'       : (-3.12, 3.65, -3.65, 3.12),      
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.388),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.856),
                    #'downconvert_shift'     : 0.639, 
                    #'downconvert_shift'     : 0.536, 
                    #'downconvert_shift'     : 0.971, 
                    'downconvert_shift'     : 0.028, 
                    'parameters'            : (1500, 0.498, 0.865, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 3, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 3, 50),

        }, 


        #'LB28-3200-I3-FC10' :{ 
        #            'inherit_from'          : 'LB28-3200-I3',
        #            'info'                  : 'Filtered Carriers - 10Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
        #            'carrier_separation'    : 10,
        #            'fft_filter'            : (-4.29, 4.65, -4.65, 4.29),
        #            'fft_interpolate'       : (-4.78, 4.22, -4.22, 4.78),
        #            'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.24),
        #            'downconvert_shift'     : 0.141, 
        #            #Filter carriers
        #            'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
        #            'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
        #}, 



        'LB28-3200-I3' :{ 

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '2.5 characters per second, 15 baud (bits per second)',
                    'symbol_block_size'     : 3200,
                    'pulses_per_block'      : 32,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.thirtyseconds_symbol_wave_function,
                    'fft_filter'            : (-4, 4, -4, 4),
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    #'extrapolate'           : 'yes',

                    #'dcs_type'              : ocn.DCS_GENERAL,
                    'dcs_type'              : ocn.DCS_FREQUENCY_SPECIFIC,
                    'dcs_by_frequency'      : {'160':0.159, '200':0.601, '320':0.27, '640':0.27, '800':0.922, '960':0.291, '1000':0.622, '1010':0.616, '1040':0.78, '1080':0.707, '1120':0.51, '1160':0.12, '2000':0.76, '2640':0.866, '2720':0.97 },

                    'downconvert_shift'     : 0.32,     # 1.1 of 3,  0.4 of 3,   0.5 of 3,  0.5 of 3

                    #'parameters'            : (1500, 0.125, 1.0, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),  #2 of 3
                    #'parameters'            : (1500, 0.223, 0.991, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),  #2.6 of 3
                    #'parameters'            : (1500, 0.822, 0.997, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), #3 of 3 perfect!

                    #'parameters'            : (1500, 0.284, 0.865, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),  #3 of 3 perfect!!!
                    #'parameters'            : (1500, 0.077, 0.731, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.512, 0.466, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #'downconvert_shift'     : 0.017,    #0.75 of 3.   0.5 of 3,    0.75 of 3,   0 of 3
                    #'downconvert_shift'     : 0.422,    #0.3 of 3, 0.5 of 3

                    #'downconvert_shift'     : 0.425,   # 0 of 3
                    #'downconvert_shift'     : 0.946,     # 0 of 3
                    #'downconvert_shift'     : 0.54,     # 0 of 3
                    #'downconvert_shift'     : 0.992,     #0.6 of 3


                    #'parameters'            : (1500, 0.939, 0.96, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-D', 0.312),
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, 1382.5, 1417.112, 0], # available, low freq, hi freq
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.5, 17.112, 247.1610], # available, low freq, hi freq
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.49521515, 17.51528191, 247.1610], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE,  -17.495215152030596,  17.515281914900925, 247.1610], # available, low freq, hi freq
                    #'resample_params_48k'  : [ocn.RESAMPLE_AVAILABLE,  -17.495489912272888,  17.515166345342777, 247.1610], # available, low freq, hi freq
                    'resample_params_48k'  : [ocn.RESAMPLE_AVAILABLE,  -17.49399399399431,  17.515653291377703, 247.1610], # available, low freq, hi freq
                    #'resample_params_48k'  : [ocn.RESAMPLE_AVAILABLE,  -17.49399399399431,  17.515678885166608, 247.1610], # available, low freq, hi freq

                    #'parameters'            : (1500, 0.238, 0.999, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),    #0.5 of 3
                    #'parameters'            : (1500, 0.545, 0.925, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),     #0 of 3


                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.823),   # 0.2 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.602),   # 0.2 of 3
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.748), 

                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.033),   # 0.1


                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.035),    #1.6 of 3, 0.25 of 3, 0.5 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-E', 0.266),    # 1 of 3,   0.6 of 3, 0 of 3

                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.617),    #0 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-E', 0.68),     # 0.5 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.548),     #0 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.093),    #0.25 of 3
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.41),     # 0.75 of 3, 0.6

                    #next only for 48k search
                    'persistent_search'     : (1, 0.95, -0.005, "yes"), #hi range, lo range, inc, scan entire range

                    'rotation_increments'  : 100,


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

                    #'I3_parameters'        : (0.99, 0.99, 2e-3, 'E-E', 0.115),
                    #'downconvert_shift'    : 0.535,

                    'pulse_train_sigma'     : 21.57,
                    'pulse_start_sigma'     : 21.2,
                    'pulse_start_envelope_sigma' : 16.48,

                    'extrapolate_seqlen'   : 8,
                    #'extrapolate'           : 'no',

                    #'parameters'           : (1500, 0.099, 0.152, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'           : (1500, 0.982, 0.168, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'           : (1500, 0.578, 0.933, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'           : (1500, 0.748, 0.831, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01), #best for 64 length message
                    #'parameters'           : (1500, 0.423, 0.953, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01), #best for 128 length message
                    #'parameters'           : (1500, 0.73, 0.606, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01), #best for 64 & 128 length message

        }, 


        'LB28-1600-I3E' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'extrapolate'           : 'yes',
        }, 





        'LB28-1600-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 10,
                    #'fft_filter'            : (-4.29, 4.65, -4.65, 4.29),
                    #'fft_interpolate'       : (-4.78, 4.22, -4.22, 4.78),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.24),
                    #'downconvert_shift'     : 0.32, 
                    #'parameters'            : (1500, 0.273, 0.24, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),

        }, 




        'LB28-1600-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '5.0 characters per second, 30.0 baud (bits per second)',
                    'symbol_block_size'     : 1600,
                    'pulses_per_block'      : 16,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixteenths_symbol_wave_function,
                    'fft_filter'            : (-4, 4, -4, 4),
                    'fft_interpolate'       : (-3, 2, -2, 3),

                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.856),

                    #'downconvert_shift'     : 0.906,
                    'downconvert_shift'     : 0.441,
                    #'downconvert_shift'     : 0.87,
                    #'downconvert_shift'     : 0.949,

                    #'parameters'            : (1500, 0.734, 0.76, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.255, 0.894, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    #'parameters'            : (1500, 0.141, 1.0, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 2 of 4
                    #'parameters'            : (1500, 0.973, 0.884, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), #????
                    #'parameters'            : (1500, 0.549, 0.271, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 1 of 3
                    #'parameters'            : (1500, 0.112, 0.785, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 1 of 3
                    #'parameters'            : (1500, 0.024, 0.214, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 1 of 3

                    #'parameters'            : (1500, 0.751, 0.89, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), #4 of 4
                    'parameters'            : (1500, 0.273, 0.24, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 3 of 3


                    'persistent_search'     : (1, 0.95, -0.005, "yes"), #hi range, lo range, inc, scan entire range
                    #'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -15, 14.224, 0], # available, low freq, hi freq
                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE,  -17.68012094363803, 18.907336705869966, 247.1610], # available, low freq, hi freq
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 70, 5, 50),

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 80, 5, 50),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 87, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 87, 5, 50),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 91, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 91, 5, 50),

        }, 



        'LB28-800-I3E-FEC' :{ 
                    'inherit_from'          : 'LB28-800-I3-FEC',
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
                    'extrapolate'           : 'yes',

        }, 


        'LB28-800-I3-FEC' :{ 
                    'inherit_from'         : 'LB28-800-I3',
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
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
                    #'downconvert_shift'    : 0.019,
                    'downconvert_shift'    : 0.158,
                    #'downconvert_shift'    : 0.04,
                    'FDM_parameters'       : [2, 95.768], #frequency division mutiplexing: multiplier, separation
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 200, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 200, 5, 50),
                    'tx_filter'             : (ocn.FILTER_NONE, ocn.FILTER_NONE, 200, 5, 50),
                    'rx_filter'             : (ocn.FILTER_NONE, ocn.FILTER_NONE, 200, 5, 50),
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'B-D', 0.178), 
        }, 


        'LB28-800-I3E' :{ 
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
                    'inherit_from'          : 'LB28-800-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-800-I3' :{ 
                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.eighths_symbol_wave_function,
                    'symbol_block_size'     : 800,
                    'pulses_per_block'      : 8,
                    'fft_filter'            : (-20, 16, -16, 20),
                    'fft_interpolate'       : (-3, 2, -2, 3),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 85, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 85, 5, 50),

                    #'pulse_train_sigma'    : 12.33,
                    #'pulse_train_sigma'     : 13.36,
                    'pulse_train_sigma'     : 11.09,
                    #'pulse_start_sigma'     : 24.47,
                    'pulse_start_sigma'     : 10.73,
                    #'pulse_start_envelope_sigma' : 17.7,
                    'pulse_start_envelope_sigma' : 5.99,

                    #'downconvert_shift'    : 0.123,  #151
                    #'downconvert_shift'    : 0.876,
                    'downconvert_shift'    : 0.086,

                    # TEST CODE ONLY
                    #'I3_offsets_type'      : ocn.OFFSETS_MSGLEN_SPECIFIC,
                    #'pattern_by_msglen'    : {'16':('A-E', 0.298), '32':('B-B', 0.461), '64':('B-B', 0.461), '128':('A-A', 0.153), '256':('B-B', 0.366), '512':('B-B', 0.461)},

                    'persistent_search'     : (1, 0.95, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.90, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.80, -0.002, "yes"), #hi range, lo range, inc, scan entire range

                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-D', 0.548), 
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'C-E', 0.209), 

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
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 68, 5, 50),

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 75, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 75, 5, 50),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 150, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 150, 5, 50),

                    'persistent_search'     : (1, 0.95, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.93, -0.001, "yes"), #hi range, lo range, inc, scan entire range
                    'symbol_block_size'     : 400,
                    #'downconvert_shift'     : 0.514,
                    #'downconvert_shift'     : 0.595,
                    'downconvert_shift'     : 0.083,
                    #'I3_offsets_type'      : ocn.OFFSETS_MANUAL,
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-C', 0.701), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-D', 0.418), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'A-A', 0.008), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'C-C', 0.387), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'E-E', 0.804), 
                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'B-D', 0.436), 
                    #'I3_parameters'         : (0.99, 0.99, 2e-3, 'E-E', 0.057), 
                    'extrapolate'           : 'no', #no rotation tables yet!!!
                    #'I3_pulse_shape_type'  : ocn.PULSE_SHAPE_MANUAL,
                    #'parameters'            : (1500, 0.661, 0.063, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'parameters'            : (1500, 0.481, 0.077, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),
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
