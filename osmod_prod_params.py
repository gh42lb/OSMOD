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
        

        'LB28-51200-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-51200-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 0.15625 characters per second, 0.9375 baud (bits per second).',
                    'carrier_separation'    : 40,

                    #'fft_filter'            : (-6.33,6.72,-6.72,6.33),
                    #'fft_interpolate'       : (-4.29,1.44,-1.44,4.29),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.909),
                    #'downconvert_shift'     : 0.932, 
                    #'parameters'            : (1500, 0.042, 0.885, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),


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

                    'fft_filter'           : (-0.6, 0.6, -0.6, 0.6),
                    'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'E-E', 0.177), # 12 Hz
                    'downconvert_shift'     : 0.175,
                    'parameters'            : (1500, 0.216, 0.925, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    'carrier_separation'    : 10,
                    #'carrier_separation'    : 7,

                    #params for 15 Hz wide signal
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.552), # 15 Hz
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.434), # 9 Hz

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 12, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 12, 5, 50),

                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -18.12438208730191,  18.74388701266639, 257.8458], # available, low freq relative center, hi freq relative center

        }, 



        'LB28-25600-I3-N6' :{ 

                    'inherit_from'        : 'LB28-25600-I3',
                    'info'                  : 'Narrow 6 Hz - 0.3125 characters per second, 1.875 baud (bits per second).',

                    'carrier_separation'    : 4,

                    'fft_filter'           : (-0.6, 2.4, -2.4, 0.6), # 2 of 6 - 1 of 10 @ 18 - 3 of 6 - 2 of 10
                    'fft_interpolate'      : (-0.6, 1.0, -1.0, 0.6), # 3 of 6 - 3 of 10
                    'parameters'            : (1500, 0.368, 0.918, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 2 of 6 - 6 of 12
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.684), # 4 of 6
                    'downconvert_shift'     : 0.461, # 4 of 6 - 4 of 10 @18

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),

        }, 



        'LB28-25600-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-25600-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 0.3125 characters per second, 1.875 baud (bits per second).',
                    'carrier_separation'    : 40,

                    'fft_filter'            : (-1.072,0.509,-0.509,1.072),
                    'fft_interpolate'       : (-3.027,0.519,-0.519,3.027),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.651),
                    'downconvert_shift'     : 0.977, 
                    'parameters'            : (1500, 0.052, 0.844, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),


        }, 


        'LB28-25600-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-25600-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 0.3125 characters per second, 1.875 baud (bits per second). ',
                    'carrier_separation'    : 10,

                    #'fft_filter'            : (-5.83, 5.4, -5.4, 5.83),
                    #'fft_interpolate'       : (-6.27, 1.42, -1.42, 6.27),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.042),
                    #'downconvert_shift'     : 0.814, 
                    #'parameters'            : (1500, 0.216, 0.807, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),

        }, 


        'LB28-25600-I3' :{ 

                    'inherit_from'        : 'LB28-I3-BASE',
                    'info'                  : '0.3125 characters per second, 1.875 baud (bits per second).',
                    'symbol_block_size'     : 25600,
                    'pulses_per_block'      : 256,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.twohundredfiftysixths_symbol_wave_function,

                    'fft_filter'           : (-1.4, 1.4, -1.4, 1.4),
                    'fft_interpolate'      : (-1.4, 1.4, -1.4, 1.4),

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
                    'parameters'            : (1500, 0.35, 0.78, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.172),
                    'downconvert_shift'     : 0.784,

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 6, 5, 50),

        }, 




        'LB28-12800-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-12800-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 0.625 characters per second, 3.75 baud (bits per second).',
                    'carrier_separation'    : 40,

                    'fft_filter'            : (-0.694,0.998,-0.998,0.694),
                    'fft_interpolate'       : (-0.718,0.847,-0.847,0.718),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.916),
                    'downconvert_shift'     : 0.018, 
                    'parameters'            : (1500, 0.166, 0.953, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-56, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),


        }, 



        'LB28-12800-I3-FC10-VFEC' :{ 
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

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '0.625 characters per second, 3.75 baud (bits per second).',
                    'symbol_block_size'     : 12800,
                    'pulses_per_block'      : 128,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.onehundredtwentyeighths_symbol_wave_function,

                    'fft_filter'            : (-1, 1, -1, 1),
                    'fft_interpolate'       : (-1, 1, -1, 1),
                    'downconvert_shift'     : 0.374,
                    'parameters'            : (1500, 0.704, 0.982, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-E', 0.413),

                    'resample_params'       : [ocn.RESAMPLE_AVAILABLE, -18.123760536729833, 18.737747429275032, 257.8458], # available, low freq relative center, hi freq relative center

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




        'LB28-6400-I3-FC50' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second).',
                    'carrier_separation'    : 50,

                    #'fft_filter'            : (-6.33,6.72,-6.72,6.33),
                    #'fft_interpolate'       : (-4.29,1.44,-1.44,4.29),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.909),
                    #'downconvert_shift'     : 0.932, 
                    #'parameters'            : (1500, 0.042, 0.885, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 


                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-120, 25), 2, 60),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-120, 25), 2, 60),
                    #'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -40, 2, 60),
                    #'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -40, 2, 60),


        }, 



        'LB28-6400-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second).',
                    'carrier_separation'    : 40,

                    # final (for now)
                    'fft_filter'            : (-2.416,2.039,-2.039,2.416),
                    'fft_interpolate'       : (-1.111,1.895,-1.895,1.111),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.273),
                    'downconvert_shift'     : 0.665, 
                    'parameters'            : (1500, 0.148, 0.923, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-115, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-115, 20), 2, 40),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-53, 20), 2, 40),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-53, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -47, 2, 40),

        }, 


        'LB28-6400-I3-FC15' :{ 
                    'inherit_from'          : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second).',
                    'carrier_separation'    : 15,

                    #'fft_filter'            : (-1.66,1.92,-1.92,1.66),
                    #'fft_interpolate'       : (-2.01,1.93,-1.93,2.01),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-C', 0.6),
                    #'downconvert_shift'     : 0.836, 
                    #'parameters'            : (1500, 0.331, 0.63, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 2, 4, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 2, 4, 50),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -49, 4, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -49, 4, 50),

                    #'extrapolate'           : 'yes',

        }, 


        'LB28-6400-I3-FC10-VFEC' :{ 
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
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50), 
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -2, 5, 50), 
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -2, 5, 50),

        }, 




        'LB28-6400-I3-FC5' :{ 
                    'inherit_from'        : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 4 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second)',
                    'carrier_separation'    : 5,

                    'fft_filter'           : (-2.641,3.476,-3.476,2.641),
                    'fft_interpolate'      : (-2.588,0.729,-0.729,2.588),
                    'parameters'            : (1500, 0.727, 0.82, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'E-E', 0.894),
                    'downconvert_shift'     : 0.744,

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),

        }, 



        'LB28-6400-I3-FC4' :{ 
                    'inherit_from'        : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 4 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second)',
                    'carrier_separation'    : 4,

                    'fft_filter'           : (-2.28,2.84,-2.84,2.28),
                    'fft_interpolate'      : (-2.01,3.72,-3.72,2.01),
                    #'parameters'            : (1500, 0.35, 0.78, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-C', 0.47),
                    'downconvert_shift'     : 0.944,

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),

        }, 



        'LB28-6400-I3-FC3' :{ 
                    'inherit_from'        : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 7.25 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second)',
                    'carrier_separation'    : 3,

                    'fft_filter'           : (-2.12,1.52,-1.52,2.12),
                    'fft_interpolate'      : (-2.0,1.5,-1.5,2.0),
                    'parameters'            : (1500, 0.6, 0.928, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.111),
                    'downconvert_shift'     : 0.44,

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),

        }, 



        'LB28-6400-I3-FC2' :{ 
                    'inherit_from'        : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 5.47 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second)',
                    'carrier_separation'    : 2,

                    'fft_filter'           : (-1.737,1.135,-1.135,1.737),
                    'fft_interpolate'      : (-2.311,1.21,-1.21,2.311),
                    'parameters'            : (1500, 0.589, 0.721, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-D', 0.147),
                    'downconvert_shift'     : 0.552,

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -31, 2, 30),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -31, 2, 30),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),

        }, 


        'LB28-6400-I3-FC1' :{ 
                    'inherit_from'        : 'LB28-6400-I3',
                    'info'                  : 'Filtered Carriers - 4 Hz Wide - 1.25 characters per second, 7.5 baud (bits per second)',
                    'carrier_separation'    : 1,

                    'fft_filter'           : (-1.737,1.135,-1.135,1.737),
                    'fft_interpolate'      : (-2.311,1.21,-1.21,2.311),
                    'parameters'            : (1500, 0.589, 0.721, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-D', 0.147),
                    'downconvert_shift'     : 0.552,

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -19, 4, 40),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -19, 4, 40),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -22, 4, 40),

        }, 


        'LB28-6400-I3' :{ 

                    'inherit_from'          : 'LB28-I3-BASE',
                    'info'                  : '1.25 characters per second, 7.5 baud (bits per second)',
                    'symbol_block_size'     : 6400,
                    'pulses_per_block'      : 64,
                    'symbol_wave_function'  : self.osmod.mod_2fsk8psk.sixtyfourths_symbol_wave_function,

                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE, -17.497683197851984, 18.725503490166602, 257.8458], # available, low freq relative center, hi freq relative center


                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.544),
                    'parameters'            : (1500, 0.27, 1.0, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),   #3 of 3 almost perfect!!!

                    #'rotation_increments'   : 100,

                    #'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3E-FEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FEC',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3-VFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,

                    'extrapolate_seqlen'   : 8,

        }, 


        'LB28-3200-I3E' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'extrapolate'           : 'yes',
        }, 


        'LB28-3200-I3-FC40-LFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC40',
                    'info'                  : 'LDPC - Filtered Carriers - 10 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'inherit_from'          : 'LB28-3200-I3-FC10-VFEC',

                    'FEC'                   : ocn.FEC_LDPC,
                    #'fec_params'           : (400, 361, 2, 20, 500),   # 10/12
                    #'fec_params'           : (400, 381, 2, 40, 500), # not as effective    10/12
                    #'fec_params'            : (400, 339, 4, 25, 500),    # 10/12
                    #'fec_params'           : (500, 481, 2, 50, 500),   # works well     10/14
                    #'fec_params'            : (500, 401, 2, 10, 500),  # works well     10/14
                    'fec_params'            : (560, 339, 4, 10, 7),     # works well     10/15
                    #'fec_params'           : (600, 481, 2, 10, 500),       # 10/17
                    #'fec_params'           : (600, 589, 2, 100, 500),      # 10/17
                    #'fec_params'           : (600, 581, 2, 60, 500),      # 10/17

                    #'fec_params'            : (1005, 202, 2, 3, 500),     # 10/9 ?????
                    #'fec_params'            : (1665, 202, 4, 5, 500),    #??????

        }, 


        """
        'LB28-3200-I3-FC40-VFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC40',
                    'info'                  : 'Viterbi - Filtered Carriers - 10 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTripletFEC,

                    'FEC'                   : ocn.FEC_VITERBI,
                    #'fec_params'            : (13 , 5890 , 6271, []),      # 1/2   works well

                    'fec_params'           : (16, 256, 809, [1,1,0,1]), #2/3      works
                    #'fec_params'           : (16, 23095, 44876, [1,1,0,1]), #2/3      works 
                    #'fec_params'           : (13, 7515, 5754, [1,1,0,1]), #2/3      works
                    #'fec_params'           : (13, 7515, 5754, [0,1,1,1]), #2/3      works
                    #'fec_params'           : (17, 85365, 59157, [0,1,1,1]), #2/3      works

                    #'fec_params'           : (16, 120, 595, [1,0,0,0,1,0,1,0,1,1,1,0]),     #5/6       
                    #'fec_params'           : (15, 23552, 21839, [1,0,0,0,1,0,1,0,1,1,1,0]), #5/6       
                    #'fec_params'           : (16, 120, 595, [1,0,0,1,1,0,1,0,1,1,0,0]),      #5/6      
                    #'fec_params'           : (16, 3805, 3370, [1,0,1,1]), #2/3      
                    #'fec_params'           : (16, 256, 809, [1,1,1,0]), #2/3       
                    #'fec_params'           : (16, 12, 167, [1,0,1,1]), #2/3        


                    #???
                    #'fec_params'           : (16, 256, 809, [1,1,1,1,0,0]),                 #3/4         ???
                    #'fec_params'           : (16, 256, 809, [1,1,1,1,0,0,0,1]),             #4/5         ???
                    #'fec_params'           : (16, 120, 595, [0,0,1,1,0,0,1,0,1,0,0,1,0,1]), #7/8         ???



                    'holographic_decode'    : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'          : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'              : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'    : 8,

                    #'extrapolate'           : 'yes',

        }, 
        """



        'LB28-3200-I3-FC50' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'carrier_separation'    : 50,

                    'fft_filter'            : (-3.05,3.26,-3.26,3.05),
                    'fft_interpolate'       : (-0.26,4.77,-4.77,0.26),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-E', 0.544),
                    'downconvert_shift'     : 0.37, 
                    'parameters'            : (1500, 0.603, 0.982, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_NOTCH, -50, 2, 40),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_NOTCH, -50, 2, 40),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 25), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 25), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),

        }, 



        'LB28-3200-I3-FC40-VFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC40',
                    'info'                  : 'Viterbi - Filtered Carriers - 52.6 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                   : ocn.FEC_VITERBI,
                    'fec_params'            : (11 , 861 , 2 , [0,1,1,1]), # 6 of 20 low distortion

                    'holographic_decode'    : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'          : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'              : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'    : 8,

        }, 



        'LB28-3200-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'carrier_separation'    : 40,

                    # final (for now)
                    'fft_filter'            : (-3.08,2.6,-2.6,3.08),
                    'fft_interpolate'       : (-6.22,6.05,-6.05,6.22),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.626),
                    'downconvert_shift'     : 0.336, 
                    'parameters'            : (1500, 0.849, 0.948, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #distortion
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-190, 20), 2, 30),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-190, 20), 2, 30),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),

        }, 


        'LB28-3200-I3-FC10-LFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC10',
                    'info'                  : 'LDPC - Filtered Carriers - 10 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'inherit_from'          : 'LB28-3200-I3-FC10-VFEC',

                    'FEC'                   : ocn.FEC_LDPC,
                    'fec_params'            : (560, 339, 4, 10, 6),     # works well
                    #'fec_params'            : (560, 339, 4, 10, -11),     # works well
                    #'fec_params'            : (560, 339, 4, 10, 0),     # works well

                    #'fec_params'           : (400, 361, 2, 20, 500),
                    #'fec_params'           : (400, 381, 2, 40, 500), # not as effective
                    #'fec_params'           : (500, 481, 2, 50, 500),   # works well
                    #'fec_params'            : (500, 401, 2, 10, 500),  # works well
                    #'fec_params'           : (600, 481, 2, 10, 500),
                    #'fec_params'           : (600, 589, 2, 100, 500),
                    #'fec_params'           : (600, 581, 2, 60, 500),
                    #'fec_params'            : (1005, 202, 2, 3, 500),  
                    #'fec_params'            : (1665, 202, 4, 5, 500),  
                    #'fec_params'            : (400, 339, 4, 25, 500),  
                    #'fec_params'            : (560, 339, 4, 10, 7),     # works well

        }, 



        'LB28-3200-I3-FC10-VFEC' :{ 
                    'inherit_from'          : 'LB28-3200-I3-FC10',
                    'info'                  : 'Viterbi - Filtered Carriers - 10 Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTripletFEC,

                    'FEC'                   : ocn.FEC_VITERBI,
                    'fec_params'            : (13 , 5890 , 6271, []),

                    'holographic_decode'    : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'          : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'              : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'    : 8,

                    #'extrapolate'           : 'yes',

        }, 




        'LB28-3200-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-3200-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 2.5 characters per second, 15 baud (bits per second). ',
                    'carrier_separation'    : 10,
                    #'extrapolate'           : 'yes',

                    'fft_filter'            : (-2.97, 5.25, -5.25, 2.97),
                    'fft_interpolate'       : (-3.12, 3.65, -3.65, 3.12),      
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.856),
                    'downconvert_shift'     : 0.028, 
                    'parameters'            : (1500, 0.498, 0.865, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 3, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 3, 50),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 5, 50),

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
                    'downconvert_shift'     : 0.32,     # 1.1 of 3,  0.4 of 3,   0.5 of 3,  0.5 of 3
                    'parameters'            : (1500, 0.512, 0.466, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-B', 0.748), 

                    #'extrapolate'           : 'yes',

                    #'dcs_type'              : ocn.DCS_GENERAL,
                    'dcs_type'              : ocn.DCS_FREQUENCY_SPECIFIC,
                    'dcs_by_frequency'      : {'160':0.159, '200':0.601, '320':0.27, '640':0.27, '800':0.922, '960':0.291, '1000':0.622, '1010':0.616, '1040':0.78, '1080':0.707, '1120':0.51, '1160':0.12, '2000':0.76, '2640':0.866, '2720':0.97 },

                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE,  -17.495215152030596,  17.515281914900925, 247.1610], # available, low freq, hi freq
                    'resample_params_48k'  : [ocn.RESAMPLE_AVAILABLE,  -17.49399399399431,  17.515653291377703, 247.1610], # available, low freq, hi freq

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


        'LB28-1600-I3-VFEC' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'text_encoder'         : self.osmod.mod_2fsk8psk.stringToTripletFEC,
                    'FEC'                  : ocn.FEC_VITERBI,
                    'fec_params'           : (13 , 5890 , 6271, []),
                    'holographic_decode'   : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'         : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'             : ocn.MSGTYPE_FIXED_LENGTH,

                    'pulse_train_sigma'     : 21.57,
                    'pulse_start_sigma'     : 21.2,
                    'pulse_start_envelope_sigma' : 16.48,

                    'extrapolate_seqlen'   : 8,
                    #'extrapolate'           : 'no',


        }, 


        'LB28-1600-I3E' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'extrapolate'           : 'yes',
        }, 


        # not effective
        'LB28-1600-I3-FC100' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 110.5 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 100,

                    # final (for now)
                    'fft_filter'            : (-5.24,1.96,-1.96,5.24),
                    'fft_interpolate'       : (-6.95,5.35,-5.35,6.95),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.727),
                    'downconvert_shift'     : 0.184, 
                    'parameters'            : (1500, 0.167, 0.952, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-200, 50), 2, 30), #0.28 to 0.4    0.4
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-200, 50), 2, 30),

                    #distortion
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-190, 50), 2, 30), #0.22 to 0.4    1.4
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-190, 50), 2, 30),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 50), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 50), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),

                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 50), 2, 30), #0.23 to 0.36    2.2
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 50), 2, 30),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-220, 40), 2, 30), #0.25 to 0.34    0.6
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-220, 40), 2, 30),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-230, 40), 2, 30), #0.28 to 0.45    -1.2
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-230, 40), 2, 30),

        }, 


        # bypass for now
        'LB28-1600-I3-FC80' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 91 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 80,

                    #'fft_filter'            : (-5.78,5.65,-5.65,5.78),
                    #'fft_interpolate'       : (-0.99,4.48,-4.48,0.99),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'C-C', 0.485),
                    #'downconvert_shift'     : 0.136, 
                    #'parameters'            : (1500, 0.926, 0.74, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 


                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 12.5), 2, 30), #0.36 to 0.43    2.1
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 12.5), 2, 30),
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 40), 2, 30), #0.30 to 0.44
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 40), 2, 30),

        }, 


        'LB28-1600-I3-FC50' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 59.2 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 50,

                    # final (for now)
                    'fft_filter'            : (-1.29,6.94,-6.94,1.29),
                    'fft_interpolate'       : (-4.6,3.66,-3.66,4.6),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.236),
                    'downconvert_shift'     : 0.656, 
                    'parameters'            : (1500, 0.366, 0.926, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #distortion
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 12.5), 2, 30), 
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-180, 12.5), 2, 30),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 25), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 25), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),

        }, 

        'LB28-1600-I3-FC40' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 52.6 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 40,

                    # final (for now)
                    'fft_filter'            : (-6.33,6.72,-6.72,6.33),
                    'fft_interpolate'       : (-4.29,1.44,-1.44,4.29),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'D-D', 0.909),
                    'downconvert_shift'     : 0.932, 
                    'parameters'            : (1500, 0.042, 0.885, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #distortion
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-160, 12.5), 2, 30),  #0.13 to 26     1.76
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-160, 12.5), 2, 30),

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 20), 2, 40),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-50, 20), 2, 40),
                    'tx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),
                    'rx_filter2'            : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, -50, 2, 40),

        }, 

        'LB28-1600-I3-FC25' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 36 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 25,

                    # final (for now)
                    'fft_filter'            : (-5.5,6.64,-6.64,5.5),
                    'fft_interpolate'       : (-3.86,1.05,-1.05,3.86),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-A', 0.769),
                    'downconvert_shift'     : 0.699, 
                    'parameters'            : (1500, 0.381, 0.746, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-15, 12.5), 2, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS_X2, (-15, 12.5), 2, 50),

        }, 


        'LB28-1600-I3-FC20' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 31.7 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 20,

                    # final (for now)
                    'fft_filter'            : (-5.88,6.91,-6.91,5.88),
                    'fft_interpolate'       : (-4.77,6.09,-6.09,4.77),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-B', 0.816),
                    'downconvert_shift'     : 0.128, 
                    'parameters'            : (1500, 0.929, 0.977, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 3, 3, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 3, 3, 50),

                    'tx_filter2'           : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_NOTCH_2, 40, 2, 30),
                    'rx_filter2'           : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_NOTCH_2, 40, 2, 30),

        }, 


        'LB28-1600-I3-FC15' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 25.6 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 15,

                    # final (for now)
                    'fft_filter'            : (-5.3,5.8,-5.8,5.3),
                    'fft_interpolate'       : (-1.73,6.64,-6.64,1.73),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-B', 0.6),
                    'downconvert_shift'     : 0.598, 
                    'parameters'            : (1500, 0.453, 0.761, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 2, 4, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 2, 4, 50),

        }, 



        'LB28-1600-I3-FC10-LFEC' :{ 
                    'inherit_from'          : 'LB28-1600-I3-FC10',
                    'info'                  : 'LDPC - Filtered Carriers - 23.3 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',

                    'FEC'                   : ocn.FEC_LDPC,
                    'fec_params'            : (560, 339, 4, 10, 7),     # works well
                    #'fec_params'           : (500, 481, 2, 50, 500),   # works well     10/14
                    #'fec_params'            : (500, 401, 2, 10, 500),  # works well     10/14
                    #'fec_params'            : (560, 339, 4, 10, 7),     # works well     10/15

        }, 



        'LB28-1600-I3-FC10-VFEC2' :{ 
                    'inherit_from'          : 'LB28-1600-I3-FC10-VFEC',
                    'info'                  : 'Viterbi - Filtered Carriers - 23.3 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',

                    #'fec_params'            : (11 , 861 , 2 , [0,1,1,1]), # 6 of 20 low distortion
                    #'fec_params'            : (11, 861, 2, [0,1,1,1]), # 3 of 10 with low decode distortion
                    'fec_params'            : (19, 93986, 444204, [0,1,1,1]), # 6 of 10 but decode distortion
                    #'fec_params'            : (13 , 5248 , 4576 , [1,1,1,0])
                    #'fec_params'            : (12 , 16 , 2212 , [1,1,0,1])
                    #'fec_params'            : (11 , 6 , 675 , [1,0,1,1])
                    #'fec_params'            : (13 , 2419 , 669 , [1,1,1,0])


        }, 


        'LB28-1600-I3-FC10-VFEC' :{ 
                    'inherit_from'          : 'LB28-1600-I3-FC10',
                    'info'                  : 'Viterbi - Filtered Carriers - 23.3 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTripletFEC,

                    'FEC'                   : ocn.FEC_VITERBI,

                    'fec_params'            : (11 , 861 , 2 , [0,1,1,1]), # 6 of 20 low distortion

                    #'fec_params'            : (12 , 1423 , 16 , [1,1,0,1]), # 2 of 20
                    #'fec_params'            : (11, 861, 2, [0,1,1,1]), # 3 of 10 with low decode distortion
                    #'fec_params'            : (19, 93986, 444204, [0,1,1,1]), # 6 of 10 but decode distortion

                    #'fec_params'            : (16 , 52915 , 12072 , [1,1,1,1,0,0,0,0,0,1,0,1]),
                    #'fec_params'            : (19 , 168183 , 20892 , [1,1,1,1,1,0,0,0,1,0,0,0]),
                    #'fec_params'            : (16 , 52915 , 12072 , [1,1,1,1,0,0,0,0,0,1,0,1]),

                    #'fec_params'            : (14, 9725, 8427, [1,1,1,1,0,1,0,0,1,0,0,0]),     #5/6       
                    #'fec_params'            : (17, 92104, 28536, [1,1,1,1,0,0,0,0,1,0,0,1]),     #5/6       
                    #'fec_params'            : (19, 125643, 152972, [1,1,1,1,0,1,0,0,1,0,1,0]),     #5/6       
                    #'fec_params'            : (18, 16519, 138370, [1,1,1,1,0,0,0,0,1,0,1,0]),     #5/6       
                    #'fec_params'            : (14, 1432, 528, [1,1,1,1,0,0,0,0,1,0,0,1]),     #5/6       
                    #'fec_params'            : (17, 68853, 40425, [1,1,1,1,0,0,0,0,1,0,0,1]),     #5/6       
                    #'fec_params'            : (17, 22831, 5325, [1,1,1,1,0,1,0,0,0,0,1,0]),     #5/6       



                    # also ran...
                    #'fec_params'            : (15, 20637, 7659, [1,0,1,1]), # 2 of 10
                    #'fec_params'            : (17 , 22952 , 31765, [0,1,1,1]), # 3 of 10
                    #'fec_params'            : (12 , 1933 , 942, []), # 2 of 10
                    #'fec_params'            : (13 , 5890 , 6271, []), # 7 of 10
                    #'fec_params'            : (11 , 317 , 1840, [1,1,0,1]), # 3 of 10

                    #'fec_params'           : (16, 120, 595, [1,0,0,0,1,0,1,0,1,1,1,0]),     #5/6       


                    'holographic_decode'    : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'          : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'              : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'    : 8,

        }, 


        'LB28-1600-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-1600-I3',
                    'info'                  : 'Filtered Carriers - 23.3 Hz Wide - 5.0 characters per second, 30.0 baud (bits per second). ',
                    'carrier_separation'    : 10,

                    # final (for now)
                    'fft_filter'            : (-6.65, 4.98, -4.98, 6.65),
                    'fft_interpolate'       : (-6.26, 1.24, -1.24, 6.26),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'A-C', 0.716),
                    'downconvert_shift'     : 0.283, 
                    'parameters'            : (1500, 0.717, 0.733, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),

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
                    'downconvert_shift'     : 0.441,
                    'parameters'            : (1500, 0.273, 0.24, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), # 3 of 3

                    'persistent_search'     : (1, 0.95, -0.005, "yes"), #hi range, lo range, inc, scan entire range

                    'resample_params'      : [ocn.RESAMPLE_AVAILABLE,  -17.68012094363803, 18.907336705869966, 247.1610], # available, low freq, hi freq

                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 87, 5, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 87, 5, 50),

        }, 



        'LB28-800-I3E-FEC' :{ 
                    'inherit_from'          : 'LB28-800-I3-FEC',
                    'info'                  : 'based on LB28-800-8-2-37-I3E8-FEC  10.0 characters per second, 60.0 baud (bits per second)',
                    'extrapolate'           : 'yes',

        }, 


        'LB28-800-I3-VFEC' :{ 
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
                     'I3_parameters'        : (0.99, 0.99, 2e-3, 'A-E', 0.205),
                    'extrapolate_seqlen'   : 8,
                    #'extrapolate'           : 'yes',
                    'extrapolate'           : 'no',
                    'downconvert_shift'    : 0.535,
                    'parameters'           : (1500, 0.982, 0.168, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

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


        'LB28-800-I3-FC10-VFEC' :{ 
                    'inherit_from'          : 'LB28-800-I3-FC10',
                    'info'                  : 'Viterbi - Filtered Carriers - 10 Hz Wide - 10.0 characters per second, 60.0 baud (bits per second) ',
                    'text_encoder'          : self.osmod.mod_2fsk8psk.stringToTripletFEC,

                    'FEC'                   : ocn.FEC_VITERBI,
                    'fec_params'            : (13 , 5890 , 6271, []),

                    'holographic_decode'    : ocn.HOLOGRAPH_DECODE_NONE,
                    'msg_sections'          : (8,0,48), #init sequence length, msg ID length, message length
                    'msg_type'              : ocn.MSGTYPE_FIXED_LENGTH,
                    'extrapolate_seqlen'    : 8,

                    #'extrapolate'           : 'yes',

        }, 

        # bypass for now
        'LB28-800-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-800-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 10.0 characters per second, 60.0 baud (bits per second) ',
                    'carrier_separation'    : 10,

                    'fft_filter'            : (-6.27,6.39,-6.39,6.27),
                    'fft_interpolate'       : (-4.79,6.01,-6.01,4.79),
                    'I3_parameters'         : (0.99, 0.99, 0.002, 'C-C', 0.249),
                    'downconvert_shift'     : 0.247, 
                    'parameters'            : (1500, 0.338, 0.864, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #'rotation_increments'   : 10,
                    #'extrapolate'           : 'yes',
                    #'pulse_train_sigma'     : 11.09,
                    #'pulse_start_sigma'     : 10.73,
                    #'pulse_start_envelope_sigma' : 5.99,

                    #Filter carriers
                    #'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50), #0.19
                    #'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 1, 50), #0.16
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 1, 50),

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

                    'downconvert_shift'    : 0.086,

                    # TEST CODE ONLY
                    #'I3_offsets_type'      : ocn.OFFSETS_MSGLEN_SPECIFIC,
                    #'pattern_by_msglen'    : {'16':('A-E', 0.298), '32':('B-B', 0.461), '64':('B-B', 0.461), '128':('A-A', 0.153), '256':('B-B', 0.366), '512':('B-B', 0.461)},

                    'persistent_search'     : (1, 0.95, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.90, -0.002, "yes"), #hi range, lo range, inc, scan entire range
                    #'persistent_search'     : (1, 0.80, -0.002, "yes"), #hi range, lo range, inc, scan entire range

                    'I3_parameters'         : (0.99, 0.99, 2e-3, 'C-E', 0.209), 
                    'parameters'            : (1500, 0.115, 0.028, 10000, 4, 98, 0.7072, 0.1, 0.1414, 0.01),

        }, 





        'LB28-400-I3-FC10' :{ 
                    'inherit_from'          : 'LB28-400-I3',
                    'info'                  : 'Filtered Carriers - 10Hz Wide - 10.0 characters per second, 60.0 baud (bits per second) ',
                    'carrier_separation'    : 10,
                    #'fft_filter'            : (-5.83, 5.4, -5.4, 5.83),
                    #'fft_interpolate'       : (-6.27, 1.42, -1.42, 6.27),
                    #'I3_parameters'         : (0.99, 0.99, 0.002, 'B-D', 0.042),
                    #'downconvert_shift'     : 0.814, 
                    #'parameters'            : (1500, 0.216, 0.807, 10000, 8, 98, 0.7072, 0.1, 0.1414, 0.01), 

                    #Filter carriers
                    'tx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),
                    'rx_filter'             : (ocn.FILTER_BUTTERWORTH, ocn.FILTER_BAND_PASS, 1, 4, 50),

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
