#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import wave
import sys
import csv
import random
import FreeSimpleGUI as sg
import numpy as np
import colorsys
import ctypes
import gc
import threading
import math

from queue import Queue

from osmod_test import OsmodTest

from scipy.interpolate import splrep, splev

from scipy.stats import zscore

from osmod_c_interface import ptoc_float_array

"""
MIT License

Copyright (c) 2022-2026 Lawrence Byng

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

class OsmodSonic(object):

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None
  #continuous = True
  continuous = False

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_INFO)
    self.debug.info_message("__init__")
    self.osmod = osmod
    self.kernel_thread_initialized = False
    self.KernelQueue = Queue()
    self.kernelInitialized = False

  def getKernelQueue(self):
    return self.KernelQueue

  def pushKernelQueue(self, data):    
    self.KernelQueue.put(data)

  def popKernelQueue(self):
    return self.KernelQueue.get_nowait()

  def resetKernelQueue(self):
    self.KernelQueue = Queue()

  def isKernelQueueEmpty(self):
    return self.KernelQueue.empty()


  def send_threaded(self, window, values, form_gui):
    self.debug.info_message("send_threaded()")
    try:
      #self.t1_decoder = threading.Thread(target=self.send, args=(window, values, form_gui, ))
      if self.kernel_thread_initialized == False:
        self.t1_decoder = threading.Thread(target=self.kernelLoop, args=(window, values, form_gui, ))
        self.t1_decoder.start()
        self.kernel_thread_initialized = True
    except:
      sys.stdout.write("Exception in send_threaded: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def kernelLoop(self, window, values, form_gui):
    self.debug.info_message("kernelLoop()")
    try:

      """ initialize kernel """
      if self.kernelInitialized == False:
        self.kernelInitialized = True
        self.osmod.set_sd_blocksize_tx()
        self.osmod.set_sd_blocksize_rx()
        self.osmod.set_symbol_blocksize_rx()
        use_existing_txblocks = False
        input_device  = form_gui.window['combo_main_modem_input_device'].get()
        output_device = form_gui.window['combo_main_modem_output_device'].get()
        self.debug.info_message("input_device: " + str(input_device))
        self.debug.info_message("output_device: " + str(output_device))
        self.initialize(input_device, output_device, self.osmod.getTxSampleRate(), self.osmod.get_sd_blocksize_tx() )

        if self.continuous: #self.osmod.form_gui.window['cb_continuous_decode'].get() == True:
          self.startInputStream()

        #center_frequency = values['slider_frequency']
        #separation_override = values['slider_carrier_separation']
        #self.watch_frequency = self.calcCarrierFrequencies(center_frequency, separation_override)[0]
        #self.debug.info_message("watch_frequency: " + str(self.watch_frequency) )



      """ main kernel loop """
      kernel_loop_active = True
      while kernel_loop_active:
        if self.isKernelQueueEmpty():

          if self.continuous: #self.osmod.form_gui.window['cb_continuous_decode'].get() == True:
            block = self.getLastSdBlock() #self.getMagnitudeFrame()
            present_freq, present_mag, fft_output, data_len, fdd = self.osmod.modulation_object.getIsSignalPresent(block, self.watch_frequency + 0.5)
            self.osmod.form_gui.window['text_input_signal_magnitude_passband'].update(f"Magnitude: {present_mag:.3f}")
            self.osmod.form_gui.spectralDensityQueue.put(fdd)

          time.sleep(1)

        else:
          kernel_action = self.popKernelQueue()
          #self.debug.info_message("kernel_action: " + str(kernel_action))

          if kernel_action == ocn.KERNEL_TX_NOW:
            gc.disable()
            num_sd_blocks = self.tx_now(window, values, form_gui)
            self.startOutputStream()
            time.sleep(num_sd_blocks + 1)
            self.stopOutputStream()
            gc.enable()
            gc.collect()

          elif kernel_action == ocn.KERNEL_TX_BEACON_GENERAL:
            gc.disable()

            mode = form_gui.window['combo_main_modem_prod_modes'].get()
            locator = form_gui.window['in_locator_grid_square'].get().strip().upper()

            send_text = "BG(" + locator + ")"
            txblocks = self.osmod.createTxBlocks(mode, values, self.osmod.getSliderAwgn(),int(form_gui.window['combo_text_options'].get().split(':')[0]),self.osmod.getSliderCarrierSeparation(), self.osmod.getSliderAmplitude(), False, send_text)
            txblocks = txblocks * (2**3 - 1) / np.max(np.abs(txblocks))
            txblocks = txblocks.astype(np.float32)
            txblocks = txblocks * self.osmod.getOutputGain()
            #self.debug.info_message("getOutputGain() : " + str(self.osmod.getOutputGain()))
            self.sendTxBuffer(txblocks)
            num_sd_blocks = int(math.ceil(len(txblocks) / self.osmod.get_sd_blocksize_tx()))

            self.startOutputStream()
            time.sleep(num_sd_blocks + 1)
            self.stopOutputStream()
            gc.enable()
            gc.collect()


          elif kernel_action == ocn.KERNEL_RX_SQUELCH:
            separation_override = self.osmod.getSliderCarrierSeparation()
            self.watch_frequency = self.osmod.calcCarrierFrequencies(self.osmod.center_frequency, separation_override)[0]
            gc.disable()
            bufferStart = self.getRxBufferStart()
            self.startInputStream()
            self.rx_squelch(window, values, form_gui, bufferStart, 24)
            self.stopInputStream()
            gc.enable()
            gc.collect()
          elif kernel_action == ocn.KERNEL_TXRX_NOW:
            separation_override = self.osmod.getSliderCarrierSeparation()
            self.watch_frequency = self.osmod.calcCarrierFrequencies(self.osmod.center_frequency, separation_override)[0]
            gc.disable()
            bufferStart = self.getRxBufferStart()
            self.txrx_now(window, values, form_gui, bufferStart)

            if self.continuous: #self.osmod.form_gui.window['cb_continuous_decode'].get() == True:
              self.stopOutputStream()
            else:
              self.stopStream()

            gc.enable()
            gc.collect()


          elif kernel_action == ocn.KERNEL_TX_TIME_SYNC:
            self.tx_time_sync()
          elif kernel_action == ocn.KERNEL_RX_TIME_SYNC:
            self.rx_time_sync()
          elif kernel_action == ocn.KERNEL_EXIT:
            kernel_loop_active = False


      """ exit kernel """
      if self.kernelInitialized == True:
        self.terminate()

    except:
      sys.stdout.write("Exception in kernelLoop: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def tx_now(self, window, values, form_gui):
    self.debug.info_message("tx_now()")

    try:
      self.osmod.useProdMode()
      #mode = values['combo_main_modem_prod_modes']
      mode = form_gui.window['combo_main_modem_prod_modes'].get()

      self.osmod.setInitializationBlock(mode)
      use_existing_txblocks = False

      #self.osmod.set_sd_blocksize_tx()
      #self.osmod.set_sd_blocksize_rx()
      #self.osmod.set_symbol_blocksize_tx()
      #self.osmod.set_symbol_blocksize_rx()

      if use_existing_txblocks == False:
        #noise = values['btn_slider_awgn']
        noise = self.osmod.getSliderAwgn()

        #text_num = values['combo_text_options'].split(':')[0]
        text_num = int(form_gui.window['combo_text_options'].get().split(':')[0])

        #amplitude = values['slider_amplitude']
        amplitude = self.osmod.getSliderAmplitude()

        #carrier_separation_override = values['slider_carrier_separation']
        carrier_separation_override = self.osmod.getSliderCarrierSeparation()

        use_preset_message = form_gui.window['cb_use_preset_message'].get()
        if use_preset_message:
          txblocks = self.osmod.createTxBlocks(mode, values, noise, text_num, carrier_separation_override, amplitude, True, "")
        else:
          send_text = form_gui.window['ml_txrx_sendtext'].get()
          txblocks = self.osmod.createTxBlocks(mode, values, noise, text_num, carrier_separation_override, amplitude, False, send_text)

      txblocks = txblocks * (2**3 - 1) / np.max(np.abs(txblocks))
      txblocks = txblocks.astype(np.float32)

      txblocks = txblocks * self.osmod.getOutputGain()
      self.debug.info_message("getOutputGain() : " + str(self.osmod.getOutputGain()))

      self.sendTxBuffer(txblocks)

      #self.startStream()

      return int(math.ceil(len(txblocks) / self.osmod.get_sd_blocksize_tx()))
    except:
      sys.stdout.write("Exception in tx_now: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def txrx_now(self, window, values, form_gui, bufferStart):
    self.debug.info_message("txrx_now()")

    try:
      form_gui.TxStatusActive()

      num_sd_blocks = self.tx_now(window, values, form_gui)

      if self.continuous: #self.osmod.form_gui.window['cb_continuous_decode'].get() == False:
        self.startOutputStream()
      else:
        self.startStream()

      rxData, fdd = self.rx_squelch(window, values, form_gui, bufferStart, num_sd_blocks)

      form_gui.TxStatusInactive()

      self.decodeData(form_gui, values, rxData, fdd)

    except:
      sys.stdout.write("Exception in tx_now: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def rx_squelch(self, window, values, form_gui, bufferStart, num_sd_blocks):
    self.debug.info_message("rx_squelch()")

    try:
      #num_chars = 58
      #max_sd_blocks = int(num_chars * (self.osmod.get_symbol_blocksize_rx() / self.osmod.get_sd_blocksize_rx()))
      #for loop_count in range(max_sd_blocks):
      #  self.debug.info_message("loop_count: " + str(loop_count) )

      form_gui.RxStatusActive()

      for _ in range(num_sd_blocks):
        #if self.osmod.form_gui.window['cb_continuous_decode'].get() == False:
        block = self.getLastSdBlock()
        present_freq, present_mag, fft_output, data_len, fdd = self.osmod.modulation_object.getIsSignalPresent(block, self.watch_frequency + 0.5)
        self.osmod.form_gui.window['text_input_signal_magnitude_passband'].update(f"Magnitude: {present_mag:.3f}")

        #self.osmod.form_gui.window['text_input_signal_magnitude_passband'].update(f"{present_mag:.3f}")
        #self.form_gui.window['text_input_signal_magnitude_passband_smoothed'].update(str(self.previous_mag))
        #self.previous_mag = (present_mag/5) + (self.previous_mag * (4/5))

        #if present_mag < self.osmod.signal_squelch_value: # self.getSignalSquelch():
        #  break 

        self.osmod.form_gui.spectralDensityQueue.put(fdd)

        time.sleep(1)

      form_gui.RxStatusInactive()

      #time.sleep(num_sd_blocks + 1)

      self.debug.info_message("count complete" )

      self.debug.info_message("bufferStart: " + str(bufferStart) )

      rxData = self.getRxBuffer(bufferStart, num_sd_blocks)

      self.debug.info_message("getRxBuffer complete" )

      #self.stopStream()

      #self.processRxData(values, rxData, fdd)
      return rxData, fdd

    except:
      sys.stdout.write("Exception in send: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def decodeData(self, form_gui, values, rxData, fdd):
    self.debug.info_message("decodeData")
    try:
      form_gui.DecodeStatusActive()

      self.processRxData(values, rxData, fdd)

      form_gui.DecodeStatusInactive()

    except:
      sys.stdout.write("Exception in decodeData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def sendTxBuffer(self, txblocks):
    try:
      c_blocks = ptoc_float_array(txblocks)

      self.osmod.compiled_lib.buffer_tx_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
      self.osmod.compiled_lib.buffer_tx_data.restype = ctypes.c_int
      self.osmod.compiled_lib.buffer_tx_data(c_blocks, len(txblocks))

    except:
      sys.stdout.write("Exception in sendTxBuffer: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def getRxBufferStart(self):
    try:
      self.osmod.compiled_lib.get_intput_buffer_start.argtypes = []
      self.osmod.compiled_lib.get_intput_buffer_start.restype = ctypes.c_int
      buffer_start = self.osmod.compiled_lib.get_intput_buffer_start()

      return buffer_start
    except:
      sys.stdout.write("Exception in getRxBufferStart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def getRxBuffer(self, buffer_start, num_sd_blocks):
    try:
      self.debug.info_message("buffer_start: " + str(buffer_start))
      self.debug.info_message("num_sd_blocks: " + str(num_sd_blocks))


      #rxblocks = np.zeros((num_sd_blocks * self.osmod.getRxSampleRate(),), dtype = np.float32)
      rxblocks = np.zeros((num_sd_blocks * self.osmod.get_sd_blocksize_rx(),), dtype = np.float32)
      c_blocks = ptoc_float_array(rxblocks)

      self.osmod.compiled_lib.get_rx_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
      self.osmod.compiled_lib.get_rx_data.restype = ctypes.c_int
      self.osmod.compiled_lib.get_rx_data(c_blocks, buffer_start, num_sd_blocks)

      return rxblocks
    except:
      sys.stdout.write("Exception in getRxBuffer: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")



  def getLastSdBlock(self):
    try:
      rxblocks = np.zeros((1 * self.osmod.get_sd_blocksize_rx(),), dtype = np.float32)
      c_blocks = ptoc_float_array(rxblocks)

      self.osmod.compiled_lib.get_last_sd_block.argtypes = [ctypes.POINTER(ctypes.c_float)]
      self.osmod.compiled_lib.get_last_sd_block.restype = ctypes.c_int
      self.osmod.compiled_lib.get_last_sd_block(c_blocks)

      return rxblocks
    except:
      sys.stdout.write("Exception in getLastSdBlock: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")




  def startStream(self):
    try:
      self.osmod.compiled_lib.startStream.argtypes = []
      self.osmod.compiled_lib.startStream.restype = ctypes.c_int
      self.osmod.compiled_lib.startStream()

    except:
      sys.stdout.write("Exception in startStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def startOutputStream(self):
    try:
      self.osmod.compiled_lib.startOutputStream.argtypes = []
      self.osmod.compiled_lib.startOutputStream.restype = ctypes.c_int
      self.osmod.compiled_lib.startOutputStream()

    except:
      sys.stdout.write("Exception in startOutputStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def startInputStream(self):
    try:
      self.osmod.compiled_lib.startInputStream.argtypes = []
      self.osmod.compiled_lib.startInputStream.restype = ctypes.c_int
      self.osmod.compiled_lib.startInputStream()

    except:
      sys.stdout.write("Exception in startInputStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def stopStream(self):
    try:
      self.osmod.compiled_lib.stopStream.argtypes = []
      self.osmod.compiled_lib.stopStream.restype = ctypes.c_int
      self.osmod.compiled_lib.stopStream()

    except:
      sys.stdout.write("Exception in stopStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def stopOutputStream(self):
    try:
      self.osmod.compiled_lib.stopOutputStream.argtypes = []
      self.osmod.compiled_lib.stopOutputStream.restype = ctypes.c_int
      self.osmod.compiled_lib.stopOutputStream()

    except:
      sys.stdout.write("Exception in stopOutputStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def stopInputStream(self):
    try:
      self.osmod.compiled_lib.stopInputStream.argtypes = []
      self.osmod.compiled_lib.stopInputStream.restype = ctypes.c_int
      self.osmod.compiled_lib.stopInputStream()

    except:
      sys.stdout.write("Exception in stopInputStream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def terminate(self):
    try:
      self.osmod.compiled_lib.terminate.argtypes = []
      self.osmod.compiled_lib.terminate.restype = ctypes.c_int
      self.osmod.compiled_lib.terminate()

    except:
      sys.stdout.write("Exception in terminate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")



  def getMagnitudeFrame(self):
    try:
      self.osmod.compiled_lib.getFrame.argtypes = []
      self.osmod.compiled_lib.getFrame.restype = ctypes.c_int
      self.osmod.compiled_lib.getFrame()

    except:
      sys.stdout.write("Exception in getMagnitudeFrame: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def initialize(self, inputDeviceName, outputDeviceName, sample_rate, sd_blocksize):
    self.debug.info_message("initialize()")
    try:
      #sd_blocksize = self.osmod.get_sd_blocksize_tx()

      c_inputDeviceName = ctypes.c_char_p(inputDeviceName.encode('utf-8'))
      c_outputDeviceName = ctypes.c_char_p(outputDeviceName.encode('utf-8'))

      self.osmod.compiled_lib.initialize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
      self.osmod.compiled_lib.initialize.restype = ctypes.c_int
      self.osmod.compiled_lib.initialize(c_inputDeviceName, c_outputDeviceName, sample_rate, sd_blocksize)
    except:
      sys.stdout.write("Exception in initialize: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def processRxData(self, values, multi_block, fdd):

    try:

      if True:
        #self.osmod.modulation_object.findDecodeCandidates(multi_block)

        self.osmod.modulation_object.writeFileWavSR("sampled_audio.wav", multi_block, self.osmod.getRxSampleRate())

        save_sampled_signal_checked = self.osmod.form_gui.window['cb_savesampledsignal'].get()
        if save_sampled_signal_checked:
          sampled_signal_name = self.osmod.form_gui.window['in_sampledsignalname'].get()
          self.osmod.modulation_object.writeFileWavSR(sampled_signal_name, multi_block, self.osmod.getRxSampleRate())

        multi_block = self.osmod.getInputGain() * 0.001 * multi_block * (2**15 - 1) / np.max(np.abs(multi_block)) 

        #center_frequency = values['slider_frequency']
        #separation_override = values['slider_carrier_separation']
        separation_override = self.osmod.getSliderCarrierSeparation()


        """ adjust for doppler shift """
        multi_block = self.osmod.modulation_object.adjustFrequencyShiftAndDopplerShiftSR(multi_block, values, self.osmod.center_frequency, self.osmod.getTxSampleRate())


        """ DEBUG CODE ONLY"""
        #multi_block = self.osmod.modulation_object.alignTimePointT0(multi_block, self.osmod.getRxSampleRate(), self.osmod.getRxSymbolBlockSize())
        #self.osmod.modulation_object.alignTimePointT0(multi_block, self.osmod.getRxSampleRate(), self.osmod.getRxSymbolBlockSize())
        automatic_mode_detection = self.osmod.form_gui.window['cb_enable_automatic_mode_detection'].get()
        if automatic_mode_detection:
          multi_block = self.osmod.modulation_object.findDecodeCandidates(multi_block)



        #mode = self.osmod.getRealMode(values, self.osmod.form_gui)
        mode = self.osmod.form_gui.window['combo_main_modem_prod_modes'].get()


        self.osmod.setInitializationBlock(mode)
        frequency = self.osmod.calcCarrierFrequenciesSR(self.osmod.center_frequency, separation_override, self.osmod.getRxSampleRate())

        """ filter the input signal """
        rx_filter_params = self.osmod.rx_filter
        multi_block = self.osmod.modulation_object.apply_filterSR(multi_block, rx_filter_params, self.osmod.center_frequency, self.osmod.getRxSampleRate())

        use_hifi_rx = self.osmod.form_gui.window['cb_enable_hifi_input_sampling'].get()
        if use_hifi_rx == True:
          multi_block = scipy_signal.resample(multi_block, int(len(multi_block) * 1/6))

        """ reset the remainder """
        self.osmod.resetDecoder()

        try:
          self.osmod.decoder_callback(multi_block, frequency)
        except:
          self.debug.error_message("Exception in decodeProcessing: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


        SNR_db = self.osmod.mod_2fsk8psk.calculateSNR(multi_block, frequency)
        self.osmod.form_gui.window['text_snr_value_new'].update("SNR dB: "f"{SNR_db:.3f}")



    except:
      sys.stdout.write("Exception in processRxData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")




