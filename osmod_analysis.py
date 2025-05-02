#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import wave
import sys
import csv
import random

from osmod_test import OsmodTest


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

class OsmodAnalysis(object):

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_INFO)
    self.debug.info_message("__init__")
    self.osmod = osmod

  def writeDataToFile(self, data):
    self.debug.info_message("writeDataToFile")

    """ discard the data if it has invalid decodes"""
    if self.osmod.has_invalid_decodes == True:
      return

    for item in data:
      self.debug.info_message("data type: " + str(type(item)))

    try:
      file_name = "osmod_results_data.csv"
      #with open(file_name, 'w', newline='') as csvfile:
      with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([data])

    except:
      self.debug.error_message("Exception in writeDataToFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def readDataFromFile(self):
    self.debug.info_message("readDataFromFile")

    data = []

    try:
      file_name = "osmod_results_data.csv"
      with open(file_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
          data.append(row)

      return data

    except:
      self.debug.error_message("Exception in readDataFromFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def generateTestData(self, form_gui, window, values, test_type, plus_minus_amount, increments, num_cycles):
    self.debug.info_message("generateTestData")

    try:
      test = OsmodTest(form_gui.osmod, window)
      mode = values['combo_main_modem_modes']
      text_num = values['combo_text_options'].split(':')[0]
      chunk_num = values['combo_chunk_options'].split(':')[0]
      amplitude = values['slider_amplitude']
      carrier_separation_override = values['slider_carrier_separation']
      self.debug.info_message("plus_minus_amount: " + str(plus_minus_amount))

      for count in range(0,int(num_cycles)):
        noise = values['btn_slider_awgn']
        random_value = random.randint(1,int(increments))
        self.debug.info_message("random_value: " + str(random_value))
        adjustment = (float(plus_minus_amount) * 2.0 * ((float(random_value) - (float(increments)/2))/ float(increments)))
        self.debug.info_message("adjustment: " + str(adjustment))
        if window['cb_enable_awgn'].get():
          noise = noise + adjustment
        test.testRoutine2(mode, form_gui, noise, text_num, chunk_num, carrier_separation_override, amplitude)

    except:
      self.debug.error_message("Exception in generateTestData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      
      
