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

from osmod_test import OsmodTest

from scipy.interpolate import splrep, splev

from scipy.stats import zscore


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
    self.match_table = []

  def writeDataToFile(self, data):
    self.debug.info_message("writeDataToFile")

    """ discard the data if it has invalid decodes"""
    if self.osmod.has_invalid_decodes == True:
      return

    for item in data:
      self.debug.info_message("data type: " + str(type(item)))

    try:
      file_name = self.osmod.form_gui.window['in_resultsdatafilename'].get()

      #file_name = "osmod_v0-1-0_results_data.csv"
      #with open(file_name, 'w', newline='') as csvfile:
      with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([data])

    except:
      self.debug.error_message("Exception in writeDataToFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def writeDataToFile2(self, data, file_name, write_append):
    self.debug.info_message("writeDataToFile2")

    """ discard the data if it has invalid decodes"""
    if self.osmod.has_invalid_decodes == True:
      return

    for item in data:
      self.debug.info_message("data type: " + str(type(item)))

    try:
      #file_name = self.osmod.form_gui.window['in_resultsdatafilename'].get()

      #file_name = "osmod_v0-1-0_results_data.csv"
      #with open(file_name, 'w', newline='') as csvfile:
      #with open(file_name, 'a', newline='') as csvfile:
      with open(file_name, write_append, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    except:
      self.debug.error_message("Exception in writeDataToFile2: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def readDataFromFile(self):
    self.debug.info_message("readDataFromFile")

    data = []

    try:
      file_name = self.osmod.form_gui.window['in_analysisresultsdatafilename'].get()

      row_count = 0
      with open(file_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
          self.debug.info_message("row: " + str(row))
          data.append(row)
          row_count = row_count + 1

    except:
      self.debug.error_message("Exception in readDataFromFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      self.debug.info_message("row: " + str(row))
      self.debug.info_message("row_count: " + str(row_count))

    return data


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

      randomize_pattern = form_gui.window['cb_randomizepattern'].get()

      for count in range(0,int(num_cycles)):
        if test_type == 'AWGN Factor':
          noise = values['btn_slider_awgn']
          random_value = random.randint(1,int(increments))
          self.debug.info_message("random_value: " + str(random_value))
          adjustment = (float(plus_minus_amount) * 2.0 * ((float(random_value) - (float(increments)/2))/ float(increments)))
          self.debug.info_message("adjustment: " + str(adjustment))
          if window['cb_enable_awgn'].get():
            noise = noise + adjustment
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)
        elif test_type == 'I3 Standing Wave':
          form_gui.window['cb_override_standingwaveoffsets'].update(True)
          noise = values['btn_slider_awgn']
          if count % 3 == 0:
            rand_num = random.randint(0,1000)
            form_gui.window['in_standingwavelocation'].update(rand_num/1000)

            if randomize_pattern:
              rand_num = random.randint(0,14)
              form_gui.window['combo_standingwave_pattern'].update(form_gui.combo_standingwave_pattern_options[rand_num])
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)
        elif test_type == 'I3 Pattern':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_override_standingwavepattern'].update(True)
          if count % 3 == 0:
            rand_num = random.randint(0,49)
            form_gui.window['combo_selectstandingwavepattern'].update(form_gui.combo_standingwave_patterns[rand_num])
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'RRC Alpha & T':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_override_rrc_alpha'].update(True)
          form_gui.window['cb_override_rrc_t'].update(True)
          if count % 3 == 0:
            rand_num = random.randint(0,1000)
            form_gui.window['in_rrc_alpha'].update(rand_num/1000)
            rand_num = random.randint(0,1000)
            form_gui.window['in_rrc_t'].update(rand_num/1000)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'Test Pulse Shape':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_override_rrc_alpha'].update(True)
          form_gui.window['cb_override_rrc_t'].update(True)
          if count % 3 == 0:
            rand_num = random.randint(0,len(self.osmod.test_pulse_shapes)-1)
            rrc_alpha = self.osmod.test_pulse_shapes[rand_num][0]
            rrc_T     = self.osmod.test_pulse_shapes[rand_num][1]
            form_gui.window['in_rrc_alpha'].update(rrc_alpha)
            form_gui.window['in_rrc_t'].update(rrc_T)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'Best Pulse Shape':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_override_rrc_alpha'].update(True)
          form_gui.window['cb_override_rrc_t'].update(True)
          if count % 3 == 0:
            rand_num = random.randint(0,len(self.osmod.best_pulse_shapes)-1)
            rrc_alpha = self.osmod.best_pulse_shapes[rand_num][0]
            rrc_T     = self.osmod.best_pulse_shapes[rand_num][1]
            form_gui.window['in_rrc_alpha'].update(rrc_alpha)
            form_gui.window['in_rrc_t'].update(rrc_T)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)


        elif test_type == 'Pulse Train Sigma':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_overridepulsetrainsigma'].update(True)
          if count % 3 == 0:
            #rand_num = random.randint(0,len(self.osmod.best_pulse_shapes)-1)
            #rrc_alpha = self.osmod.best_pulse_shapes[rand_num][0]
            #rrc_T     = self.osmod.best_pulse_shapes[rand_num][1]
            pulse_train_sigma = 0.1 + (random.randint(0,3000) / 100)
            form_gui.window['in_pulsetrainsigma'].update(pulse_train_sigma)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)


        #override_pulse_train_sigma = self.osmod.form_gui.window['cb_overridepulsetrainsigma'].get()
        ##if override_pulse_train_sigma:
        #  pulse_train_sigma_template = float(self.osmod.form_gui.window['in_pulsetrainsigma'].get())

        elif test_type == 'Gaussian Sigma':
          self.debug.info_message("Gaussian Sigma")

        elif test_type == 'Test Standing Wave':
          form_gui.window['cb_override_standingwaveoffsets'].update(True)
          noise = values['btn_slider_awgn']
          rand_num_1 = random.randint(0,len(self.osmod.test_sw_patterns)-1)
          sw_series  = self.osmod.test_sw_patterns[rand_num_1]
          rand_num_2 = random.randint(0,len(sw_series)-1)
          sw_type  = sw_series[rand_num_2][0]
          sw_value = sw_series[rand_num_2][1]
          form_gui.window['combo_standingwave_pattern'].update(sw_type)
          form_gui.window['in_standingwavelocation'].update(sw_value)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'Downconvert Shift':
          form_gui.window['cb_overridedownconvertshift'].update(True)
          noise = values['btn_slider_awgn']
          rand_num = random.randint(0,1000)
          form_gui.window['in_downconvertshift'].update(rand_num/1000)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'Best Pulse Shapes':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_override_rrc_alpha'].update(True)
          form_gui.window['cb_override_rrc_t'].update(True)
          rand_num_1 = random.randint(0,len(self.osmod.all_pulse_shapes)-1)
          ps_series  = self.osmod.all_pulse_shapes[rand_num_1]
          rand_num_2 = random.randint(0,len(ps_series)-1)
          rrc_alpha = ps_series[rand_num_2][0]
          rrc_T     = ps_series[rand_num_2][1]
          form_gui.window['in_rrc_alpha'].update(rrc_alpha)
          form_gui.window['in_rrc_t'].update(rrc_T)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'FEC Generator Polynomials':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_overridegeneratorpolynomials'].update(True)
          #depth = 13
          gpdepth    = random.randint(11, 19)
          #gpdepth = 15
          rand_num_1 = random.randint(1,2 ** gpdepth)
          rand_num_2 = random.randint(1,2 ** gpdepth)
          form_gui.window['in_fecgeneratorpolynomialdepth'].update(gpdepth)
          form_gui.window['in_fecgeneratorpolynomial1'].update(rand_num_1)
          form_gui.window['in_fecgeneratorpolynomial2'].update(rand_num_2)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

        elif test_type == 'Test FEC Generator Polynomials':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_overridegeneratorpolynomials'].update(True)
          rand_num_1 = random.randint(0,len(self.osmod.all_viterbi_gps)-1)
          gp_series  = self.osmod.all_viterbi_gps[rand_num_1]
          rand_num_2 = random.randint(0,len(gp_series)-1)
          gpdepth    = gp_series[rand_num_2][0]
          gp_poly_1  = gp_series[rand_num_2][1]
          gp_poly_2  = gp_series[rand_num_2][2]
          form_gui.window['in_fecgeneratorpolynomialdepth'].update(gpdepth)
          form_gui.window['in_fecgeneratorpolynomial1'].update(gp_poly_1)
          form_gui.window['in_fecgeneratorpolynomial2'].update(gp_poly_2)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)


        elif test_type == 'Best FEC Generator Polynomials':
          noise = values['btn_slider_awgn']
          form_gui.window['cb_overridegeneratorpolynomials'].update(True)
          rand_num   = random.randint(0,len(self.osmod.best_viterbi_gps)-1)
          gpdepth    = self.osmod.best_viterbi_gps[rand_num][0]
          gp_poly_1  = self.osmod.best_viterbi_gps[rand_num][1]
          gp_poly_2  = self.osmod.best_viterbi_gps[rand_num][2]
          form_gui.window['in_fecgeneratorpolynomialdepth'].update(gpdepth)
          form_gui.window['in_fecgeneratorpolynomial1'].update(gp_poly_1)
          form_gui.window['in_fecgeneratorpolynomial2'].update(gp_poly_2)
          test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)


    except:
      self.debug.error_message("Exception in generateTestData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      
      



  """
DATA_MODE                  = 0
DATA_EBN0_DB               = 1
DATA_SNR_EQUIV_DB          = 2
DATA_BER                   = 3
DATA_CPS                   = 4
DATA_BPS                   = 5
DATA_NOISE_FACTOR          = 6
DATA_SW_PATTERN_TYPE       = 7
DATA_SW_LOCATION           = 8
DATA_PRESET_PATTERN        = 9
DATA_CHUNK_SIZE            = 10
DATA_RRC_ALPHA             = 11
DATA_RRC_T                 = 12
DATA_EXTRACT_TYPE          = 13
DATA_PULSE_TRAIN_SIGMA     = 14
DATA_DETECTOR_THRESHOLD_1  = 15
DATA_DETECTOR_THRESHOLD_2  = 16
DATA_BASEBAND_FREQ_DELTA   = 17
DATA_COSTAS_DAMPING        = 18
DATA_COSTAS_LOOP_BANDWIDTH = 19
DATA_COSTAS_K1             = 20
DATA_COSTAS_K2             = 21
DATA_CALC_1                = 22
DATA_CALC_2                = 23
  """
  def compare_a_equals_b(self, a, b):
    if a == b:
      return True
    else:
      return False

  def compare_a_greater_than_b(self, a, b):
    if a > b:
      return True
    else:
      return False

  def compare_a_less_than_b(self, a, b):
    if a < b:
      return True
    else:
      return False

  def obtainDatasetValuesSingle(self, data, data_ID):
    self.debug.info_message("obtainDatasetValuesSingle")
    dataset_values = []
    try:
      for point in range(0, len(data)): 
        dataset_value = data[point][data_ID]
        if dataset_value not in dataset_values:
          dataset_values.append(dataset_value)

      return dataset_values
    except:
      self.debug.error_message("Exception in obtainDatasetValuesSingle: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def obtainDatasetValuesDouble(self, data, data_ID_1, data_ID_2):
    self.debug.info_message("obtainDatasetValuesDouble")
    dataset_values = []
    try:
      for point in range(0, len(data)): 
        dataset_value = str(data[point][data_ID_1]) + " : " + str(data[point][data_ID_2])
        if dataset_value not in dataset_values:
          dataset_values.append(dataset_value)

      return dataset_values
    except:
      self.debug.error_message("Exception in obtainDatasetValuesDouble: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def drawDotPlotCharts(self, data, chart_type, window, values, form_gui):

    self.debug.info_message("drawDotPlotCharts")
    try:
      def hsb_to_rgb(h, s, b):
        r,g,b = colorsys.hsv_to_rgb(h, s, b)
        return (int(r *255), int(g*255), int(b*255))

      self.dot_plot_subset = []

      colors = []

      legend_type = window['combo_analysis_legend'].get()
      filter1_matchtype = window['combo_filter1_matchtype'].get()
      filter_threeinarow = window['cb_analysis_three_in_a_row'].get()


      num_range_increments = 20
      if legend_type == 'AWGN Range' or legend_type == 'BER Range' or legend_type == 'BER Range All':
        hue        = float(values['slider_wave_hue'])
        saturation = float(values['slider_wave_saturation'])
        for count in range(0,num_range_increments + 1):
          color = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, (count/num_range_increments))
          colors.append(color)
      else:
        for count in range(0,300):
          rand_rgb_color = [random.randint(0,255) for _ in range(3)]
          rand_color_hex = "#{:02x}{:02x}{:02x}".format(*rand_rgb_color)
          colors.append(rand_color_hex)


      self.debug.info_message("colors: " + str(colors))

      dict_name_colors = {}
      dict_pattern_colors = {}
      dict_sw_location = {}
      dict_preset_pattern = {}
      color_count = 0
      dict_lookup = {'Eb/N0': ocn.DATA_EBN0_DB, 'BER':ocn.DATA_BER ,'BPS':ocn.DATA_BPS ,'CPS':ocn.DATA_CPS ,'Chunk Size':ocn.DATA_CHUNK_SIZE ,'SNR':ocn.DATA_SNR_EQUIV_DB, 'Noise Factor':ocn.DATA_NOISE_FACTOR, 'Pattern Type': ocn.DATA_SW_PATTERN_TYPE, 'Preset Pattern': ocn.DATA_PRESET_PATTERN, 'RRC Alpha': ocn.DATA_RRC_ALPHA, 'RRC T': ocn.DATA_RRC_T, 'AWGN': ocn.DATA_NOISE_FACTOR }

      filter_1 = window['cb_analysis_filter1'].get()
      filter_2 = window['cb_analysis_filter2'].get()
      mode_to_match = window['combo_analysis_modes'].get()
      #self.match_table = [mode_to_match]
      item_to_compare_index = dict_lookup[window['combo_analysis_itemtocompare'].get()]
      compare_operator = window['combo_analysis_campare_operator'].get()
      if filter_2:
        compare_value = float(window['in_analysis_comparewithvalue'].get())
      compare_func = None
      if compare_operator == '=':
        compare_func = self.compare_a_equals_b
      elif compare_operator == '>':
        compare_func = self.compare_a_greater_than_b
      elif compare_operator == '<':
        compare_func = self.compare_a_less_than_b

      graph = self.osmod.form_gui.window['graph_dotplotdata']
      graph.erase()

      x_max = 1300
      y_max = 400
      x_chart_offset = 130
      y_chart_offset = 60
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)


      if chart_type == 'X:Eb/N0 Y:BER':
        x_index = ocn.DATA_EBN0_DB
        y_index = ocn.DATA_BER
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:Eb/No':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:ChunkSize Y:Eb/N0':
        x_index = ocn.DATA_CHUNK_SIZE
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Chunk Size', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:BER':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_BER
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:BER Y:Eb/N0':
        x_index = ocn.DATA_BER
        y_index = ocn.DATA_EBN0_DB
        graph.draw_text('Bit Error Rate', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:AWGN Y:BER':
        x_index = ocn.DATA_NOISE_FACTOR
        y_index = ocn.DATA_BER
        graph.draw_text('AWGN', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:Rotation Y:Pulse Train Length':
        x_index = ocn.DATA_ROTATION_LO
        y_index = ocn.DATA_PULSE_TRAIN_LENGTH
        graph.draw_text('Rotation', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Pulse Train Length', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:Rotation Lo Y:Rotation Hi':
        x_index = ocn.DATA_ROTATION_LO
        y_index = ocn.DATA_ROTATION_HI
        graph.draw_text('Rotation Low', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Rotation High', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:CPS Y:Eb/N0+ABS(Eb/N0)*BER':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_CALC_1
        for point in range(0, len(data)):
          calculated_value = float(data[point][ocn.DATA_EBN0_DB]) + (abs(float(data[point][ocn.DATA_EBN0_DB]) * float(data[point][ocn.DATA_BER] )))
          data[point].append(calculated_value)
          self.debug.info_message("data[point]: " + str(data[point]))
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb/N0+ABS(Eb/N0)*BER', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:BER Y:Pulse Train Sigma':
        x_index = ocn.DATA_BER
        y_index = ocn.DATA_PULSE_TRAIN_SIGMA
        graph.draw_text('Bit Error Rate', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Pulse Train Sigma', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:Pulse Train Sigma Y:Pulse Train Length':
        x_index = ocn.DATA_PULSE_TRAIN_SIGMA
        y_index = ocn.DATA_PULSE_TRAIN_LENGTH
        graph.draw_text('Pulse Train Sigma', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Pulse Train Length', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:Eb / N0 (dB) Y:Pulse Train Sigma':
        x_index = ocn.DATA_EBN0_DB
        y_index = ocn.DATA_PULSE_TRAIN_SIGMA
        graph.draw_text('Eb / N0 (dB)', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Pulse Train Sigma', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:Pulse Train Length Y:Disposition':
        x_index = ocn.DATA_PULSE_TRAIN_LENGTH
        y_index = ocn.DATA_DISPOSITION
        graph.draw_text('Pulse Train Length', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Disposition', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:BER Y:Disposition':
        x_index = ocn.DATA_BER
        y_index = ocn.DATA_DISPOSITION
        graph.draw_text('Bit Error Rate', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Disposition', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')
      elif chart_type == 'X:DC Shift Y:BER':
        x_index = ocn.DATA_DOWNCONVERT_SHIFT
        y_index = ocn.DATA_BER
        graph.draw_text('Downconvert Shift', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Bit Error Rate', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')


      awgn_range_max = float(data[0][ocn.DATA_NOISE_FACTOR])
      awgn_range_min = float(data[0][ocn.DATA_NOISE_FACTOR])
      ber_range_max = float(data[0][ocn.DATA_BER])
      ber_range_min = float(data[0][ocn.DATA_BER])
      if legend_type == 'AWGN Range':
        for point in range(1, len(data)): 
          if (filter1_matchtype == 'Disposition ==' and data[point][ocn.DATA_DISPOSITION] == mode_to_match) or (filter1_matchtype == 'Mode Name ==' and data[point][ocn.DATA_MODE] == mode_to_match) or (filter1_matchtype == 'Pulse Train Length ==' and data[point][ocn.DATA_PULSE_TRAIN_LENGTH] in self.match_table) or (filter1_matchtype == 'Pattern Type ==' and data[point][ocn.DATA_SW_PATTERN_TYPE] == mode_to_match) or (filter1_matchtype == 'Preset Pattern ==' and data[point][ocn.DATA_PRESET_PATTERN] == mode_to_match) or (filter1_matchtype == 'AWGN ==' and data[point][ocn.DATA_NOISE_FACTOR] == mode_to_match) or (filter1_matchtype == 'Pulse Shape ==' and (str(data[point][ocn.DATA_RRC_ALPHA]) + " : " + str(data[point][ocn.DATA_RRC_T])) == mode_to_match):
            awgn_range_max = max(awgn_range_max, float(data[point][ocn.DATA_NOISE_FACTOR]))
            awgn_range_min = min(awgn_range_min, float(data[point][ocn.DATA_NOISE_FACTOR]))
      elif legend_type == 'BER Range':
        for point in range(1, len(data)): 
          if (filter1_matchtype == 'Disposition ==' and data[point][ocn.DATA_DISPOSITION] == mode_to_match) or (filter1_matchtype == 'Mode Name ==' and data[point][ocn.DATA_MODE] == mode_to_match) or (filter1_matchtype == 'Pulse Train Length ==' and data[point][ocn.DATA_PULSE_TRAIN_LENGTH] in self.match_table) or (filter1_matchtype == 'Pattern Type ==' and data[point][ocn.DATA_SW_PATTERN_TYPE] == mode_to_match) or (filter1_matchtype == 'Preset Pattern ==' and data[point][ocn.DATA_PRESET_PATTERN] == mode_to_match) or (filter1_matchtype == 'AWGN ==' and data[point][ocn.DATA_NOISE_FACTOR] == mode_to_match) or (filter1_matchtype == 'Pulse Shape ==' and (str(data[point][ocn.DATA_RRC_ALPHA]) + " : " + str(data[point][ocn.DATA_RRC_T])) == mode_to_match):
            ber_range_max = max(ber_range_max, float(data[point][ocn.DATA_BER]))
            ber_range_min = min(ber_range_min, float(data[point][ocn.DATA_BER]))
      elif legend_type == 'BER Range All':
        for point in range(1, len(data)): 
          ber_range_max = max(ber_range_max, float(data[point][ocn.DATA_BER]))
          ber_range_min = min(ber_range_min, float(data[point][ocn.DATA_BER]))


      """ allocate colors for the data """
      data_x_max = -1000000000
      data_x_min = 1000000000
      data_y_max = -1000000000
      data_y_min = 1000000000
      threeinarow_count = 0
      threeinarow_last_value = ''
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if (filter1_matchtype == 'Disposition ==' and data[point][ocn.DATA_DISPOSITION] == mode_to_match) or (filter1_matchtype == 'Mode Name ==' and data[point][ocn.DATA_MODE] == mode_to_match) or (filter1_matchtype == 'Pulse Train Length ==' and data[point][ocn.DATA_PULSE_TRAIN_LENGTH] in self.match_table) or (filter1_matchtype == 'Pattern Type ==' and data[point][ocn.DATA_SW_PATTERN_TYPE] == mode_to_match) or (filter1_matchtype == 'Preset Pattern ==' and data[point][ocn.DATA_PRESET_PATTERN] == mode_to_match) or (filter1_matchtype == 'AWGN ==' and data[point][ocn.DATA_NOISE_FACTOR] == mode_to_match) or (filter1_matchtype == 'Pulse Shape ==' and (str(data[point][ocn.DATA_RRC_ALPHA]) + " : " + str(data[point][ocn.DATA_RRC_T])) == mode_to_match):
            if filter_2:
              if compare_func(float(data[point][item_to_compare_index]), compare_value):
                include = True
              else:
                include = False
            else:
              include = True
        elif filter_2:
          if compare_func(float(data[point][item_to_compare_index]), compare_value):
            include = True
          else:
            include = False
        else:
          include = True


        if filter_threeinarow and include == True:
          include = False
          #if (legend_type == 'SW Location' and data[point][ocn.DATA_SW_LOCATION] != threeinarow_last_value):
          if (legend_type == 'SW Location' and str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION]) != threeinarow_last_value):
            threeinarow_count = 1
            #threeinarow_last_value = data[point][ocn.DATA_SW_LOCATION]
            threeinarow_last_value = str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION])
          elif (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] != threeinarow_last_value):
            threeinarow_count = 1
            threeinarow_last_value = data[point][ocn.DATA_PRESET_PATTERN]
          elif (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) != threeinarow_last_value):
            threeinarow_count = 1
            threeinarow_last_value = str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T])
          #else:
          #elif (legend_type == 'SW Location' and data[point][ocn.DATA_SW_LOCATION] == threeinarow_last_value) or (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] == threeinarow_last_value) or (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) == threeinarow_last_value):
          elif (legend_type == 'SW Location' and str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION]) == threeinarow_last_value) or (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] == threeinarow_last_value) or (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) == threeinarow_last_value):
            threeinarow_count = threeinarow_count + 1
            if threeinarow_count == 3:
              include = True
              threeinarow_count = 0


        if include:
          self.debug.info_message("data_x: " + str(float(data[point][x_index])))
          data_x_max = max(data_x_max, float(data[point][x_index]))
          data_x_min = min(data_x_min, float(data[point][x_index]))
          self.debug.info_message("data_y: " + str(float(data[point][y_index])))
          data_y_max = max(data_y_max, float(data[point][y_index]))
          data_y_min = min(data_y_min, float(data[point][y_index]))
          if legend_type == 'Mode' and data[point][ocn.DATA_MODE] not in dict_name_colors:
            dict_name_colors[data[point][ocn.DATA_MODE]] = color_count
            color_count = color_count + 1
          elif legend_type == 'Pattern Type' and data[point][ocn.DATA_SW_PATTERN_TYPE] not in dict_pattern_colors:
            dict_pattern_colors[data[point][ocn.DATA_SW_PATTERN_TYPE]] = color_count
            color_count = color_count + 1
          #elif legend_type == 'SW Location' and data[point][ocn.DATA_SW_LOCATION] not in dict_sw_location:
          elif legend_type == 'SW Location' and str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION]) not in dict_sw_location:
            dict_sw_location[str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION])] = color_count
            #dict_sw_location[data[point][ocn.DATA_SW_LOCATION]] = color_count
            color_count = color_count + 1
          elif legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] not in dict_preset_pattern:
            dict_preset_pattern[data[point][ocn.DATA_PRESET_PATTERN]] = color_count
            color_count = color_count + 1
          elif legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) not in dict_preset_pattern:
            dict_preset_pattern[str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T])] = color_count
            color_count = color_count + 1
          elif legend_type == 'Pulse Train Length' and data[point][ocn.DATA_PULSE_TRAIN_LENGTH] not in dict_preset_pattern:
            dict_preset_pattern[data[point][ocn.DATA_PULSE_TRAIN_LENGTH]] = color_count
            color_count = color_count + 1
          elif legend_type == 'Rotation Lo Hi' and 'Lo' not in dict_preset_pattern:
            dict_preset_pattern['Lo'] = color_count
            color_count = color_count + 1
            dict_preset_pattern['Hi'] = color_count
            color_count = color_count + 1
          elif legend_type == 'AWGN Range' and int((( float(data[point][ocn.DATA_NOISE_FACTOR]) - awgn_range_min) / (awgn_range_max - awgn_range_min) ) * num_range_increments ) not in dict_preset_pattern:
            color_value = int((( float(data[point][ocn.DATA_NOISE_FACTOR]) - awgn_range_min) / (awgn_range_max - awgn_range_min) ) * num_range_increments )
            dict_preset_pattern[color_value] = color_value
            color_count = color_count + 1
          elif legend_type == 'BER Range' and int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments ) not in dict_preset_pattern:
            color_value = int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments )
            dict_preset_pattern[color_value] = color_value
            color_count = color_count + 1
          elif legend_type == 'BER Range All' and int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments ) not in dict_preset_pattern:
            color_value = int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments )
            dict_preset_pattern[color_value] = color_value
            color_count = color_count + 1
          elif legend_type == 'DC Shift' and data[point][ocn.DATA_DOWNCONVERT_SHIFT] not in dict_preset_pattern:
            dict_preset_pattern[data[point][ocn.DATA_DOWNCONVERT_SHIFT]] = color_count
            color_count = color_count + 1
          elif legend_type == 'Generator Polynomials' and str(data[point][ocn.DATA_GENERATOR_POLY_DEPTH]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_1]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_2]) not in dict_preset_pattern:
            dict_preset_pattern[str(data[point][ocn.DATA_GENERATOR_POLY_DEPTH]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_1]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_2])] = color_count
            color_count = color_count + 1


          if filter_threeinarow:
            data_x_max = max(data_x_max, float(data[point-1][x_index]))
            data_x_min = min(data_x_min, float(data[point-1][x_index]))
            data_y_max = max(data_y_max, float(data[point-1][y_index]))
            data_y_min = min(data_y_min, float(data[point-1][y_index]))
            data_x_max = max(data_x_max, float(data[point-2][x_index]))
            data_x_min = min(data_x_min, float(data[point-2][x_index]))
            data_y_max = max(data_y_max, float(data[point-2][y_index]))
            data_y_min = min(data_y_min, float(data[point-2][y_index]))


      graph.draw_text("{:.4f}".format(data_x_min), location = (x_chart_offset, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.4f}".format(data_x_max), location = (x_chart_offset + x_max, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.4f}".format(data_y_min), location = (x_chart_offset -50 , y_chart_offset), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.4f}".format(data_y_max), location = (x_chart_offset -50 , y_chart_offset + y_max), angle = 0, font = '_ 12', color = 'black', text_location = 'center')


      self.debug.info_message("data_x_max: " + str(data_x_max))
      self.debug.info_message("data_y_max: " + str(data_y_max))

      if (data_x_max - data_x_min) == 0 and data_x_max > 0:
        data_x_min = 0
      if (data_x_max - data_x_min) == 0 and data_x_max < 0:
        data_x_max = 0
      if (data_y_max - data_y_min) == 0 and data_y_max > 0:
        data_y_min = 0
      if (data_y_max - data_y_min) == 0 and data_y_max < 0:
        data_y_max = 0

      x_scaling = x_max / (data_x_max - data_x_min)
      y_scaling = y_max / (data_y_max - data_y_min)


      occurrences = []
      occurrences_less_than_mid_x = []

      for _ in range(0, color_count):      
        occurrences.append(0)
        occurrences_less_than_mid_x.append(0)

      """ draw data points for the data"""
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if (filter1_matchtype == 'Disposition ==' and data[point][ocn.DATA_DISPOSITION] == mode_to_match) or (filter1_matchtype == 'Mode Name ==' and data[point][ocn.DATA_MODE] == mode_to_match) or (filter1_matchtype == 'Pulse Train Length ==' and data[point][ocn.DATA_PULSE_TRAIN_LENGTH] in self.match_table) or (filter1_matchtype == 'Pattern Type ==' and data[point][ocn.DATA_SW_PATTERN_TYPE] == mode_to_match)  or (filter1_matchtype == 'AWGN ==' and data[point][ocn.DATA_NOISE_FACTOR] == mode_to_match)  or (filter1_matchtype == 'Pulse Shape ==' and (str(data[point][ocn.DATA_RRC_ALPHA]) + " : " + str(data[point][ocn.DATA_RRC_T])) == mode_to_match):
            if filter_2:
              if compare_func(float(data[point][item_to_compare_index]), compare_value):
                include = True
              else:
                include = False
            else:
              include = True
        elif filter_2:
          if compare_func(float(data[point][item_to_compare_index]), compare_value):
            include = True
          else:
            include = False
        else:
          include = True

        if filter_threeinarow and include == True:
          include = False
          #if (legend_type == 'SW Location' and data[point][ocn.DATA_SW_LOCATION] != threeinarow_last_value):
          if (legend_type == 'SW Location' and str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION]) != threeinarow_last_value):
            threeinarow_count = 1
            #threeinarow_last_value = data[point][ocn.DATA_SW_LOCATION]
            threeinarow_last_value = str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION])
          elif (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] != threeinarow_last_value):
            threeinarow_count = 1
            threeinarow_last_value = data[point][ocn.DATA_PRESET_PATTERN]
          elif (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) != threeinarow_last_value):
            threeinarow_count = 1
            threeinarow_last_value = str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T])
          #else:
          #elif (legend_type == 'SW Location' and data[point][ocn.DATA_SW_LOCATION] == threeinarow_last_value) or (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] == threeinarow_last_value) or (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) == threeinarow_last_value):
          elif (legend_type == 'SW Location' and str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION]) == threeinarow_last_value) or (legend_type == 'Preset Pattern' and data[point][ocn.DATA_PRESET_PATTERN] == threeinarow_last_value) or (legend_type == 'RRC Alpha & T' and str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T]) == threeinarow_last_value):
            threeinarow_count = threeinarow_count + 1
            if threeinarow_count == 3:
              include = True
              threeinarow_count = 0


        if include:
          x_point = (float(data[point][x_index]) - data_x_min) * x_scaling
          y_point = (float(data[point][y_index]) - data_y_min) * y_scaling
          self.debug.info_message("x_point: " + str(x_point))
          self.debug.info_message("y_point: " + str(y_point))

          if legend_type == 'Mode':
            plot_color_index = dict_name_colors[data[point][ocn.DATA_MODE]]
          elif legend_type == 'Pattern Type':
            plot_color_index = dict_pattern_colors[data[point][ocn.DATA_SW_PATTERN_TYPE]]
          elif legend_type == 'SW Location':
            #plot_color_index = dict_sw_location[data[point][ocn.DATA_SW_LOCATION]]
            plot_color_index = dict_sw_location[str(data[point][ocn.DATA_SW_PATTERN_TYPE]) + ' : ' + str(data[point][ocn.DATA_SW_LOCATION])]
            occurrences[plot_color_index] = occurrences[plot_color_index] + 1
          elif legend_type == 'Preset Pattern':
            plot_color_index = dict_preset_pattern[data[point][ocn.DATA_PRESET_PATTERN]]
            occurrences[plot_color_index] = occurrences[plot_color_index] + 1
          elif legend_type == 'DC Shift':
            plot_color_index = dict_preset_pattern[data[point][ocn.DATA_DOWNCONVERT_SHIFT]]
            #occurrences[plot_color_index] = occurrences[plot_color_index] + 1
          elif legend_type == 'RRC Alpha & T':
            plot_color_index = dict_preset_pattern[str(data[point][ocn.DATA_RRC_ALPHA]) + ' : ' + str(data[point][ocn.DATA_RRC_T])]
            occurrences[plot_color_index] = occurrences[plot_color_index] + 1
          elif legend_type == 'Pulse Train Length':
            plot_color_index = dict_preset_pattern[data[point][ocn.DATA_PULSE_TRAIN_LENGTH]]
            occurrences[plot_color_index] = occurrences[plot_color_index] + 1
          elif legend_type == 'AWGN Range':
            plot_color_index = dict_preset_pattern[int((( float(data[point][ocn.DATA_NOISE_FACTOR]) - awgn_range_min) / (awgn_range_max - awgn_range_min) ) * num_range_increments )]
          elif legend_type == 'BER Range':
            plot_color_index = dict_preset_pattern[int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments )]
          elif legend_type == 'BER Range All':
            plot_color_index = dict_preset_pattern[int((( float(data[point][ocn.DATA_BER]) - ber_range_min) / (ber_range_max - ber_range_min) ) * num_range_increments )]
          elif legend_type == 'Generator Polynomials':
            plot_color_index = dict_preset_pattern[str(data[point][ocn.DATA_GENERATOR_POLY_DEPTH]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_1]) + ' : ' + str(data[point][ocn.DATA_GENERATOR_POLYNOMIAL_2])]
            occurrences[plot_color_index] = occurrences[plot_color_index] + 1
            if float(data[point][x_index]) - data_x_min <  float((data_x_max - data_x_min) / 2):
              occurrences_less_than_mid_x[plot_color_index] = occurrences_less_than_mid_x[plot_color_index] + 1


          if legend_type == 'Rotation Lo Hi':
            plot_color_index_lo = dict_preset_pattern['Lo']
            plot_color_index_hi = dict_preset_pattern['Hi']
            plot_color = colors[plot_color_index_lo]
            graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)
            x_point2 = (float(data[point][x_index+1]) - data_x_min) * x_scaling
            plot_color = colors[plot_color_index_hi]
            graph.draw_point((x_chart_offset + x_point2,y_chart_offset + y_point), size=8, color=plot_color)
          else:
            self.debug.info_message("plot_color_index: " + str(plot_color_index))
            plot_color = colors[plot_color_index]
            self.debug.info_message("plot_color: " + str(plot_color))
            graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

          self.dot_plot_subset.append(data[point])

          if filter_threeinarow: # and legend_type == 'RRC Alpha & T':
            x_point = (float(data[point-1][x_index]) - data_x_min) * x_scaling
            y_point = (float(data[point-1][y_index]) - data_y_min) * y_scaling
            graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

            x_point = (float(data[point-2][x_index]) - data_x_min) * x_scaling
            y_point = (float(data[point-2][y_index]) - data_y_min) * y_scaling
            graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)


      """ draw the legend at top right """
      count = 0
      if legend_type == 'Mode':
        for mode_name, color_index in dict_name_colors.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(mode_name, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'Pattern Type':
        for pattern_type_name, color_index in dict_pattern_colors.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_type_name, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'SW Location':
        debug_string = '['
        for pattern_location_name, color_index in dict_sw_location.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          debug_string = debug_string + ',(\'' + pattern_location_name.split(':')[0].strip() + '\',' + pattern_location_name.split(':')[1] + ')'
          count = count + 1
        debug_string = debug_string + ']'
        self.debug.info_message("debug_string: " + str(debug_string))
      elif legend_type == 'Preset Pattern':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'DC Shift':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'Pulse Train Length':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'RRC Alpha & T':
        debug_string = '['
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          debug_string = debug_string + ',(' + pattern_location_name.split(':')[0] + ',' + pattern_location_name.split(':')[1] + ')'
          count = count + 1
        debug_string = debug_string + ']'
        self.debug.info_message("debug_string: " + str(debug_string))
      elif legend_type == 'AWGN Range':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[count]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text( "{:.2f}".format(((awgn_range_max - awgn_range_min) * (count / num_range_increments)) + awgn_range_min), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'BER Range':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[count]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text( "{:.2f}".format(((ber_range_max - ber_range_min) * (count / num_range_increments)) + ber_range_min), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'BER Range All':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[count]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text( "{:.2f}".format(((ber_range_max - ber_range_min) * (count / num_range_increments)) + ber_range_min), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          count = count + 1
      elif legend_type == 'Rotation Lo Hi':
        for pattern_location_name, color_index in dict_preset_pattern.items():
          plot_color = colors[color_index]
          graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
          graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]), location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
          #debug_string = debug_string + ',(' + pattern_location_name.split(':')[0] + ',' + pattern_location_name.split(':')[1] + ')'
          count = count + 1
      elif legend_type == 'Generator Polynomials':
        debug_string = '['
        for pattern_location_name, color_index in dict_preset_pattern.items():
          lower_half_ratio = "{:.2f}".format(occurrences_less_than_mid_x[color_index] / occurrences[color_index])

          #best_ratio = float(pattern_location_name.split(':')[2])
          if True:
          #if float(lower_half_ratio) > 0.7:
            plot_color = colors[color_index]
            graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
            graph.draw_text(pattern_location_name + ' - ' + str(occurrences[color_index]) + '-' + lower_half_ratio, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
            count = count + 1
            debug_string = debug_string + ',(' + pattern_location_name.split(':')[0] + ',' + pattern_location_name.split(':')[1] + ',' + pattern_location_name.split(':')[2] + ')'
        debug_string = debug_string + ']'
        self.debug.info_message("debug_string: " + str(debug_string))


    
      graph_title = window['in_analysisresultscharttitle'].get()

      graph.draw_text(graph_title, location = (x_chart_offset + (x_max/2), y_chart_offset + y_max + 20), angle = 0, font = '_ 18', color = 'black', text_location = 'center')

    except:
      self.debug.error_message("Exception in drawDotPlotCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def drawPhaseCharts(self, data, chart_type, window, form_gui, canvas_name, chart_name, plot_color, erase):

    self.debug.info_message("drawPhaseCharts. data: " + str(data))
    try:
      graph = self.osmod.form_gui.window[canvas_name]

      """ multiple calls are necessary as sometimes randomly call fails silently"""
      if erase:
        graph.erase()
        graph.erase()
        graph.erase()
        graph.erase()
        graph.erase()

      x_max = 860
      y_max = 160
      x_chart_offset = 90
      y_chart_offset = 40
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)

      graph.draw_text('Time', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text('Amplitude', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')

      graph.draw_text(chart_name, location = (x_chart_offset + (x_max/2), y_chart_offset + y_max + 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')

      data_x_max = -1000000000
      data_x_min = 1000000000
      data_y_max = -1000000000
      data_y_min = 1000000000

      data_x_max = len(data)
      data_x_min = 0
      for point in range(0, len(data)): 
        data_y_max = max(data_y_max, float(data[point]))
        data_y_min = min(data_y_min, float(data[point]))

      graph.draw_text("{:.2f}".format(data_x_min), location = (x_chart_offset, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_x_max), location = (x_chart_offset + x_max - 50, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_min), location = (x_chart_offset -50 , y_chart_offset), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_max), location = (x_chart_offset -50 , y_chart_offset + y_max), angle = 0, font = '_ 12', color = 'black', text_location = 'center')

      self.debug.info_message("data_x_max: " + str(data_x_max))
      self.debug.info_message("data_y_max: " + str(data_y_max))

      if (data_x_max - data_x_min) == 0 and data_x_max > 0:
        data_x_min = 0
      if (data_x_max - data_x_min) == 0 and data_x_max < 0:
        data_x_max = 0
      if (data_y_max - data_y_min) == 0 and data_y_max > 0:
        data_y_min = 0
      if (data_y_max - data_y_min) == 0 and data_y_max < 0:
        data_y_max = 0

      x_scaling = x_max / (data_x_max - data_x_min)
      y_scaling = y_max / (data_y_max - data_y_min)
      
      x_last = (float(0) - data_x_min) * x_scaling
      y_last = (float(data[0]) - data_y_min) * y_scaling
      for point in range(1, len(data), max(1,int((len(data) / x_max)/20))): 
        x_point = (float(point) - data_x_min) * x_scaling
        y_point = (float(data[point]) - data_y_min) * y_scaling
        graph.draw_point((x_chart_offset + x_point, y_chart_offset + y_point), size=2, color=plot_color)
    except:
      self.debug.error_message("Exception in drawPhaseCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def re_drawWaveCharts(self, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2):
    self.drawWaveCharts(self.saved_data_for_chart_1, self.saved_data_for_chart_2, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2)

  """ This method works fine"""
  def drawWaveCharts(self, chart_1_data, chart_2_data, x_magnifier1, x_magnifier2, y_magnifier1, y_magnifier2):

    self.debug.info_message("drawWaveCharts")

    self.saved_data_for_chart_1 = chart_1_data
    self.saved_data_for_chart_2 = chart_2_data

    try:
      peak_y = 2
      y_offset = 0.5
      #y_offset = 0.0
      self.debug.info_message("peak y: " + str(peak_y))
      last_y1 = 0
      last_y2 = 0
      last_x = 0
      len_chart_1_data = len(chart_1_data)
      self.debug.info_message("len_chart_1_data: " + str(len_chart_1_data))
      max_x_index = int(len(chart_1_data)/x_magnifier1)
      min_x_index = max_x_index * (x_magnifier2 / 100)
      self.debug.info_message("max_x_index: " + str(max_x_index))

      step_increment = int(1000 / (max_x_index - min_x_index))
      self.debug.info_message("step_increment: " + str(step_increment))

      graph1 = self.osmod.form_gui.window['graph_wavedata1']
      graph2 = self.osmod.form_gui.window['graph_wavedata2']

      graph1.erase()
      graph2.erase()

      for x in range (0,1000, max(step_increment, 1) ):
        x_value = x 
        x_index  = int((((x + (x_magnifier2 *10))/1000) * len(chart_1_data))/x_magnifier1)
        y1_value = (((chart_1_data[ x_index ] / peak_y)*y_magnifier1) + y_offset) * 250 
        y2_value = (((chart_2_data[ x_index ] / peak_y)*y_magnifier2) + y_offset) * 250 

        self.osmod.form_gui.window['graph_wavedata1'].draw_line(point_from=(last_x,last_y1), point_to=(x_value,y1_value), width=2, color='black')
        self.osmod.form_gui.window['graph_wavedata2'].draw_line(point_from=(last_x,last_y2), point_to=(x_value,y2_value), width=2, color='black')

        last_y1 = y1_value
        last_y2 = y2_value
        last_x = x_value

    except:
      self.debug.error_message("Exception in drawWaveCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    finally:
      self.debug.info_message("Completed drawWaveCharts: ")

  def calcScaling(self, data, max_values, add_buffer):

    self.debug.info_message("calcScaling")
    try:
      x_max = max_values[0]
      y_max = max_values[1]

      data_x_max = -1000000000
      data_x_min = 1000000000
      data_y_max = -1000000000
      data_y_min = 1000000000

      data_x_max = len(data)-1
      data_x_min = 0
      for point in range(0, len(data)): 
        data_y_max = max(data_y_max, float(data[point]))
        data_y_min = min(data_y_min, float(data[point]))

      self.debug.info_message("data_x_max: " + str(data_x_max))
      self.debug.info_message("data_y_max: " + str(data_y_max))

      if (data_x_max - data_x_min) == 0 and data_x_max > 0:
        data_x_min = 0
      elif (data_x_max - data_x_min) == 0 and data_x_max < 0:
        data_x_max = 0

      if (data_y_max - data_y_min) == 0 and data_y_max > 0:
        data_y_min = 0
      elif (data_y_max - data_y_min) == 0 and data_y_max < 0:
        data_y_max = 0
      elif (data_y_max - data_y_min) == 0 and data_y_max == 0:
        data_y_max = 1.0
        data_y_min = -1.0

      x_extent = (data_x_max - data_x_min)
      y_extent = (data_y_max - data_y_min)

      """ add an optional buffer around the charts """
      if add_buffer:
        data_x_max = data_x_max + (x_extent / 10)
        data_x_min = data_x_min - (x_extent / 10)
        data_y_max = data_y_max + (y_extent / 10)
        data_y_min = data_y_min - (y_extent / 10)
        x_extent = (data_x_max - data_x_min)
        y_extent = (data_y_max - data_y_min)

      x_scaling = x_max / x_extent
      y_scaling = y_max / y_extent

      return [data_x_max, data_x_min, data_y_max, data_y_min, x_scaling, y_scaling, x_max, y_max]
  
    except:
      self.debug.error_message("Exception in calcScaling: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
  
  def calcDeltas(self, data):

    self.debug.info_message("calcDeltas")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))

      def normalizeData(data_param):
        normalized_data = []
        rolling_adjust = 0
        y_point_previous = data_param[0]

        normalized_data.append(data_param[0])
        for point in range(1, len(data_param)): 
          x_point = point
          y_point = data_param[point] + rolling_adjust

          """ in range """
          distance_1 = y_point - y_point_previous
          """ wraps at top of range """
          distance_2 = (y_point + pulse_length) - y_point_previous
          """ wraps at bottom of range """
          distance_3 = y_point - (y_point_previous  + pulse_length)

          #if abs(distance_1) < abs(distance_2) and abs(distance_1) < abs(distance_3):
          #  normalized_data.append(y_point)
          if abs(distance_2) < abs(distance_1) and abs(distance_2) < abs(distance_3):
            rolling_adjust = rolling_adjust + pulse_length
          elif abs(distance_3) < abs(distance_1) and abs(distance_3) < abs(distance_2):
            rolling_adjust = rolling_adjust - pulse_length

          normalized_data.append(y_point + rolling_adjust)

          y_point_previous = y_point + rolling_adjust

        return normalized_data

      def findOutliers(data_param):

        z_scores = np.abs(zscore(data_param))
        outliers_index = []
        outliers_value = []
        for x in range(0, len(data_param) ): 
          if z_scores[x] > 2:
            outliers_index.append(x)
            outliers_value.append(data_param[x])
        return [outliers_index, outliers_value]

      def findOutliersIQR(data_param):
        Q1 = np.percentile(data_param, 25)
        Q3 = np.percentile(data_param, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_index = []
        outliers_value = []
        for x in range(0, len(data_param) ): 
          if data_param[x] < lower_bound or data_param[x] > upper_bound:
            outliers_index.append(x)
            outliers_value.append(data_param[x])
        return [outliers_index, outliers_value]
        #return [x for x in data_param if x < lower_bound or x > upper_bound]


      """ data represents shift relative to constant velocity i.e. horizontal line"""
      normalized_data = normalizeData(data)
      self.debug.info_message("normalized_data: " + str(normalized_data))


      """ delta represents doppler shift...+ve value == stretched, -ve value == compressed"""
      delta_values = []
      y_point_previous = data[0]
      for point in range(1, len(data)): 
        x_point = point
        y_point = data[point]
        delta = y_point - y_point_previous
        delta_values.append(delta)

        y_point_previous = y_point

      self.debug.info_message("delta_values: " + str(delta_values))
      self.debug.info_message("outliers index: " + str(findOutliers(delta_values)[0]))
      self.debug.info_message("outliers value: " + str(findOutliers(delta_values)[1]))

      return findOutliers(delta_values)

    except:
      self.debug.error_message("Exception in calcDeltas: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def drawSplineChart(self, data, chart_type, window, form_gui, canvas_name, chart_name, plot_color, fixed_scale, scaling_params):

    self.debug.info_message("drawSplineChart")
    try:


      def filterData(x_param, y_param, outliers_param):
        cleaned_data_x = []
        cleaned_data_y = []
        outliers_count = 0
        outliers_len = len(outliers_param)
        for i in range(0, len(x_param)):
          if outliers_count < outliers_len and x_param[i] == outliers_param[outliers_count]:
            outliers_count = outliers_count + 1
          else:
            cleaned_data_x.append(x_param[i])
            cleaned_data_y.append(y_param[i])

        return cleaned_data_x, cleaned_data_y

      def findOutliersFractionMedianZ(data_param, divisions):
        self.debug.info_message("findOutliersFractionMedianZ()")

        fraction_indices = [0] * (divisions + 1)
        data_len = len(data_param)
        for i in range(1, divisions):
          fraction_indices[i] = int(data_len * (i/divisions))
        fraction_indices[divisions] = data_len - 1
        self.debug.info_message("fraction_indices: " + str(fraction_indices))

        all_outlier_indices = np.array([]) 
        outlier_indices = np.array([]) 
        for i in range(0, divisions):
          median_data = []
          fractional_data_param = data_param[fraction_indices[i]:fraction_indices[i+1]]
          median = np.median(fractional_data_param)
          #median_deviation = np.median(np.abs(fractional_data_param - median))
          median_deviation = np.mean(np.abs(fractional_data_param - median))
          self.debug.info_message("median_deviation: " + str(median_deviation))
        
          if median_deviation != 0:
            median_data = 0.6745 * (fractional_data_param - median) / median_deviation
            self.debug.info_message("median_data: " + str(median_data))
            #outlier_indices = np.where(np.abs(median_data) > 2.5)[0]
            outlier_indices = np.where(np.abs(median_data) >= 2)[0]

          all_outlier_indices = np.append(all_outlier_indices, outlier_indices + fraction_indices[i])
        self.debug.info_message("all_outlier_indices: " + str(all_outlier_indices))

        return [fraction_indices, all_outlier_indices]

      def findOutliersMedianZ(data_param):

        median_data = []
        median = np.median(data_param)
        median_deviation = np.median(np.abs(data_param - median))
        
        outlier_indices = np.array([])
        if median_deviation != 0:
          median_data = 0.6745 * (data_param - median) / median_deviation
          outlier_indices = np.where(np.abs(median_data) > 1.5)[0]

        return [median_data, outlier_indices]
        """
        z_scores = np.abs(zscore(data_param))
        outliers_index = []
        outliers_value = []
        for x in range(0, len(data_param) ): 
          if z_scores[x] > 2:
            outliers_index.append(x)
            outliers_value.append(data_param[x])
        return [outliers_index, outliers_value]
        """

      #outliers = findOutliersFractionMedianZ(data, 4)
      #outliers = findOutliersFractionMedianZ(data, 2)
      outliers = findOutliersMedianZ(data)

      self.debug.info_message("median data: " + str(outliers[0]))
      self.debug.info_message("outlier indices: " + str(outliers[1]))

      spline_smoothing = float(self.osmod.form_gui.window['in_analysissplinesmoothvalue'].get())

      x = np.linspace(0, len(data)-1, len(data))


      filtered_x, filtered_y = filterData(x, data, outliers[1])
      self.debug.info_message("filtered_x: " + str(filtered_x))
      self.debug.info_message("filtered_y: " + str(filtered_y))

      """
      self.debug.info_message("removing outliers for spline")
      self.debug.info_message("x: " + str(x))
      cleaned_data_x = []
      cleaned_data_y = []
      outliers_count = 0
      outliers_len = len(outliers[0])
      ignore_next_outlier = False
      for i in range(0, len(x)):
        if outliers_count < outliers_len and x[i] == outliers[0][outliers_count]:
          if ignore_next_outlier == False:
            ignore_next_outlier = True
          else:
            cleaned_data_x.append(x[i])
            cleaned_data_y.append(data[i])
            ignore_next_outlier = False

          outliers_count = outliers_count + 1

        else:
          cleaned_data_x.append(x[i])
          cleaned_data_y.append(data[i])
          ignore_next_outlier = False
        
        
      self.debug.info_message("cleaned_data_x: " + str(cleaned_data_x))
      self.debug.info_message("cleaned_data_y: " + str(cleaned_data_y))
      """

      interpolation_type = self.osmod.form_gui.window['combo_splinetype'].get() 

      if self.osmod.sample_rate == 48000:
        pulses_per_offset = int(self.osmod.pulses_per_block / 8)
      #x_smooth = np.linspace(-1, len(data), (len(data) + 1) * pulses_per_offset)
      x_smooth = np.linspace(-1/2, len(data) - (1/2), (len(data) * pulses_per_offset) + 1 )

      if interpolation_type == 'B-Spline':
        y_smooth = self.osmod.modulation_object.BetaSplineCurveInterpolation(x, x_smooth, data, spline_smoothing)
      elif interpolation_type == 'Cubic-Spline':
        y_smooth = self.osmod.modulation_object.CubicSplineCurveInterpolation(x, x_smooth, data, spline_smoothing)
      elif interpolation_type == 'Pchip':
        y_smooth = self.osmod.modulation_object.PchipCurveInterpolation(x, x_smooth, data, spline_smoothing)
      elif interpolation_type == 'Chebyshev':
        y_smooth = self.osmod.modulation_object.chebyshevCurveInterpolation(x, x_smooth, data, spline_smoothing)

      self.drawPulseTrainCharts(y_smooth, chart_type, window, form_gui, canvas_name, chart_name, plot_color, False, fixed_scale, scaling_params)


    except:
      self.debug.error_message("Exception in drawSplineChart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def drawPulseTrainCharts(self, data, chart_type, window, form_gui, canvas_name, chart_name, plot_color, erase, fixed_scale, scaling_params):

    self.debug.info_message("drawPulseTrainCharts. data: " + str(data))
    try:
      dot_size = 5

      graph = self.osmod.form_gui.window[canvas_name]

      """ multiple calls are necessary as sometimes randomly call fails silently"""
      if erase:
        graph.erase()
        graph.erase()
        graph.erase()
        graph.erase()
        graph.erase()

      x_max = 860
      y_max = 420
      x_chart_offset = 90
      y_chart_offset = 40
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)

      graph.draw_text('Time', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text('Amplitude', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')

      graph.draw_text(chart_name, location = (x_chart_offset + (x_max/2), y_chart_offset + y_max + 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')

      # [data_x_max, data_x_min, data_y_max, data_y_min, x_scaling, y_scaling, x_max, y_max]
      if fixed_scale:
        self.debug.info_message("use fixed scale")
        data_x_max = scaling_params[0]
        data_x_min = scaling_params[1]
        data_y_max = scaling_params[2]
        data_y_min = scaling_params[3]
        x_scaling  = scaling_params[4]
        y_scaling  = scaling_params[5]
      else:
        self.debug.info_message("calculating scale")
        data_x_max = -1000000000
        data_x_min = 1000000000
        data_y_max = -1000000000
        data_y_min = 1000000000

        data_x_max = len(data)-1
        data_x_min = 0
        for point in range(0, len(data)): 
          data_y_max = max(data_y_max, float(data[point]))
          data_y_min = min(data_y_min, float(data[point]))


      graph.draw_text("{:.2f}".format(data_x_min), location = (x_chart_offset, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_x_max), location = (x_chart_offset + x_max - 10, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_min), location = (x_chart_offset -50 , y_chart_offset), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_y_max), location = (x_chart_offset -50 , y_chart_offset + y_max), angle = 0, font = '_ 12', color = 'black', text_location = 'center')

      self.debug.info_message("data_x_max: " + str(data_x_max))
      self.debug.info_message("data_y_max: " + str(data_y_max))

      if not fixed_scale:
        if (data_x_max - data_x_min) == 0 and data_x_max > 0:
          data_x_min = 0
        elif (data_x_max - data_x_min) == 0 and data_x_max < 0:
          data_x_max = 0

        if (data_y_max - data_y_min) == 0 and data_y_max > 0:
          data_y_min = 0
        elif (data_y_max - data_y_min) == 0 and data_y_max < 0:
          data_y_max = 0
        elif (data_y_max - data_y_min) == 0 and data_y_max == 0:
          data_y_max = 1.0
          data_y_min = -1.0

        x_scaling = x_max / (data_x_max - data_x_min)
        y_scaling = y_max / (data_y_max - data_y_min)
      
      #x_last = (float(0) - data_x_min) * x_scaling
      #y_last = (float(data[0]) - data_y_min) * y_scaling
      for point in range(0, len(data), max(1,int((len(data) / x_max)/20))): 
        x_point = (float(point) - data_x_min) * x_scaling
        y_point = (float(data[point]) - data_y_min) * y_scaling
        graph.draw_point((x_chart_offset + x_point, y_chart_offset + y_point), size=dot_size, color=plot_color)
    except:
      self.debug.error_message("Exception in drawPulseTrainCharts: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
