#!/usr/bin/env python

import time
import debug as db
import constant as cn
import osmod_constant as ocn
import wave
import sys
import csv
import random
import numpy as np
import colorsys
import FreeSimpleGUI as sg

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

class OsmodSimulator(object):

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_INFO)
    self.debug.info_message("__init__")
    self.osmod = osmod
    self.sample_with_noise = None
    self.noise_free_sample = None
    self.have_sample_data = False

  """ generate a sample data set using the osmod modulation routines for LB28-6400 mode"""
  def generateSampleData(self, values):
    self.debug.info_message("generateSampleData")

    try:
      self.osmod.startTimer('init')

      mode = 'LB28-6400-64-2-15-I'

      """ initialize the block"""
      self.osmod.setInitializationBlock(mode)

      """ figure out the carrier frequencies"""
      center_frequency = 1400
      #carrier_separation_override = '15'
      carrier_separation_override = values['slider_carrier_separation']
      frequency = self.osmod.calcCarrierFrequencies(center_frequency, carrier_separation_override)
      self.debug.info_message("center frequency: " + str(center_frequency))
      self.debug.info_message("carrier frequencies: " + str(frequency))

      text = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
      self.debug.info_message("encoding text: " + str(text))

      bit_groups, sent_bitstring = self.osmod.text_encoder(text)
      data2 = self.osmod.modulation_object.modulate(frequency, bit_groups)

      """ write to file """
      self.debug.info_message("size of signal data: " + str(len(data2)))
      self.osmod.modulation_object.writeFileWav(mode + "_simulation.wav", data2)

      """ read file """
      audio_array = self.osmod.modulation_object.readFileWav(mode + "_simulation.wav")
      self.debug.info_message("audio data type: " + str(audio_array.dtype))
      total_audio_length = len(audio_array)

      """ add noise for testing..."""
      amplitude = values['slider_amplitude']
      noise_free_signal = audio_array*0.00001 * float(amplitude)   #* 0.7

      noise_value = values['btn_slider_awgn']
      self.debug.info_message("noise value: " + str(noise_value))
      value = float(noise_value)

      self.debug.info_message("adding noise")

      audio_array = noise_free_signal
      if self.window['cb_enable_awgn'].get():
        audio_array = self.osmod.modulation_object.addAWGN(audio_array, value, frequency)
      if self.window['cb_enable_timing_noise'].get():
        audio_array = self.osmod.modulation_object.addTimingNoise(audio_array)
      if self.window['cb_enable_phase_noise'].get():
        audio_array = self.osmod.modulation_object.addPhaseNoise2(audio_array)

      self.debug.info_message("generate sample data complete")

      self.sample_with_noise = audio_array
      self.noise_free_sample = noise_free_signal


    except:
      self.debug.error_message("Exception in generateSampleData: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def drawSimulation(self, chart_type, window, values, form_gui):

    self.debug.info_message("drawSimulation")
    try:
      self.window = window
      if self.have_sample_data == False:
        self.have_sample_data = True
        self.generateSampleData(values)

      pi_char = '\u03C0'

      wave_scale_factor = float(values['slider_wave_scale'])
      hue        = float(values['slider_wave_hue'])
      saturation = float(values['slider_wave_saturation'])

      self.debug.info_message("creating simulation. please wait....")

      def hsb_to_rgb(h, s, b):
        r,g,b = colorsys.hsv_to_rgb(h, s, b)
        return (int(r *255), int(g*255), int(b*255))

      color1  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.0)
      color2  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.05)
      color3  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.1)
      color4  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.15)
      color5  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.2)
      color6  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.25)
      color7  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.3)
      color8  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.35)
      color9  = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.4)
      color10 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.45)
      color11 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.5)
      color12 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.55)
      color13 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.6)
      color14 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.65)
      color15 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.7)
      color16 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.75)
      color17 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.8)
      color18 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.85)
      color19 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.9)
      color20 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 0.95)
      color21 = '#%02x%02x%02x' % hsb_to_rgb(hue, saturation, 1.0)

      num_colors = 20

      color = '#%02x%02x%02x' % hsb_to_rgb(0.5, 0.7, 0.1)

      #colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'indigo', 'violet', 'black']
      colors = [color1, color2, color3, color4, color5, color6, color7, color8, color9, color10, color11, color12, color13, color14, color15, color16, color17, color18, color19, color20, color21]
      #colors = []

      self.debug.info_message("creating simulation. please wait 2....")

      """
      for count in range(0,30):
        rand_rgb_color = [random.randint(0,255) for _ in range(3)]
        rand_color_hex = "#{:02x}{:02x}{:02x}".format(*rand_rgb_color)
        colors.append(rand_color_hex)

      self.debug.info_message("colors: " + str(colors))

      dict_name_colors = {}
      color_count = 0
      dict_lookup = {'Eb/N0': ocn.DATA_EBN0_DB, 'BER':ocn.DATA_BER ,'BPS':ocn.DATA_BPS ,'CPS':ocn.DATA_CPS ,'Chunk Size':ocn.DATA_CHUNK_SIZE ,'SNR':ocn.DATA_SNR_EQUIV_DB, 'Noise Factor':ocn.DATA_NOISE_FACTOR, 'Amplitude':ocn.DATA_AMPLITUDE }

      filter_1 = window['cb_analysis_filter1'].get()
      filter_2 = window['cb_analysis_filter2'].get()
      mode_to_match = window['combo_analysis_modes'].get()
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
      """

      graph = self.osmod.form_gui.window['graph_simulation1']
      graph.erase()

      """ virtual chart bounds within canvas """
      x_max = 1300
      y_max = 380
      x_chart_offset = 300
      y_chart_offset = 80
      twiddle = 5

      box_color = 'cyan'

      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset - twiddle,y_chart_offset + y_max + twiddle), point_to=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle, y_chart_offset + y_max + twiddle), point_to=(x_chart_offset  + x_max + twiddle,y_chart_offset - twiddle), width=2, color=box_color)
      graph.draw_line(point_from=(x_chart_offset + x_max + twiddle ,y_chart_offset - twiddle), point_to=(x_chart_offset - twiddle,y_chart_offset - twiddle), width=2, color=box_color)

      self.debug.info_message("creating simulation. please wait 3....")


      plot_points = [[0, 0, 1], [10,10,2], [20,20,3]]

      modulated_wave_signal = np.array( [] )

      num_samples = self.osmod.symbol_block_size
      time = np.arange(num_samples) / self.osmod.sample_rate
      frequency1 = 200
      phase1 = 0

      term5 = 2 * np.pi * time
      term6 = term5 * frequency1
      #term7 = term5 * frequency[1]
      term8 = term6 + phase1
      #term9 = term7 + phase2
      #symbol_wave1 = np.cos(term8) + np.sin(term8)

      symbol_wave1 = self.noise_free_sample[0:1000]

      wave_max = np.max(symbol_wave1)
      wave_min = np.min(symbol_wave1)
      ratio = wave_max - wave_min
      #symbol_wave1 = (symbol_wave1 * ratio) + wave_min
      #symbol_wave2 = self.amplitude * np.cos(term9) + self.amplitude * np.sin(term9)
      #symbol_wave = self.osmod.symbol_wave_function([symbol_wave1, symbol_wave2])
      #symbol_wave = self.osmod.symbol_wave_function([symbol_wave1, symbol_wave2])

      """ define plot canvas"""
      data_x_max = len(symbol_wave1)
      data_x_min = 0
      data_y_max = len(symbol_wave1)
      data_y_min = 0

      source_x = [50, 50, 50, 50, 50]
      source_y = [750, 500, 250, 1000, 0]
      #source_1_x = 50
      #source_1_y = 750
      #source_2_x = 50
      #source_2_y = 500
      #source_3_x = 50
      #source_3_y = 250

      x_scaling = (x_max / (data_x_max - data_x_min)) * 0.9
      y_scaling = y_max / (data_y_max - data_y_min)


      font_size = 18
      #graph.draw_point((x_chart_offset + (source_x[3] -10 - data_x_min) * x_scaling, y_chart_offset + (source_y[3] - data_y_min) * y_scaling), size=8, color=source_color)
      #graph.draw_point((x_chart_offset + (source_x[4] -10 - data_x_min) * x_scaling, y_chart_offset + (source_y[4] - data_y_min) * y_scaling), size=8, color=source_color)

      """ draw y axis indents"""
      x_labels_location = 65
      graph.draw_text("t", location = (x_chart_offset - x_labels_location, y_chart_offset + (source_y[1] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t+" + pi_char + "/2", location = (x_chart_offset - x_labels_location, y_chart_offset + (source_y[0] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t-" + pi_char + "/2", location = (x_chart_offset - x_labels_location, y_chart_offset + (source_y[2] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t+" + pi_char, location = (x_chart_offset - x_labels_location, y_chart_offset + (source_y[3] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t-" + pi_char, location = (x_chart_offset - x_labels_location, y_chart_offset + (source_y[4] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')

      """ draw x axis indents"""
      y_labels_location = 15
      graph.draw_text("t", location = (x_chart_offset + (x_max/2) + (0) * x_scaling, y_chart_offset - y_labels_location), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t+" + pi_char + "/2", location = (x_chart_offset + (x_max/2) + (250) * x_scaling, y_chart_offset - y_labels_location), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t+" + pi_char, location = (x_chart_offset + (x_max/2) + (500) * x_scaling, y_chart_offset - y_labels_location), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t-" + pi_char + "/2", location = (x_chart_offset + (x_max/2) - (250) * x_scaling, y_chart_offset - y_labels_location), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text("t-" + pi_char, location = (x_chart_offset + (x_max/2) - (500) * x_scaling, y_chart_offset - y_labels_location), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')

      """ axes labels"""
      graph.draw_text('Time per unit wavelength Signal (Source Wave)', location = (x_chart_offset + (x_max/2), y_chart_offset - 30), angle = 0, font = ('Arial', font_size), color = 'black', text_location = 'center')
      graph.draw_text('Time per unit wavelength Observer', location = (x_chart_offset - 180,y_chart_offset + (y_max/2)), angle = 90, font = ('Arial', font_size), color = 'black', text_location = 'center')

      """ legend intensity """
      count = 0
      for color_index in range(0, len(colors)):
        plot_color = colors[color_index]
        graph.draw_point((x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
        graph.draw_text("{:.2f}".format(color_index/20), location = (x_chart_offset + x_max + 75, y_chart_offset + y_max - (count*14)), angle = 0, font = ('Arial', font_size), color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
        count = count + 1


      self.debug.info_message("creating simulation. please wait 4....")


      #if chart_type == 'Tx Waveform':
      def drawAngleLine(angle_delta):
        for point in range(0, len(symbol_wave1)): 

          """ plot line at angle """
          x_point = (source_x[1] + point * (np.cos(angle_delta) ) - data_x_min) * x_scaling
          y_point = (source_y[1] + point * (np.sin(angle_delta) ) - data_y_min) * y_scaling
          plot_color_index = ((symbol_wave1[point] - wave_min) / ratio) * num_colors
          plot_color = colors[int(plot_color_index)]

          if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
            if x_chart_offset + x_point > x_chart_offset and y_chart_offset + y_point > y_chart_offset:
              graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

      def drawAngleLineFromCoordinates():
        for x in range(0,700):
          y = x * 0.3
          x_point = ((source_x[1] + x) - data_x_min) * x_scaling
          y_point = ((source_y[1] + y) - data_y_min) * y_scaling
          distance = int(np.sqrt((x*x) + (y*y)))
           
          plot_color_index = ((symbol_wave1[distance] - wave_min) / ratio) * num_colors
          plot_color = colors[int(plot_color_index)]

          if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
            if x_chart_offset + x_point > x_chart_offset and y_chart_offset + y_point > y_chart_offset:
              graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

      def drawBlockFromCoordinatesIntraTriple():
        ref_point = 1
        points_offset_angle_proxy = 250
        #wave_scale_factor = 0.5
        for x in range(0,1000,2):
          for y in range(-500,500,2):
          #for y in range(0,400):
            x_point = ((source_x[ref_point] + x) - data_x_min) * x_scaling
            #x_point = ((source_x[0] + x) - data_x_min) * x_scaling
            y_point = ((source_y[ref_point] + y) - data_y_min) * y_scaling
            #y_point = ((source_y[1] + y - 200) - data_y_min) * y_scaling
            
            #distance4 = - points_offset_angle_proxy + int(np.sqrt((x*x) + ((y-500)*(y-500))))
            distance1 = points_offset_angle_proxy   + int(np.sqrt(((x-500)*(x-500)) + ((y-250)*(y-250))))
            distance2 = int(np.sqrt(((x-500)*(x-500)) + (y*y)))
            distance3 = -points_offset_angle_proxy  + int(np.sqrt(((x-500)*(x-500)) + ((y+250)*(y+250))))
            #distance5 = points_offset_angle_proxy   + int(np.sqrt((x*x) + ((y+500)*(y+500))))
            value1 = symbol_wave1[int(distance1 * wave_scale_factor) % 999]
            value2 = symbol_wave1[int(distance2 * wave_scale_factor) % 999]
            value3 = symbol_wave1[int(distance3 * wave_scale_factor) % 999]
            #value4 = symbol_wave1[int(distance4 * wave_scale_factor) % 999]
            #value5 = symbol_wave1[int(distance5 * wave_scale_factor) % 999]
            #combined_value = (value1+value2+value3+value4+value5) / 5           
            combined_value = (value1+value2+value3) / 3

            plot_color_index = ((combined_value - wave_min) / ratio) * num_colors
            plot_color = colors[int(plot_color_index)]

            if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
              if x_chart_offset + x_point > x_chart_offset and y_chart_offset + y_point > y_chart_offset:
                graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

      def drawBlockFromCoordinatesSingle():
        ref_point = 1
        points_offset_angle_proxy = 250
        for x in range(0,1000):
          for y in range(-500,500):
            x_point = ((source_x[ref_point] + x) - data_x_min) * x_scaling
            y_point = ((source_y[ref_point] + y) - data_y_min) * y_scaling
            
            distance2 = int(np.sqrt((x*x) + (y*y)))
            value2 = symbol_wave1[int(distance2 * wave_scale_factor) % 999]
            combined_value = value2

            plot_color_index = ((combined_value - wave_min) / ratio) * num_colors
            plot_color = colors[int(plot_color_index)]

            if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
              if x_chart_offset + x_point > x_chart_offset and y_chart_offset + y_point > y_chart_offset:
                graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)


      if chart_type == 'Intra Triple':
        drawBlockFromCoordinatesIntraTriple()
      if chart_type == 'Single':
        drawBlockFromCoordinatesSingle()


      """ plot the source points """
      source_color = 'blue'
      graph.draw_point((x_chart_offset + (source_x[0] +500 - data_x_min) * x_scaling, y_chart_offset + (source_y[0] - data_y_min) * y_scaling), size=8, color=source_color)
      graph.draw_point((x_chart_offset + (source_x[1] +500 - data_x_min) * x_scaling, y_chart_offset + (source_y[1] - data_y_min) * y_scaling), size=8, color=source_color)
      graph.draw_point((x_chart_offset + (source_x[2] +500 - data_x_min) * x_scaling, y_chart_offset + (source_y[2] - data_y_min) * y_scaling), size=8, color=source_color)
      graph.draw_text("A", location = (x_chart_offset + (source_x[0] +500 - 20 - data_x_min) * x_scaling, y_chart_offset + (source_y[0] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'cyan', text_location = 'center')
      graph.draw_text("B", location = (x_chart_offset + (source_x[0] +500 - 20 - data_x_min) * x_scaling, y_chart_offset + (source_y[1] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'cyan', text_location = 'center')
      graph.draw_text("C", location = (x_chart_offset + (source_x[0] +500 - 20 - data_x_min) * x_scaling, y_chart_offset + (source_y[2] - data_y_min) * y_scaling), angle = 0, font = ('Arial', font_size), color = 'cyan', text_location = 'center')


      #drawAngleLineFromCoordinates()
      #drawAngleLine(np.pi/4)
      #drawAngleLine(np.pi/8)
      #drawAngleLine(0)
      #drawAngleLine(-np.pi/4)
      #drawAngleLine(-np.pi/8)

      """
      for point in range(0, len(symbol_wave1)): 
        angle_delta = np.pi/4
        x_point = (source_2_x + point * (np.cos(angle_delta) ) - data_x_min) * x_scaling
        y_point = (source_2_y + point * (np.sin(angle_delta) ) - data_y_min) * y_scaling
        plot_color_index = ((symbol_wave1[point] - wave_min) / ratio) * 8
        plot_color = colors[int(plot_color_index)]

        if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
          graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)
        angle_delta = 0
        x_point = (source_2_x + point * (np.cos(angle_delta) ) - data_x_min) * x_scaling
        y_point = (source_2_y + point * (np.sin(angle_delta) ) - data_y_min) * y_scaling
        plot_color_index = ((symbol_wave1[point] - wave_min) / ratio) * 8
        plot_color = colors[int(plot_color_index)]

        if x_chart_offset + x_point < x_chart_offset + x_max and y_chart_offset + y_point < y_chart_offset + y_max:
          graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)
      """

        #plot_color = 'blue'
        #self.debug.info_message("plot_color: " + str(plot_color))
        #self.debug.info_message("x_point: " + str(x_point))
        #self.debug.info_message("y_point: " + str(y_point))
        #self.debug.info_message("plot_color_index: " + str(plot_color_index))
        #x_point = (float(plot_points[point][0]) - data_x_min) * x_scaling
        #y_point = (float(plot_points[point][1]) - data_y_min) * y_scaling
        #plot_color_index = plot_points[point][2]


      """
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
      elif chart_type == 'X:CPS Y:Eb/N0+ABS(Eb/N0)*BER':
        x_index = ocn.DATA_CPS
        y_index = ocn.DATA_CALC_1
        for point in range(0, len(data)):
          calculated_value = float(data[point][ocn.DATA_EBN0_DB]) + (abs(float(data[point][ocn.DATA_EBN0_DB]) * float(data[point][ocn.DATA_BER] )))
          data[point].append(calculated_value)
          self.debug.info_message("data[point]: " + str(data[point]))
        graph.draw_text('Characters Per Second', location = (x_chart_offset + (x_max/2), y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
        graph.draw_text('Eb/N0+ABS(Eb/N0)*BER', location = (x_chart_offset - 50,y_chart_offset + (y_max/2)), angle = 90, font = '_ 12', color = 'black', text_location = 'center')

      """

      """

      data_x_max = -1000000000
      data_x_min = 1000000000
      data_y_max = -1000000000
      data_y_min = 1000000000
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if data[point][ocn.DATA_MODE] == mode_to_match:
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

        if include:
          self.debug.info_message("data_x: " + str(float(data[point][x_index])))
          data_x_max = max(data_x_max, float(data[point][x_index]))
          data_x_min = min(data_x_min, float(data[point][x_index]))
          self.debug.info_message("data_y: " + str(float(data[point][y_index])))
          data_y_max = max(data_y_max, float(data[point][y_index]))
          data_y_min = min(data_y_min, float(data[point][y_index]))
          if data[point][ocn.DATA_MODE] not in dict_name_colors:
            dict_name_colors[data[point][ocn.DATA_MODE]] = color_count
            color_count = color_count + 1

      """

      """

      graph.draw_text("{:.2f}".format(data_x_min), location = (x_chart_offset, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
      graph.draw_text("{:.2f}".format(data_x_max), location = (x_chart_offset + x_max, y_chart_offset - 20), angle = 0, font = '_ 12', color = 'black', text_location = 'center')
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
      
      for point in range(0, len(data)): 
        include = False
        if filter_1:
          if data[point][ocn.DATA_MODE] == mode_to_match:
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

        if include:
          x_point = (float(data[point][x_index]) - data_x_min) * x_scaling
          y_point = (float(data[point][y_index]) - data_y_min) * y_scaling
          self.debug.info_message("x_point: " + str(x_point))
          self.debug.info_message("y_point: " + str(y_point))
          plot_color_index = dict_name_colors[data[point][ocn.DATA_MODE]]
          self.debug.info_message("plot_color_index: " + str(plot_color_index))
          plot_color = colors[plot_color_index]
          self.debug.info_message("plot_color: " + str(plot_color))
          graph.draw_point((x_chart_offset + x_point,y_chart_offset + y_point), size=8, color=plot_color)

      count = 0
      for mode_name, color_index in dict_name_colors.items():
        plot_color = colors[color_index]
        graph.draw_point((x_chart_offset + x_max + 25, y_chart_offset + y_max - (count*14)), size=16, color=plot_color)
        graph.draw_text(mode_name, location = (x_chart_offset + x_max + 50, y_chart_offset + y_max - (count*14)), angle = 0, font = '_ 12', color = 'black', text_location = sg.TEXT_LOCATION_LEFT)
        count = count + 1

      """
      self.debug.info_message("simulation complete")

      graph.draw_text('LB28 Interpolated Mode - Wave Intensity \ Phase Amplification', location = (x_chart_offset + (x_max/2), y_chart_offset + y_max + 20), angle = 0, font = '_ 18', color = 'black', text_location = 'center')

    except:
      self.debug.error_message("Exception in drawSimulation: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


