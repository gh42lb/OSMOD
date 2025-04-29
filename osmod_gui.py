#!/usr/bin/env python


import json
import time
import sys
import select
import constant as cn
import osmod_constant as ocn
import threading
import debug as db
import FreeSimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import platform
import os

from socket import socket, AF_INET, SOCK_STREAM
from app_pipes import AppPipes
from osmod_main import osModem
from queue import Queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.signal import find_peaks
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

class FormGui(object):

  window = None
  plotQueue = Queue()
  drawitQueue = Queue()
  spectralDensityQueue = Queue()

  txwindowQueue = Queue()

  debug = db.Debug(ocn.DEBUG_OSMOD)

  """
  debug level 0=off, 1=info, 2=warning, 3=error
  """
  def __init__(self, group_arq, debug):  
    self.osmod = osModem(self)
    return

  def plotWaveCanvasPrepare(self, N, data, canvas):
    self.debug.info_message("plotWaveCanvas")
    self.debug.info_message("num points to plot: " + str(N))
    self.debug.info_message("data len: " + str(len(data)))
    plt.figure(figsize=(4,4))
    time = np.linspace(-N/2, N/2, N, endpoint=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    fig, ax = plt.subplots()
    ax.set_xlim(-N/2, N/2)
    ax.plot(time, data, label = 'Wave Data')

    return fig

  def plotWaveCanvasDraw(self, fig, canvas):
    figure_canvas = FigureCanvasTkAgg(fig, canvas)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)


  """
  create the main GUI window
  """

  def createMainTabbedWindow(self, text, js):

    combo_server_or_client = 'Server Listen,Client Connect'.split(',')
    combo_modem_modes  = 'LB28-20-100,LB28-5-10,LB28-10-40,LB28-10-20,\
LB28-2.5-10I,LB28-1.25-10I,LB28-0.625-10I,LB28-0.3125-10I,LB28-0.15625-10I'.split(',')

    combo_noise_modes  = '2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10'.split(',')

    combo_chunk_options  = '5,10,15,20,25,30,35,40,45,50'.split(',')

    combo_align_options  = '5,10,15,20,25,30,35,40,45,50,100'.split(',')

    combo_separation_options  = '5,10,15,20,25,30,35,40,45,50,100'.split(',')

    combo_text_options  = '0:peter piper,1:jack be nimble,2:row row row,3:hickory dickory,4:its raining,5:jack and jill,6:humpty dumpty,7:wise old owl,8:hey diddle diddle,9:baa baa,10:twinkle twinkle,11:on a boat,12:queen of hearts'.split(',')

    sg.theme('DarkGrey8')

    os_name = os.name
    self.debug.info_message("os_name: " + str(os_name))
    platform_system = platform.system()
    self.debug.info_message("platform_system: " + str(platform_system))
    platform_info = platform.uname()
    self.debug.info_message("platform_info: " + str(platform_info))
    node_name = platform.node()
    self.debug.info_message("node_name: " + str(node_name))
    sys_platform = sys.platform
    self.debug.info_message("sys_platform: " + str(sys_platform))
    machine_name = platform.uname().machine
    self.debug.info_message("machine_name: " + str(machine_name))
    system_name = platform.uname().system
    self.debug.info_message("system_name: " + str(system_name))

    optimize_library_name = 'lb28_compiled_' + str(sys_platform) + '_' + str(node_name)
    directory = str(os.getcwd())
    full_library_name = str(directory) +'/' + str(optimize_library_name) + '.so'
    self.debug.info_message("attempting to load optimization library: " + full_library_name)

    self.osmod.use_compiled_c_code = False

    try:
      self.osmod.compiled_lib = ctypes.CDLL(full_library_name)
      self.debug.info_message("lb28 compiled library: " + full_library_name + " loaded successfully.")
      self.loaded_compiled_library = True
    except:
      """ fallback option...aka plan B """
      try:
        optimize_library_name = 'lb28_compiled_' + str(sys_platform) + '_' + str(machine_name)
        full_library_name = str(directory) +'/' + str(optimize_library_name) + '.so'
        self.osmod.compiled_lib = ctypes.CDLL(full_library_name)
        self.debug.info_message("lb28 compiled library: " + full_library_name + " loaded successfully.")
        self.loaded_compiled_library = True

      except:
        self.debug.error_message("Unable to load lb28 compiled library: " + full_library_name + '. Error:' + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
        self.loaded_compiled_library = False

    if self.loaded_compiled_library == True:
      #combo_code_options  = 'interpreted python,compiled c'.split(',')
      combo_code_options  = 'compiled c,interpreted python'.split(',')
      self.osmod.use_compiled_c_code = True
    else:
      combo_code_options  = 'interpreted python'.split(',')


    self.layout_charts = [

                        [ sg.Frame('Wave Data 1', [

                          [sg.Slider(range=(1,1000), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart1_xmag')],
                          [sg.Slider(range=(1,99), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart2_xmag')],

                          [sg.Graph(key='graph_wavedata1', canvas_size = (900, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False),

                           sg.Slider(range=(0.1,1000), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart1_ymag')],

                        ], size=(1010, 270) )],

                        [ sg.Frame('Wave Data 2', [


                          [sg.Graph(key='graph_wavedata2', canvas_size = (900, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False),

                           sg.Slider(range=(0.1,1000), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart2_ymag')],

                        ], size=(1010, 270) )],


                        ] 


    self.layout_rxtx = [
                          [
                           sg.Combo(combo_modem_modes, key='combo_main_modem_modes', default_value=combo_modem_modes[8], enable_events=True),
                           sg.Button('CW encoder', size=(11, 1), key='btn_cw_start', visible = False),
                           sg.Button('init output stream', size=(11, 1), key='btn_init_ostream', visible = False),
                           sg.Button('START', size=(5, 1), key='btn_init_test', visible = False),
                           sg.Button('STOP', size=(5, 1), key='btn_stop8pskdecoder', visible = False),
                           sg.Button('TEST', size=(5, 1), key='btn_testit', visible = False),
                           sg.Button('Run Test', size=(11, 1), key='btn_testit2'),
                           sg.Button('draw plot', size=(11, 1), key='btn_canvasdrawplotwaveform', visible = False),
                           sg.Button('start 8psk decoder', size=(11, 1), key='btn_8pskdecoder', visible = False),
                           sg.CBox('Enable AWGN', key='cb_enable_awgn', default=True ),
                           sg.Text('Code Options: ')  ,
                           sg.Combo(combo_code_options, key='combo_code_options', default_value=combo_code_options[0], enable_events=True),
                           sg.Text('Text Options: ')  ,
                           sg.Combo(combo_text_options, key='combo_text_options', default_value=combo_text_options[0], enable_events=True)],
                          [sg.Text('Chunk Size: ')  ,
                           sg.Combo(combo_chunk_options, key='combo_chunk_options', default_value=combo_chunk_options[5], enable_events=True),
                           sg.CBox('Align 1st Carrier', key='cb_enable_align', default=True ),
                           sg.Combo(combo_align_options, key='option_carrier_alignment', default_value=combo_align_options[3] ),
                           sg.CBox('Override carrier separation', key='cb_enable_separation_override', default=True ),
                           sg.Combo(combo_separation_options, key='option_separation_options', default_value=combo_separation_options[2] )],


                        [sg.Frame('AWGN Factor', [

                          [
                           sg.Slider(range=(0,10), default_value = 7.8, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='btn_slider_awgn')],
 
                        ], size=(1000, 60) )],

                          [sg.Text('Eb/N0 (dB): -----', size=(30, 1), key='text_ebn0db_value' ) ,
                           sg.Text('BER: -----', size=(30, 1), key='text_ber_value')  ,
                           sg.Text('Eb/N0: -----', size=(30, 1), key='text_ebn0_value' ) ,
                           sg.Text('SNR (dB): -----', size=(30, 1), key='text_snr_value' )] ,

                          [
                            sg.MLine('', size=(64, 5), font=("Courier New", 9), key='ml_txrx_sendtext', text_color='black', background_color='white', expand_x = True, expand_y=False, disabled = False)], 

                          [
                            sg.MLine('', size=(64, 5), font=("Courier New", 9), key='ml_txrx_recvtext', text_color='black', background_color='white', expand_x = True, expand_y=False, disabled = False)], 



                        [sg.Frame('Chart 1', [

                          [sg.Canvas(key='canvas_waveform', size=(100, 50), expand_x=False, expand_y=False)],

                        ], size=(500, 250) ),

                         sg.Frame('Chart 2', [

                          [sg.Canvas(key='canvas_second_waveform', size=(11, 100), expand_x=False, expand_y=False)],

                        ], size=(500, 250) )],

                        [ sg.Frame('Density plot', [

                          [ sg.Text('0   100   200   300   400   500   600   700   800  900 1000 1100 1200 1300 1400 1500  1600 1700 1800 1900 2000 2100 2200 2300  2400 2500 2600 2700 2800 2900 3000' )] ,

                          [sg.Graph(key='graph_density', canvas_size = (970, 200), graph_bottom_left=(0,0), graph_top_right = (3000, 100), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1010, 400), visible=False )],



                        ] 


    self.tabgrp = [

                       [#sg.Button('About', size=(9, 1), key='btn_about'),
                        sg.Button('Exit')],
                       [sg.TabGroup([[
                             sg.Tab('Test Options', self.layout_rxtx, title_color='Blue',border_width =10, background_color='Gray' ),
                             sg.Tab('Charts', self.layout_charts, title_color='Blue',border_width =10, background_color='Gray' )]],
                       tab_location='centertop',
                       title_color='Blue', tab_background_color='Dark Gray', background_color='Dark Gray', size=(1010, 550), selected_title_color='Black', selected_background_color='White', key='tabgrp_main' )]]  

    self.window = sg.Window("OSMOD v0.0.3 Alpha - Test and Reference Code for LB28 Modulation", self.tabgrp, default_element_size=(40, 1), grab_anywhere=False, disable_close=True)                       

    return (self.window)

  def runReceive(self, window, dispatcher):
    self.form_events = dispatcher
    try:
      while True:
        event, values = self.window.read(timeout=100)
       
        try:
          dispatcher.dispatch[event](self.form_events, window, values, self)
        except:
          dispatcher.event_catchall(window, values, self)

        if event in ('Exit', None):
          break

      dispatcher.event_exit_receive(values)
      self.window.close()
    except:
      sys.stdout.write("Exception in runReceive: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    self.window.close()


class OliviaTxRx(object):

  def runTx(self, window):
    oltx.__main__(window)
    return()

  def runRx(self, window):
    olrx.__main__(window)
    return()


class ReceiveControlsProc(object):

  window_initialized = False
  chart_timer = 0
  
  def __init__(self):  
    self.debug = db.Debug(cn.DEBUG_INFO)
    return

  def event_catchall(self, window, values, form_gui):

    if(self.window_initialized == False and form_gui.window != None):
      self.window_initialized = True		

    if self.window_initialized == True:

      if (self.chart_timer % 5) == 0:
        window['graph_density'].Move(0,1)
        if (self.chart_timer) == 0:
          for x in range(0,3000, 100):
            window['graph_density'].draw_line(point_from=(x,0), point_to=(x,100), width=2, color='black')
          for y in range(0,100, 20):
            window['graph_density'].draw_line(point_from=(0,y), point_to=(3000,y), width=2, color='black')
        elif (self.chart_timer % 100) == 0:
          window['graph_density'].draw_line(point_from=(0,0), point_to=(3000,0), width=2, color='black')
        else:
          for x in range(0,3000, 100):
            window['graph_density'].draw_line(point_from=(x,0), point_to=(x,1), width=2, color='black')

        while form_gui.txwindowQueue.empty() == False:
          txdata = form_gui.txwindowQueue.get_nowait()
          window['ml_txrx_sendtext'].print(str(txdata), end="", text_color='black', background_color = 'white')

        if form_gui.spectralDensityQueue.empty() == False:
          while form_gui.spectralDensityQueue.empty() == False:
            data = form_gui.spectralDensityQueue.get_nowait()

          strong_freqs = form_gui.osmod.mod_psk.getStrongestFrequency(data)
          window['graph_density'].draw_line(point_from=(int(strong_freqs),0), point_to=(int(strong_freqs),1), width=5, color='blue')

      self.chart_timer = self.chart_timer + 1


      if form_gui.drawitQueue.empty() == False:
        tuple_data = form_gui.drawitQueue.get_nowait()
        fig = tuple_data[0]
        canvas_name = tuple_data[1]
        form_gui.plotWaveCanvasDraw(fig, window[canvas_name].tk_canvas)

    return()

  def event_rxtxsend(self, window, values, form_gui):
    sys.stdout.write("event_rxtxsend\n")

    oliv_txrx = OliviaTxRx()
    t1 = threading.Thread(target=oliv_txrx.runTx, args=(window, ))
    t1.start()

    return()

  def event_rxtxrecv(self, window, values, form_gui):
    sys.stdout.write("event_rxtxrecv\n")

    oliv_txrx = OliviaTxRx()
    t1 = threading.Thread(target=oliv_txrx.runRx, args=(window, ))
    t1.start()

    return()


  def event_canvasdrawplotwaveform(self, window, values, form_gui):
    sys.stdout.write("event_canvasdrawplotwaveform\n")

    if(self.window_initialized == True and form_gui.window != None):
      data = form_gui.osmod.mod_psk.modulateChunk8PSK(form_gui.osmod.carrier_frequency_reference, [0,1,1, 0, 1, 0, 1, 1, 1, 0, 1, 1]) 
      t1 = threading.Thread(target=form_gui.osmod.mod_psk.plotWaveCanvas, args=(2400, data, window['canvas_waveform'].tk_canvas, ))
      t1.start()

  def event_8pskdecoder(self, window, values, form_gui):
    sys.stdout.write("event_8pskdecoder\n")

    form_gui.osmod.startDecoder("8psk", window)

  def event_stop8pskdecoder(self, window, values, form_gui):
    sys.stdout.write("event_stop8pskdecoder\n")
    form_gui.osmod.stopEncoder()
    form_gui.osmod.stopDecoder()

  def event_sliderchart1xmag(self, window, values, form_gui):
    sys.stdout.write("event_sliderchart1xmag\n")
    chart1_xmag = values['slider_chart1_xmag']
    chart1_ymag = values['slider_chart1_ymag']
    chart2_xmag = values['slider_chart2_xmag']
    chart2_ymag = values['slider_chart2_ymag']
    form_gui.osmod.demodulation_object.re_drawWaveCharts(chart1_xmag, chart2_xmag, chart1_ymag, chart2_ymag)

  def event_testit(self, window, values, form_gui):
    sys.stdout.write("event_testit\n")
    test = OsmodTest(form_gui.osmod, window)
    mode = values['combo_main_modem_modes']
    test.testInterpolate(mode)

  def event_testit2(self, window, values, form_gui):
    sys.stdout.write("event_testit\n")
    test = OsmodTest(form_gui.osmod, window)
    mode = values['combo_main_modem_modes']
    noise = values['btn_slider_awgn']
    text_num = values['combo_text_options'].split(':')[0]
    chunk_num = values['combo_chunk_options'].split(':')[0]

    test.testRoutine2(mode, form_gui, noise, text_num, chunk_num)


  def event_codeoptions(self, window, values, form_gui):
    sys.stdout.write("event_codeoptions\n")
    temp = values['combo_code_options']
    if temp == 'compiled c':
      form_gui.osmod.use_compiled_c_code = True
      sys.stdout.write("setting for compiled C code\n")
    else:
      form_gui.osmod.use_compiled_c_code = False
      sys.stdout.write("setting for interpreted python code\n")
    sys.stdout.write("code option: " + form_gui.osmod.code_option + "\n")


  def event_inittest(self, window, values, form_gui):
    sys.stdout.write("event_inittest\n")

    """ start encoder / modulator """
    form_gui.osmod.startEncoder("HELLO!", "8psk")

    """ start decoder / demodulator """
    form_gui.osmod.startDecoder("8psk", window)


  def event_initostream(self, window, values, form_gui):
    sys.stdout.write("event_initostream\n")
    try:
      sample_rate = 44100
      bandwidth = 1000
      symbols = 32
      freq_sep = bandwidth/symbols
      tone_sep = int(np.ceil(sample_rate/freq_sep))
      form_gui.osmod.initOutputStream(sample_rate, tone_sep)
    except:
      sys.stdout.write("Exception in event_initostream: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_cwstart(self, window, values, form_gui):
    sys.stdout.write("event_cwstart\n")

    try:
      morse = MorseEncoder()

    except:
      self.debug.error_message("Exception in event_cwstart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  def event_exit_receive(self, values):

    try:
      sys.stdout.write("IN event_exit_receive\n")

    except:
      sys.stdout.write("Exception in event_exit_receive: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")
      
    return()




  dispatch = {
      'btn_rxtx_send'    : event_rxtxsend,
      'btn_rxtx_recv'    : event_rxtxrecv,
      'btn_cw_start'     : event_cwstart,
      'btn_init_ostream' : event_initostream,
      'btn_init_test'    : event_inittest,
      'btn_canvasdrawplotwaveform' : event_canvasdrawplotwaveform,
      'btn_8pskdecoder'  : event_8pskdecoder,
      'btn_stop8pskdecoder'  : event_stop8pskdecoder,
      'btn_testit'       : event_testit,
      'btn_testit2'       : event_testit2,
      'slider_chart1_xmag' :  event_sliderchart1xmag,
      'slider_chart1_ymag' :  event_sliderchart1xmag,
      'slider_chart2_xmag' :  event_sliderchart1xmag,
      'slider_chart2_ymag' :  event_sliderchart1xmag,
      'combo_code_options' :  event_codeoptions,
  }


def main():

  debug = db.Debug(cn.DEBUG_INFO)

  """ create the main gui controls event handler """
  form_gui = FormGui(None, None)
  window = form_gui.createMainTabbedWindow('', None)
  dispatcher = ReceiveControlsProc()

  t2 = threading.Thread(target=form_gui.runReceive, args=(window, dispatcher,))
  t2.start()

  while True:
    if form_gui.plotQueue.empty() == False:
      sys.stdout.write("Preparing plot on main thread\n")
      tuple_data = form_gui.plotQueue.get_nowait()
      data = tuple_data[0]
      canvas_name = tuple_data[1]
      fig = form_gui.plotWaveCanvasPrepare(len(data), data, None)
      form_gui.drawitQueue.put((fig, canvas_name))
    time.sleep(2)

if __name__ == '__main__':
    main()




