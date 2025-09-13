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
#import matplotlib.pyplot as plt
import ctypes
import platform
import os
import random

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
    #plt.figure(figsize=(8,4))
    time = np.linspace(-N/2, N/2, N, endpoint=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    fig, ax = plt.subplots()
    ax.set_xlim(-N/2, N/2)
    ax.plot(time, data, label = 'Phase Data')

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
    combo_modem_modes  = 'LB28-2-2-100-N,LB28-160-2-2-100-N,LB28-240-2-2-100-N,LB28-2-2-100-N,LB28-160-4-2-100-N,LB28-160-4-2-50-N,LB28-4-2-40-N,LB28-4-2-20-N,LB28-320-8-2-50-N,LB28-8-2-10-N,LB28-16-2-15-I,LB28-16-2-10-I,LB28-3200-32-2-15-I,\
LB28-32-2-10-I,LB28-6400-64-2-15-I,LB28-6400-64-2-15-I3S3,LB28-6400-64-2-15-I3F,LB28-6400-64-2-15-I3E8,LB28-12800-128-2-37-I3E8-FEC,LB28-3200-32-2-37-I3E8-FEC,LB28-1600-16-2-37-I3E8-FEC,LB28-800-8-2-37-I3E8-FEC,LB28-6400-64-2-37-I3E8,LB28-6400-64-2-37-I3E8-FEC,LB28-25600-256-2-37-I3E8-FEC,LB28-51200-512-2-37-I3E8-FEC,LB28-102400-1024-2-37-I3E8-FEC,LB28-64-2-15-I,LB28-64-2-10-I,LB28-6400-128-2-15-I,LB28-128-2-15-I,LB28-128-2-10-I,LB28-25600-256-2-15-I,LB28-256-2-15-I,LB28-256-2-10-I,\
LB28-25600-512-2-15-I,LB28-51200-512-2-15-I,LB28-512-2-15-I,LB28-512-2-10-I,LB28-51200-1024-2-15-I,LB28-102400-1024-2-15-I,LB28-1024-2-15-I,LB28-1024-2-10-I,LB28-102400-2048-2-15-I,LB28-204800-2048-2-15-I,LB28-2048-2-15-I,LB28-2048-2-10-I'.split(',')
    self.combo_modem_modes = combo_modem_modes

    combo_analysis_chart_options = 'X:BER Y:Eb/N0,X:Eb/N0 Y:BER,X:CPS Y:Eb/No,X:ChunkSize Y:Eb/N0,X:CPS Y:BER,X:CPS Y:Eb/N0+ABS(Eb/N0)*BER,X:AWGN Y:BER,X:Rotation Y:Pulse Train Length,X:Rotation Lo Y:Rotation Hi,X:BER Y:Pulse Train Sigma,X:Pulse Train Sigma Y:Pulse Train Length,X:Eb / N0 (dB) Y:Pulse Train Sigma,X:Pulse Train Length Y:Disposition,X:BER Y:Disposition,X:DC Shift Y:BER'.split(',')

    combo_simulator_chart_options = 'Intra Triple,Single'.split(',')

    combo_analysis_compareitem_options = 'Eb/N0,BER,BPS,CPS,Chunk Size,SNR,Noise Factor,Pattern Type,AWGN'.split(',')

    combo_standingwave_pattern_options = 'A-B,A-C,A-D,A-E,B-C,B-D,B-E,C-D,C-E,D-E,A-A,B-B,C-C,D-D,E-E'.split(',')
    self.combo_standingwave_pattern_options = combo_standingwave_pattern_options

    combo_standingwave_patterns = 'Pattern 1,Pattern 2,Pattern 3,Pattern 4,Pattern 5,Pattern 6,Pattern 7,Pattern 8,Pattern 9,Pattern 10,\
Pattern 11,Pattern 12,Pattern 13,Pattern 14,Pattern 15,Pattern 16,Pattern 17,Pattern 18,Pattern 19,Pattern 20,Pattern 21,\
Pattern 22,Pattern 23,Pattern 24,Pattern 25,Pattern 26,Pattern 27,Pattern 28,Pattern 29,Pattern 30,Pattern 31,Pattern 32,\
Pattern 33,Pattern 34,Pattern 35,Pattern 36,Pattern 37,Pattern 38,Pattern 39,Pattern 40,Pattern 41,Pattern 42,Pattern 43,\
Pattern 44,Pattern 45,Pattern 46,Pattern 47,Pattern 48,Pattern 49,Pattern 50'.split(',')

    self.combo_standingwave_patterns = combo_standingwave_patterns

    combo_legend_options = 'Mode,Pattern Type,SW Location,Preset Pattern,RRC Alpha & T,AWGN Range,Rotation Lo Hi,Pulse Train Length,BER Range,BER Range All,DC Shift,Generator Polynomials'.split(',')

    combo_filter1_matchtypes = 'Mode Name ==,Pattern Type ==,Preset Pattern ==,AWGN ==,Pulse Shape ==,Pulse Train Length ==,Disposition =='.split(',')

    #combo_plotpulsetraintypes = 'Test,Smoothed Signal,Block Offsets,Block Offsets Half,Block Offsets Fourth,Block Offsets Fourth b,Interpolated Pulse Offsets,Dominant Series 1'.split(',')
    combo_plotpulsetraintypes = 'Refresh'.split(',')

    combo_test_routines = 'Calculate Phase Angles,Interpolation,Calculate Rotation Tables'.split(',')

    combo_analysis_compare_operator = '=,>,<'.split(',')

    combo_chart_options = 'Before Mean Average,After Mean Average,Both,FFT,EXP,Phase Error,Frequency & EXP,EXP Intra Triple,Chart Data Dictionary'.split(',')

    combo_splinecharttypes = 'B-Spline,Cubic-Spline,Pchip,Chebyshev'.split(',')

    combo_dopplercurvefittypes = 'B-Spline,Cubic-Spline,Pchip,Chebyshev'.split(',')

    combo_downconvert_options = 'costas_loop,fast'.split(',')

    combo_noise_modes  = '2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10'.split(',')

    combo_intra_combine_options = 'Type 1,Type 2,Type 3,Type 4,Type 5,Type 6,Type 7,Type 8,Type 9,Type 10'.split(',')

    combo_intra_extract_options = 'Type 1,Type 2,Type 3,Type 4,Type 5,Type 6,Type 7'.split(',')

    combo_chunk_options  = '5,10,15,20,25,30,35,40,45,50'.split(',')

    combo_align_options  = '5,10,15,20,25,30,35,40,45,50,100'.split(',')

    combo_separation_options  = '5,10,15,20,25,30,35,40,45,50,100'.split(',')

    combo_text_options  = '0:cq,1:cqcq,2:cqcqcq,3:peter piper,4:jack be nimble,5:row row row,6:hickory dickory,7:its raining,8:jack and jill,9:humpty dumpty,10:wise old owl,11:hey diddle diddle,12:baa baa,13:twinkle twinkle,14:on a boat,15:queen of hearts'.split(',')

    combo_random_choice_options = 'AWGN Factor,I3 Standing Wave,I3 Pattern,RRC Alpha & T,Gaussian Sigma,Test Pulse Shape,Best Pulse Shape,Pulse Train Sigma,Test Standing Wave,Downconvert Shift,Best Pulse Shapes,FEC Generator Polynomials,Test FEC Generator Polynomials,Best FEC Generator Polynomials'.split(',')

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


    about_text = '\n\
                      OSMOD de WH6GGO v0.1.0 Alpha - Open Source Modem Test and Reference Platform for LB28 Modulation.  \n\
\n\
\n\
\n\
MIT License\n\
\n\
\n\
Permission is hereby granted, free of charge, to any person obtaining a copy\n\
of this software and associated documentation files (the "Software"), to deal\n\
in the Software without restriction, including without limitation the rights\n\
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
copies of the Software, and to permit persons to whom the Software is\n\
furnished to do so, subject to the following conditions:\n\
\n\
The above copyright notice and this permission notice shall be included in all\n\
copies or substantial portions of the Software.\n\
\n\
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n\
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n\
SOFTWARE.\n\
\n\
'                                                                         




    self.layout_phase_charts = [


                        [sg.Frame('Chart 1', [

                          #[sg.Canvas(key='chart_canvas_oneoffour', size=(100, 50), expand_x=True, expand_y=False)],
                          [sg.Graph(key='chart_canvas_oneoffour', canvas_size = (1200, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1200, 220) )],

                        [sg.Frame('Chart 2', [

                          #[sg.Canvas(key='chart_canvas_twooffour', size=(11, 100), expand_x=True, expand_y=False)],
                          [sg.Graph(key='chart_canvas_twooffour', canvas_size = (1200, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1200, 220) )],

                        [sg.Frame('Chart 3', [

                          #[sg.Canvas(key='chart_canvas_twooffour', size=(11, 100), expand_x=True, expand_y=False)],
                          [sg.Graph(key='chart_canvas_threeoffour', canvas_size = (1200, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1200, 220) )],

                        [sg.Frame('Chart 4', [

                          #[sg.Canvas(key='chart_canvas_twooffour', size=(11, 100), expand_x=True, expand_y=False)],
                          [sg.Graph(key='chart_canvas_fouroffour', canvas_size = (1200, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1200, 220) )],


                        ] 


    self.layout_wave_charts = [

                        [ sg.Frame('Wave Data 1', [

                          [sg.Slider(range=(1,1000), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart1_xmag')],
                          [sg.Slider(range=(1,99), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart2_xmag')],

                          [sg.Graph(key='graph_wavedata1', canvas_size = (1100, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False),

                           sg.Slider(range=(0.01,100), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart1_ymag')],

                        ], size=(1200, 270) )],

                        [ sg.Frame('Wave Data 2', [


                          [sg.Graph(key='graph_wavedata2', canvas_size = (1100, 200), graph_bottom_left=(0,0), graph_top_right = (1000, 250), background_color='white', expand_x=False, expand_y=False),

                           sg.Slider(range=(0.01,100), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart2_ymag')],

                        ], size=(1200, 270) )],


                        ] 


    self.layout_pulsetrain_charts = [

                        [sg.Button('Plot Chart', size=(10, 1), key='btn_plot_pulsetrainchart'),
                         sg.Combo(combo_plotpulsetraintypes, key='combo_plotpulsetraintype', default_value=combo_plotpulsetraintypes[0], enable_events=True),
                         sg.Button('Erase Chart', size=(10, 1), key='btn_erasechart'),
                         sg.Button('Spline', size=(10, 1), key='btn_plotsplinechart'),
                         sg.Combo(combo_splinecharttypes, key='combo_splinetype', default_value=combo_splinecharttypes[3], enable_events=True),
                         sg.InputText('12', key='in_analysissplinesmoothvalue', size=(20, 1), enable_events=True)],




                        [ sg.Frame('Pulse Train Data 1', [

                          #[sg.Slider(range=(1,1000), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart1_xmag')],
                          #[sg.Slider(range=(1,99), default_value = 1, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_chart2_xmag')],

                          [sg.Graph(key='graph_pulsetraindata1', canvas_size = (1100, 430), graph_bottom_left=(0,0), graph_top_right = (1000, 500), background_color='white', expand_x=False, expand_y=False)],

                           #sg.Slider(range=(0.01,100), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart1_ymag')],

                        ], size=(1200, 460) )],

                        [ sg.Frame('Pulse Train Data 2', [


                          [sg.Graph(key='graph_pulsetraindata2', canvas_size = (1100, 430), graph_bottom_left=(0,0), graph_top_right = (1000, 500), background_color='white', expand_x=False, expand_y=False)],

                           #sg.Slider(range=(0.01,100), default_value = 25, orientation='v', resolution=0.01, expand_y = True, enable_events = True, key='slider_chart2_ymag')],

                        ], size=(1200, 460) )],


                        ] 


    self.layout_analysis_charts = [

                        [ sg.Frame('Alanlysis_data', [

                          [sg.Button('Plot Results', size=(10, 1), key='btn_plot_results'),
                           sg.Combo(combo_analysis_chart_options, key='option_analysis_chart_types', default_value=combo_analysis_chart_options[0] ),

                           sg.Frame('Filter 1', [
                           [sg.CBox('', key='cb_analysis_filter1', default=False ),
                           sg.Combo(combo_filter1_matchtypes, key='combo_filter1_matchtype', default_value=combo_filter1_matchtypes[0], enable_events=True),
                           sg.Combo(combo_modem_modes, key='combo_analysis_modes', default_value=combo_modem_modes[0], enable_events=True, readonly=True),
                           sg.Button('Add', key='btn_analysisaddmatchtable'),
                           sg.Button('Clear', key='btn_analysisclearmatchtable')],
                           ], size=(580, 50) )],


                          [sg.Frame('Filter 2', [
                           [sg.CBox('', key='cb_analysis_filter2', default=False ),
                           sg.Combo(combo_analysis_compareitem_options, key='combo_analysis_itemtocompare', default_value=combo_analysis_compareitem_options[0], enable_events=True),
                           sg.Combo(combo_analysis_compare_operator, key='combo_analysis_campare_operator', default_value=combo_analysis_compare_operator[0], enable_events=True),
                           sg.InputText('', key='in_analysis_comparewithvalue', size=(20, 1), enable_events=True)],
                           ], size=(380, 50) ),
                           sg.Text('Filter 1 Values: ', key='text_analysisfilter1values')],


                          [sg.Text('Legend: '),
                           sg.Combo(combo_legend_options, key='combo_analysis_legend', default_value=combo_legend_options[0], enable_events=True),
                           sg.CBox('filter true if 3 in a row', key='cb_analysis_three_in_a_row', default=False ),
                           sg.InputText('osmod_v0-1-0_results_data.csv', key='in_analysisresultsdatafilename', size=(50, 1), enable_events=True),
                           sg.FileBrowse('Select File', file_types=(('CSV Files', '*.csv'), ('All Files', '*.*'))),
                           sg.Button('Save Subset', size=(10, 1), key='btn_savedotplotsubset')],

                          [sg.InputText('LB28 Modulation Over AWGN Channel', key='in_analysisresultscharttitle', size=(50, 1), enable_events=True)],

                          [sg.Graph(key='graph_dotplotdata', canvas_size = (1200, 700), graph_bottom_left=(0,0), graph_top_right = (2000, 500), background_color='white', expand_x=False, expand_y=True)],

                        ], size=(1200, 750), expand_y=True )],

                        ] 


    self.layout_simulator_charts = [

                        [ sg.Frame('Simulations', [

                          [sg.Button('Plot Results', size=(10, 1), key='btn_plot_simulation'),
                           sg.Combo(combo_simulator_chart_options, key='option_simulator_chart_types', default_value=combo_simulator_chart_options[0] ),
                           sg.Text('Hue: ')  ,
                           sg.Slider(range=(0.01,1), default_value = 0.5, orientation='h', resolution=0.01, expand_x = False, enable_events = True, key='slider_wave_hue'),
                           sg.Text('Saturation: ')  ,
                           sg.Slider(range=(0.01,1), default_value = 0.5, orientation='h', resolution=0.01, expand_x = False, enable_events = True, key='slider_wave_saturation')],
                          [sg.Slider(range=(0.001,4), default_value = 0.101, orientation='h', resolution=0.001, expand_x = True, enable_events = True, key='slider_wave_scale')],

                          [sg.Graph(key='graph_simulation1', canvas_size = (1200, 900), graph_bottom_left=(0,0), graph_top_right = (2000, 500), background_color='white', expand_x=False, expand_y=True)],

                        ], size=(1200, 950), expand_y=True )],

                        ] 




    self.layout_about = [
                          [sg.Button('Repository', key='btn_mainarea_visitgithub')],
                          [sg.MLine(about_text, size=(64, 20), font=("Courier New", 9),expand_x = True, expand_y=True, disabled = True)], 
                        ] 


    self.layout_test_main = [
                          [sg.Button('test', key='btn_testmain')],
                          [sg.MLine(about_text, size=(64, 20), font=("Courier New", 9),expand_x = True, expand_y=True, disabled = True)], 
                        ] 


    self.layout_rxtx = [
                          [
                           sg.Button('Start Decoder', size=(11, 1), key='btn_8pskdecoder'),
                           sg.Button('Stop Decoder', size=(11, 1), key='btn_stop8pskdecoder'),
                           sg.Button('init output stream', size=(18, 1), key='btn_init_ostream'),
                           sg.Button('draw plot', size=(11, 1), key='btn_canvasdrawplotwaveform'),
                           sg.Button('START', size=(5, 1), key='btn_init_test'),
                           sg.Button('TEST', size=(5, 1), key='btn_testit'),
                           sg.CBox('Save Sampled Signal', key='cb_savesampledsignal', default=False ),
                           sg.InputText('sampled.wav', key='in_sampledsignalname', size=(15, 1)),
                           sg.Button('Load and Process', size=(15, 1), key='btn_loadandprocesssampledsignal')],

                        [  sg.Text('Indices Lower: -----', size=(170, 1), key='text_indices_lower')],
                        [  sg.Text('Indices Higher: -----', size=(170, 1), key='text_indices_higher')],


                        [ sg.Frame('Density plot', [

                          [ sg.Text('0   100   200   300   400   500   600   700   800  900 1000 1100 1200 1300 1400 1500  1600 1700 1800 1900 2000 2100 2200 2300  2400 2500 2600 2700 2800 2900 3000' )] ,

                          [sg.Graph(key='graph_density', canvas_size = (970, 200), graph_bottom_left=(0,0), graph_top_right = (3000, 100), background_color='white', expand_x=False, expand_y=False)],

                        ], size=(1200, 400))],


                        ] 


    self.layout_I3_test = [

                           [sg.Text('Intra Combine and Extract: '),
                            sg.Combo(combo_intra_combine_options, key='combo_intra_combine_type', default_value=combo_intra_combine_options[8], enable_events=True),
                            sg.Combo(combo_intra_extract_options, key='combo_intra_extract_type', default_value=combo_intra_extract_options[4], enable_events=True),
                            sg.InputText('1', key='in_intra_extract_factor', size=(10, 1), enable_events=True)],

                           #[sg.InputText('0.9', key='in_intra_extract_filterratio', size=(10, 1), enable_events=True),
                           # sg.InputText('0.85', key='in_intra_extract_filterinc', size=(10, 1), enable_events=True),
                           [sg.InputText('0.99', key='in_intra_extract_filterratio', size=(10, 1), enable_events=True),
                            sg.InputText('0.99', key='in_intra_extract_filterinc', size=(10, 1), enable_events=True),
                            sg.InputText('2e-3', key='in_intra_extract_searchaccuracy', size=(10, 1), enable_events=True),
                            sg.CBox('Simulate random phase shift offset', key='cb_simulatephaseshift', default=False ),
                            sg.CBox('Additive random phase shift offset', key='cb_simulatephaseshift_add', default=False ),
                            sg.CBox('Subtractive random phase shift offset', key='cb_simulatephaseshift_remove', default=False ),
                            sg.CBox('mid-signal thirds phase delay', key='cb_simulatemidsignalphasedelatthirds', default=False )],
                           [sg.CBox('Override standing wave offsets', key='cb_override_standingwaveoffsets', default=False ),
                            sg.InputText('0.5', key='in_standingwavelocation', size=(10, 1), enable_events=True),
                            sg.Button('Random', size=(10, 1), key='btn_randomstandingwaveoffset'),
                            sg.CBox('Random Pattern', key='cb_randomizepattern', default=False ),
                            sg.Combo(combo_standingwave_pattern_options, key='combo_standingwave_pattern', default_value=combo_standingwave_pattern_options[0], enable_events=True),
                            sg.Text('Mid Signal Phase Shift Amount: '),
                            sg.InputText('3', key='in_midsignalphaseshiftamount', size=(10, 1), enable_events=True)],
                           [sg.CBox('Override Pattern', key='cb_override_standingwavepattern', default=False ),
                            sg.Combo(combo_standingwave_patterns, key='combo_selectstandingwavepattern', default_value=combo_standingwave_patterns[0], enable_events=True),
                            sg.CBox('Override Downconvert Shift', key='cb_overridedownconvertshift', default=False ),
                            sg.InputText('0.5', key='in_downconvertshift', size=(10, 1), enable_events=True),
                            sg.CBox('Override Pulse Train Sigma', key='cb_overridepulsetrainsigma', default=False ),
                            sg.InputText('5.0', key='in_pulsetrainsigma', size=(10, 1), enable_events=True),
                            sg.CBox('Override Block Offsets Sigma', key='cb_overrideblockoffsetssigma', default=False ),
                            sg.InputText('1.25', key='in_blockoffsetssigma', size=(10, 1), enable_events=True)],
                           [sg.CBox('Override Pulse Start Sigma', key='cb_overridepulsestartsigma', default=False ),
                            sg.InputText('7', key='in_pulsestartsigma', size=(10, 1), enable_events=True),
                            sg.CBox('Override Spline Smoothing', key='cb_overridesplinesmoothing', default=False ),
                            sg.InputText('130', key='in_splinesmoothing', size=(10, 1), enable_events=True),
                            sg.CBox('Override Doppler Fourths Sigma', key='cb_overridedopplerfourthssigma', default=False ),
                            sg.InputText('3.0', key='in_dopplerfourthssigma', size=(10, 1), enable_events=True)],
                           [sg.CBox('Override Doppler Curve Fit', key='cb_overridedopplercurvefit', default=False ),
                            sg.Combo(combo_dopplercurvefittypes, key='combo_dopplercurvefittype', default_value=combo_dopplercurvefittypes[0], enable_events=True),
                            sg.InputText('10', key='in_dopplersmoothing', size=(10, 1), enable_events=True),
                            sg.CBox('Override Doppler Compression Calc', key='cb_overridedopplercompressioncalc', default=False ),
                            sg.CBox('Absolute', key='cb_overridedopplercompressionabsolute', default=False ),
                            sg.InputText('3', key='in_dopplercompressionfactor', size=(10, 1), enable_events=True)],
                           [sg.CBox('Override FEC Generator Polynomials', key='cb_overridegeneratorpolynomials', default=False ),
                            sg.InputText('13', key='in_fecgeneratorpolynomialdepth', size=(10, 1), enable_events=True),
                            sg.InputText('0o171', key='in_fecgeneratorpolynomial1', size=(10, 1), enable_events=True),
                            sg.InputText('0o123', key='in_fecgeneratorpolynomial2', size=(10, 1), enable_events=True)],

                          ] 


    self.layout_test = [
                          [
                           #sg.Combo(combo_modem_modes, key='combo_main_modem_modes', default_value=combo_modem_modes[8], enable_events=True),
                           #sg.Button('CW encoder', size=(11, 1), key='btn_cw_start', visible = False),
                           #sg.Button('init output stream', size=(11, 1), key='btn_init_ostream', visible = False),
                           #sg.Button('START', size=(5, 1), key='btn_init_test', visible = False),
                           #sg.Button('STOP', size=(5, 1), key='btn_stop8pskdecoder', visible = False),
                           #sg.Button('draw plot', size=(11, 1), key='btn_canvasdrawplotwaveform', visible = False),
                           #sg.Button('start 8psk decoder', size=(11, 1), key='btn_8pskdecoder', visible = False),
                           sg.CBox('Enable AWGN', key='cb_enable_awgn', default=True ),
                           sg.CBox('Enable Timing Noise', key='cb_enable_timing_noise', default=False ),
                           sg.CBox('Enable Phase Noise', key='cb_enable_phase_noise', default=False ),
                           sg.Text('Code Options: ')  ,
                           sg.Combo(combo_code_options, key='combo_code_options', default_value=combo_code_options[0], enable_events=True)],
                           #sg.Text('Block Size: ')  ,
                           #sg.Combo(combo_separation_options, key='option_separation_options', default_value=combo_separation_options[2] ),
                          #[sg.Text('Chunk Size: ')  ,
                          # sg.Combo(combo_chunk_options, key='combo_chunk_options', default_value=combo_chunk_options[5], enable_events=True)],
                          [sg.CBox('Override RRC Alpha: ', key='cb_override_rrc_alpha', default=False ),
                           sg.InputText('0.7', key='in_rrc_alpha', size=(20, 1), enable_events=True),
                           sg.CBox('Override RRC T: ', key='cb_override_rrc_t', default=False ),
                           sg.InputText('0.9', key='in_rrc_t', size=(20, 1), enable_events=True),
                           sg.CBox('Override carrier separation', key='cb_enable_separation_override', default=True ),
                           sg.CBox('Override Extraction Threshold', key='cb_override_extractionthreshold', default=False ),
                           sg.InputText('600', key='in_extractionthreshold', size=(20, 1), enable_events=True)],
                          [sg.CBox('Override Costas Loop', key='cb_override_costasloop', default=False, enable_events=True ),
                           sg.InputText('', key='in_costasloop_dampingfactor', size=(8, 1), enable_events=True),
                           sg.InputText('', key='in_costasloop_loopbandwidth', size=(8, 1), enable_events=True),
                           sg.InputText('', key='in_costasloop_K1', size=(8, 1), enable_events=True),
                           sg.InputText('', key='in_costasloop_K2', size=(8, 1), enable_events=True),
                           sg.CBox('Override Downconvert Method', key='cb_override_downconvertmethod', default=False ),
                           sg.Combo(combo_downconvert_options, key='combo_downconvert_type', default_value=combo_downconvert_options[1], enable_events=True),
                           sg.Text('Text Options: ')  ,
                           sg.Combo(combo_text_options, key='combo_text_options', default_value=combo_text_options[3], enable_events=True)],


                        [sg.Frame('Generate Test Data', [

                          [
                           sg.Combo(combo_random_choice_options, key='option_random_choices', default_value=combo_random_choice_options[0] ),
                           sg.Text('Random Values + \\ - : '),
                           sg.InputText('0.2', key='in_random_plus_minus', size=(8, 1), enable_events=True),
                           sg.Text('Increments: '),
                           sg.InputText('1000', key='in_random_increments', size=(8, 1), enable_events=True),
                           sg.Text('Num Cycles: '),
                           sg.InputText('5', key='in_random_numcycles', size=(8, 1), enable_events=True),
                           sg.Button('Generate', size=(10, 1), key='btn_generate_test_data')],
                          [sg.InputText('osmod_v0-1-0_results_data.csv', key='in_resultsdatafilename', size=(50, 1), enable_events=True),
                           sg.FileBrowse('Select File', file_types=(('CSV Files', '*.csv'), ('All Files', '*.*')))],
 
                        ], size=(1200, 100) )],


                        [sg.Frame('AWGN Factor', [

                          [
                           sg.Slider(range=(0,100), default_value = 8, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='btn_slider_awgn')],
 
                        ], size=(1200, 60) )],

                        [sg.Frame('Amplitude', [

                          [
                           sg.Slider(range=(0.01,1), default_value = 0.7, orientation='h', resolution=0.01, expand_x = True, enable_events = True, key='slider_amplitude')],
 
                        ], size=(1200, 60) )],

                        [sg.Frame('Carrier Separation', [
                          [
                           sg.Slider(range=(1, 500), default_value = 15, orientation='h', resolution=1, expand_x = True, enable_events = True, key='slider_carrier_separation')],
                        ], size=(1200, 60) )],





                        #[sg.Frame('Chart 1', [
                        #  [sg.Canvas(key='chart_canvas_oneoffour', size=(100, 50), expand_x=False, expand_y=False)],
                        #], size=(500, 250) ),
                        # sg.Frame('Chart 2', [
                        #  [sg.Canvas(key='chart_canvas_twooffour', size=(11, 100), expand_x=False, expand_y=False)],
                        #], size=(500, 250) )],




                        ] 


    self.tabgrp = [

                       [
                        #sg.Button('Reset All Modes', size=(8, 1), key='btn_reset_all'),

                       
                        sg.Text('FFT Frequency:')  ,
                        sg.Text('-----', key='text_info_fftfreq'),
                        sg.Text('Low Frequency:')  ,
                        sg.Text('-----', key='text_info_freq1'),
                        sg.Text('High Frequency:')  ,
                        sg.Text('-----', key='text_info_freq2'),
                        sg.Text('Mode Info:')  ,
                        sg.Text('-----', key='text_info_description'),

                        sg.Text('', expand_x = True),
                        sg.Button('Save', size=(8, 1), key='btn_save'),
                        sg.Button('Exit', size=(8, 1))],

                          [sg.Text('Eb/N0 (dB): -----', size=(30, 1), key='text_ebn0db_value' ) ,
                           sg.Text('BER: -----', size=(30, 1), key='text_ber_value')  ,
                           sg.Text('Eb/N0: -----', size=(30, 1), key='text_ebn0_value' ) ,
                           sg.Text('SNR (dB): -----', size=(30, 1), key='text_snr_value' )] ,


                       [
                           sg.Combo(combo_modem_modes, key='combo_main_modem_modes', default_value=combo_modem_modes[8], enable_events=True),
                           sg.CBox('Display Phases', key='cb_display_phases', default=True ),
                           sg.Text('Chart Type: '),
                           sg.Combo(combo_chart_options, key='option_chart_options', default_value=combo_chart_options[1] ),
                           sg.Text('Chunk Size: ')  ,
                           sg.Combo(combo_chunk_options, key='combo_chunk_options', default_value=combo_chunk_options[5], enable_events=True),
                           sg.CBox('Align 1st Carrier', key='cb_enable_align', default=True ),
                           sg.Combo(combo_align_options, key='option_carrier_alignment', default_value=combo_align_options[10] )],
                          [sg.CBox('Override Sample Rate', key='cb_enable_sample_rate_override', default=False),
                           sg.InputText('', key='in_sample_rate_override', size=(8, 1), enable_events=True),
                           sg.CBox('Override Block Size: ', key='cb_override_blocksize', default=False ),
                           sg.InputText('51200', key='in_symbol_block_size', size=(20, 1), enable_events=True),
                           sg.CBox('48kHz sampling', key='cb_override_standard48k', default=False, enable_events=True ),
                           sg.CBox('16kHz sampling', key='cb_override_sampling16k', default=False, enable_events=True ),
                           sg.CBox('RX Frequency Delta', key='cb_enable_rxfrequencydelta', default=False, enable_events=True ),
                           sg.InputText('-1.05', key='in_rxfrequencydelta', size=(20, 1), enable_events=True)],


                        [sg.Frame('Frequency', [

                          [
                           sg.Slider(range=(0,3000), default_value = 1400, orientation='h', resolution=1, expand_x = True, enable_events = True, key='slider_frequency')],
 
                        ], size=(1200, 60) )],


                        [
                            sg.MLine('', size=(64, 8), font=("Courier New", 9), key='ml_txrx_sendtext', text_color='black', background_color='white', expand_x = True, expand_y=False, disabled = False)], 
                        [
                            sg.MLine('', size=(64, 8), font=("Courier New", 9), key='ml_txrx_recvtext', text_color='black', background_color='white', expand_x = True, expand_y=False, disabled = False)], 

                        [   sg.Button('Run Test', size=(11, 1), key='btn_testit2'),
                            sg.Button('Test Routine', size=(11, 1), key='btn_testit'),
                            sg.Combo(combo_test_routines, key='combo_test_routine_options', default_value=combo_test_routines[0], enable_events=True),

                            sg.Text('Sequential Test Count: '),
                            sg.Text('0', key='text_sequential_test_counter'),
                            sg.Button('Reset Test Counter', size=(11, 1), key='btn_resettestcounter'),

                            sg.Text('', expand_x = True),
                            sg.Button('Reset Mode', size=(8, 1), key='btn_reset')],


                       [sg.TabGroup([[
                             sg.Tab('General Test Options', self.layout_test, title_color='Blue',border_width =10, background_color='Gray' ),
                             sg.Tab('I3 Test Options', self.layout_I3_test, title_color='Blue',border_width =10, background_color='Gray' ),
                             sg.Tab('Audio TxRx', self.layout_rxtx, title_color='Blue',border_width =10, background_color='Gray' )]],
                       tab_location='centertop',
                       title_color='Blue', tab_background_color='Dark Gray', background_color='Dark Gray', size=(1200, 400), selected_title_color='Black', selected_background_color='White', key='tabgrp_main' )]]  



    self.layout_main_charts_tab = [

                              [sg.TabGroup([[
                               #sg.Tab('Testing', self.layout_test_main, title_color='Blue',border_width =10, key='tab_main' ),
                               sg.Tab('Phase Charts', self.layout_phase_charts, title_color='Blue',border_width =10, background_color='Gray' ),
                               sg.Tab('Wave Charts', self.layout_wave_charts, title_color='Blue',border_width =10, background_color='Gray' ),
                               sg.Tab('Pulse Train Charts', self.layout_pulsetrain_charts, title_color='Blue',border_width =10, background_color='Gray' ),
                               sg.Tab('Analysis Charts', self.layout_analysis_charts, title_color='Blue',border_width =10, background_color='Gray' ),
                               sg.Tab('Simulations', self.layout_simulator_charts, title_color='Blue',border_width =10, background_color='Gray' ),
                               sg.Tab('testing', self.layout_test_main, title_color='Blue', key='tab_main_phase_charts')]],

                               tab_location='centertop',
                               selected_title_color='Black', selected_background_color='White', key='tabgrp_main_tabs_charts' )],
                            ]


    self.layout_main_tabs = [

                              [sg.TabGroup([[
                               sg.Tab('Main', self.tabgrp, title_color='Blue',border_width =10, key='tab_main' ),
                               sg.Tab('Graphs and Analytics', self.layout_main_charts_tab, title_color='Blue', key='tab_graphsandanalytics'),
                               sg.Tab('About', self.layout_about, title_color='Blue',border_width =10, background_color='Gray' )]],

                               tab_location='centertop',
                               selected_title_color='Black', selected_background_color='White', key='tabgrp_main_tabs' )],
                            ]


    self.window = sg.Window("OSMOD de WH6GGO v0.1.0 Alpha - Test and Reference Code for LB28 Modulation", self.layout_main_tabs, default_element_size=(40, 1), grab_anywhere=False, disable_close=True)                       


    #self.window = sg.Window("OSMOD de WH6GGO v0.0.6 Alpha - Test and Reference Code for LB28 Modulation", self.tabgrp, default_element_size=(40, 1), grab_anywhere=False, disable_close=True)                       

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


class ReceiveControlsProc(object):

  window_initialized = False
  chart_timer = 0
  
  def __init__(self):  
    self.debug = db.Debug(cn.DEBUG_INFO)
    return

  def event_catchall(self, window, values, form_gui):

    if(self.window_initialized == False and form_gui.window != None):
      self.window_initialized = True		
      """ set some default values..."""
      form_gui.window['combo_main_modem_modes'].update('LB28-4-2-40-N')
      form_gui.osmod.setInitializationBlock('LB28-4-2-40-N')
      form_gui.osmod.setScreenOptions('LB28-4-2-40-N', form_gui, form_gui.osmod.opd.main_settings)  


    if self.window_initialized == True:

      if False:
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

            #strong_freqs = form_gui.osmod.mod_psk.getStrongestFrequency(data)
            #strong_freqs = form_gui.osmod.modulation_object.getStrongestFrequency(data)
            strong_freqs = form_gui.osmod.detector.getStrongestFrequencyOverRange(data)
            window['graph_density'].draw_line(point_from=(int(strong_freqs),0), point_to=(int(strong_freqs),1), width=5, color='blue')

        self.chart_timer = self.chart_timer + 1


      if form_gui.drawitQueue.empty() == False:
        tuple_data = form_gui.drawitQueue.get_nowait()
        fig = tuple_data[0]
        canvas_name = tuple_data[1]
        form_gui.plotWaveCanvasDraw(fig, window[canvas_name].tk_canvas)

    return()



  def event_canvasdrawplotwaveform(self, window, values, form_gui):
    sys.stdout.write("event_canvasdrawplotwaveform\n")

    if(self.window_initialized == True and form_gui.window != None):
      data = form_gui.osmod.mod_psk.modulateChunk8PSK(form_gui.osmod.carrier_frequency_reference, [0,1,1, 0, 1, 0, 1, 1, 1, 0, 1, 1]) 
      t1 = threading.Thread(target=form_gui.osmod.mod_psk.plotWaveCanvas, args=(2400, data, window['chart_canvas_oneoffour'].tk_canvas, ))
      t1.start()

  def event_8pskdecoder(self, window, values, form_gui):
    sys.stdout.write("event_8pskdecoder\n")

    form_gui.osmod.startTimer('init')

    mode = values['combo_main_modem_modes']

    form_gui.osmod.startDecoder(mode, window, values)

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
    #form_gui.osmod.demodulation_object.re_drawWaveCharts(chart1_xmag, chart2_xmag, chart1_ymag, chart2_ymag)
    form_gui.osmod.analysis.re_drawWaveCharts(chart1_xmag, chart2_xmag, chart1_ymag, chart2_ymag)

  def event_testit(self, window, values, form_gui):
    sys.stdout.write("event_testit\n")
    test = OsmodTest(form_gui.osmod, window)
    mode = values['combo_main_modem_modes']

    chunk_num = values['combo_chunk_options'].split(':')[0]
    amplitude = values['slider_amplitude']
    carrier_separation_override = values['slider_carrier_separation']

    routine_type = values['combo_test_routine_options']
    test.testRoutines(values, mode, routine_type, chunk_num, carrier_separation_override, amplitude)

    form_gui.osmod.test_counter = form_gui.osmod.test_counter + 1
    form_gui.window['text_sequential_test_counter'].update(str(form_gui.osmod.test_counter))

  def event_resettestcounter(self, window, values, form_gui):
    sys.stdout.write("event_resettestcounter\n")

    form_gui.osmod.test_counter = 0
    form_gui.window['text_sequential_test_counter'].update(str(form_gui.osmod.test_counter))


  def event_testit2(self, window, values, form_gui):
    sys.stdout.write("event_testit\n")

    form_gui.osmod.test_counter = form_gui.osmod.test_counter + 1
    form_gui.window['text_sequential_test_counter'].update(str(form_gui.osmod.test_counter))

    test = OsmodTest(form_gui.osmod, window)
    mode = values['combo_main_modem_modes']
    noise = values['btn_slider_awgn']
    text_num = values['combo_text_options'].split(':')[0]
    chunk_num = values['combo_chunk_options'].split(':')[0]
    amplitude = values['slider_amplitude']

    carrier_separation_override = values['slider_carrier_separation']

    form_gui.osmod.writeModeToCache(mode, form_gui, values)

    test.testRoutine2(mode, form_gui, values, noise, text_num, chunk_num, carrier_separation_override, amplitude)

  """ reset all values for the selected mode back to the default values"""
  def event_reset(self, window, values, form_gui):
    sys.stdout.write("event_reset\n")
    mode = values['combo_main_modem_modes']
    form_gui.osmod.resetMode(form_gui, mode)
    #form_gui.osmod.updateCachedSettings(values, form_gui)


  def event_reset_all(self, window, values, form_gui):
    sys.stdout.write("event_reset_all\n")

  def event_overridesampling16k(self, window, values, form_gui):
    sys.stdout.write("event_overridesampling16k\n")

    try:
      mode = values['combo_main_modem_modes']
      #sample_rate = int(form_gui.osmod.modulation_initialization_block[mode]['sample_rate'])
      sample_rate = form_gui.osmod.getParam(mode, 'sample_rate')
      #block_size  = int(form_gui.osmod.modulation_initialization_block[mode]['symbol_block_size'])
      block_size  = form_gui.osmod.getParam(mode, 'symbol_block_size')
      if values['cb_override_sampling16k'] == True:
        form_gui.window['cb_enable_sample_rate_override'].update(True)
        form_gui.window['cb_override_blocksize'].update(True)
        form_gui.window['in_sample_rate_override'].update(sample_rate * 2)
        form_gui.window['in_symbol_block_size'].update(block_size * 2)
        form_gui.osmod.sample_rate = sample_rate
        form_gui.osmod.symbol_block_size  = block_size
      else:
        form_gui.window['cb_enable_sample_rate_override'].update(False)
        form_gui.window['cb_override_blocksize'].update(False)
        form_gui.window['in_sample_rate_override'].update(sample_rate)
        form_gui.window['in_symbol_block_size'].update(block_size)

    except:
      sys.stdout.write("Exception in event_overridesampling16k: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_overridestandard48k(self, window, values, form_gui):
    sys.stdout.write("event_overridestandard48k\n")

    try:
      mode = values['combo_main_modem_modes']
      #sample_rate = int(form_gui.osmod.modulation_initialization_block[mode]['sample_rate'])
      sample_rate = form_gui.osmod.getParam(mode, 'sample_rate')
      #block_size  = int(form_gui.osmod.modulation_initialization_block[mode]['symbol_block_size'])
      block_size  = form_gui.osmod.getParam(mode, 'symbol_block_size')

      if values['cb_override_standard48k'] == True:
        form_gui.window['cb_enable_sample_rate_override'].update(True)
        form_gui.window['cb_override_blocksize'].update(True)
        form_gui.window['in_sample_rate_override'].update(sample_rate * 6)
        form_gui.window['in_symbol_block_size'].update(block_size * 6)
        form_gui.osmod.sample_rate = sample_rate
        form_gui.osmod.symbol_block_size  = block_size
        #form_gui.osmod.sample_rate = sample_rate * 6
        #form_gui.osmod.symbol_block_size  = block_size * 6
      else:
        form_gui.window['cb_enable_sample_rate_override'].update(False)
        form_gui.window['cb_override_blocksize'].update(False)
        form_gui.window['in_sample_rate_override'].update(sample_rate)
        form_gui.window['in_symbol_block_size'].update(block_size)

    except:
      sys.stdout.write("Exception in event_overridestandard48k: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_save(self, window, values, form_gui):
    sys.stdout.write("event_save\n")
    form_gui.osmod.opd.writeMainDictionaryToFile("osmod_main_settings.txt", values)

    form_gui.osmod.updateCachedSettings(values, form_gui)


  def event_mainmodemmodes(self, window, values, form_gui):
    sys.stdout.write("event_mainmodemmodes\n")

    mode = values['combo_main_modem_modes']

    form_gui.osmod.setInitializationBlock(mode)

    form_gui.osmod.setScreenOptions(mode, form_gui, form_gui.osmod.opd.main_settings)

    form_gui.window['text_info_description'].update(str(form_gui.osmod.modulation_initialization_block[mode]['info']))

  def event_randomstandingwaveoffset(self, window, values, form_gui):
    sys.stdout.write("event_randomstandingwaveoffset\n")
    rand_num = random.randint(0,1000)
    form_gui.window['in_standingwavelocation'].update(rand_num/1000)
    rand_num = random.randint(0,14)
    form_gui.window['combo_standingwave_pattern'].update(form_gui.combo_standingwave_pattern_options[rand_num])
    sys.stdout.write("combo_standingwave_pattern_options: " + str(form_gui.combo_standingwave_pattern_options[rand_num]) + "\n")


  def event_generatetestdata(self, window, values, form_gui):
    sys.stdout.write("event_generatetestdata\n")
    test_type = values['option_random_choices']
    plus_minus_amount = form_gui.window['in_random_plus_minus'].get()
    increments = form_gui.window['in_random_increments'].get()
    num_cycles = form_gui.window['in_random_numcycles'].get()
    form_gui.osmod.analysis.generateTestData(form_gui, window, values, test_type, plus_minus_amount, increments, num_cycles)

  def event_filter1matchtype(self, window, values, form_gui):
    sys.stdout.write("event_filter1matchtype\n")

    match_type = values['combo_filter1_matchtype']
    if match_type == 'Mode Name ==':
      sys.stdout.write("match Mode Name\n")
      selections = form_gui.combo_modem_modes
      #form_gui.window['combo_analysis_modes'].update(values=selections)
      #form_gui.window['combo_analysis_modes'].update(selections[0])

    elif match_type == 'Pattern Type ==':
      sys.stdout.write("match Pattern Type\n")
      selections = form_gui.combo_standingwave_pattern_options
      #form_gui.window['combo_analysis_modes'].update(values=selections)
      #form_gui.window['combo_analysis_modes'].update(selections[0])

    elif match_type == 'Preset Pattern ==':
      sys.stdout.write("match Preset Pattern\n")
      selections = form_gui.combo_standingwave_patterns
      #form_gui.window['combo_analysis_modes'].update(values=selections)
      #form_gui.window['combo_analysis_modes'].update(selections[0])

    elif match_type == 'AWGN ==':
      sys.stdout.write("match AWGN Value\n")
      data = form_gui.osmod.analysis.readDataFromFile()
      selections = form_gui.osmod.analysis.obtainDatasetValuesSingle(data, ocn.DATA_NOISE_FACTOR)
      #form_gui.window['combo_analysis_modes'].update(values=selections)
      #form_gui.window['combo_analysis_modes'].update(selections[0])

    elif match_type == 'Pulse Shape ==':
      sys.stdout.write("match Pulse Shape\n")
      data = form_gui.osmod.analysis.readDataFromFile()
      selections = form_gui.osmod.analysis.obtainDatasetValuesDouble(data, ocn.DATA_RRC_ALPHA, ocn.DATA_RRC_T)

    elif match_type == 'Pulse Train Length ==':
      #sys.stdout.write("match Pulse Shape\n")
      data = form_gui.osmod.analysis.readDataFromFile()
      selections = form_gui.osmod.analysis.obtainDatasetValuesSingle(data, ocn.DATA_PULSE_TRAIN_LENGTH)

    elif match_type == 'Disposition ==':
      data = form_gui.osmod.analysis.readDataFromFile()
      selections = form_gui.osmod.analysis.obtainDatasetValuesSingle(data, ocn.DATA_DISPOSITION)




    form_gui.window['combo_analysis_modes'].update(values=selections)
    form_gui.window['combo_analysis_modes'].update(selections[0])
  


  def event_overridecostasloop(self, window, values, form_gui):
    sys.stdout.write("event_overridecostasloop\n")
    mode = values['combo_main_modem_modes']

    if values['cb_override_costasloop'] == False and form_gui.window['in_costasloop_dampingfactor'].get().strip()=='':
      form_gui.window['in_costasloop_dampingfactor'].update(str(form_gui.osmod.modulation_initialization_block[mode]['parameters'][6]))
      form_gui.window['in_costasloop_loopbandwidth'].update(str(form_gui.osmod.modulation_initialization_block[mode]['parameters'][7]))
      form_gui.window['in_costasloop_K1'].update(str(form_gui.osmod.modulation_initialization_block[mode]['parameters'][8]))
      form_gui.window['in_costasloop_K2'].update(str(form_gui.osmod.modulation_initialization_block[mode]['parameters'][9]))


  def event_plotsplinechart(self, window, values, form_gui):
    sys.stdout.write("event_plotsplinechart\n")
    try:
      scaling_params = form_gui.osmod.analysis.calcScaling(form_gui.osmod.analysis.previous_plotted_data, [860, 420], False)
      #form_gui.osmod.analysis.drawSplineChart(form_gui.osmod.analysis.previous_plotted_data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, scaling_params)
      form_gui.osmod.analysis.drawSplineChart(form_gui.osmod.analysis.previous_plotted_data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, scaling_params)
      #form_gui.osmod.analysis.drawSplineChart(form_gui.osmod.analysis.previous_plotted_data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, scaling_params)
    except:
      sys.stdout.write("Exception in event_plotsplinechart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def event_plotpulsetraintype(self, window, values, form_gui):
    sys.stdout.write("event_plotpulsetraintype\n")

    try:
      chart_type = values['combo_plotpulsetraintype']
      if chart_type == 'Refresh':
        chart_names = form_gui.osmod.detector.getSavedDataNames()
        chart_names = ['Refresh'] + chart_names
        sys.stdout.write("chart_names: " + str(chart_names) + "\n")
        form_gui.window['combo_plotpulsetraintype'].update(values=chart_names)
        form_gui.window['combo_plotpulsetraintype'].update(chart_names[0])

    except:
      sys.stdout.write("Exception in event_plotpulsetraintype: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_plotpulsetrainchart(self, window, values, form_gui):
    sys.stdout.write("event_plotpulsetrainchart\n")
    try:
      #chart_names = form_gui.osmod.detector.getSavedDataNames()
      #sys.stdout.write("chart_names: " + str(chart_names) + "\n")

      chart_type = values['combo_plotpulsetraintype']
      if chart_type != 'Refresh':
        sys.stdout.write("plotting: " + str(chart_type) + "\n")
        data = form_gui.osmod.detector.dict_saved_data[chart_type]
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'chart_type', 'black', True, False, [])
        #scaling_params = form_gui.osmod.analysis.calcScaling(data, [860, 420], True)
        #form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, False, scaling_params)


      """
      chart_type = values['combo_plotpulsetraintype']
      if chart_type == 'Smoothed Signal':
        sys.stdout.write("plotting smoothed signal\n")
        data = form_gui.osmod.detector.dict_saved_data['smoothed signal']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, False, [])
      elif chart_type == 'Block Offsets':
        sys.stdout.write("plotting block offsets\n")
        data = form_gui.osmod.detector.dict_saved_data['block offsets']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      elif chart_type == 'Block Offsets Half':
        sys.stdout.write("plotting block offsets\n")
        data = form_gui.osmod.detector.dict_saved_data['block offsets half']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      elif chart_type == 'Block Offsets Fourth':
        sys.stdout.write("plotting block offsets\n")
        data = form_gui.osmod.detector.dict_saved_data['block offsets fourth']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      elif chart_type == 'Block Offsets Fourth b':
        sys.stdout.write("plotting block offsets\n")
        data = form_gui.osmod.detector.dict_saved_data['block offsets fourth b']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      elif chart_type == 'Interpolated Pulse Offsets':
        sys.stdout.write("plotting block offsets\n")
        data = form_gui.osmod.detector.dict_saved_data['interpolated pulse offsets']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      elif chart_type == 'Dominant Series 1':
        sys.stdout.write("plotting Dominant Series 1\n")
        data = form_gui.osmod.detector.dict_saved_data['dominant series 1']
        form_gui.osmod.analysis.previous_plotted_data = data
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', False, False, [])
      else:
        sys.stdout.write("plot default\n")
        data = [0,1,2,3,4,5,6,7,8,9,10]
        form_gui.osmod.analysis.previous_plotted_data = data
        scaling_params = form_gui.osmod.analysis.calcScaling(data, [860, 420], True)
        #form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, True, scaling_params)
        form_gui.osmod.analysis.drawPulseTrainCharts(data, 'pulse train', window, form_gui, 'graph_pulsetraindata1', 'my chart', 'black', True, False, scaling_params)
      """

    except:
      sys.stdout.write("Exception in event_plotpulsetrainchart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_erasechart(self, window, values, form_gui):
    sys.stdout.write("event_erasechart\n")
    graph = window['graph_pulsetraindata1']
    graph.erase()


  def event_sliderfrequency(self, window, values, form_gui):
    sys.stdout.write("event_sliderfrequency\n")
    center_frequency = values['slider_frequency']
    separation_override = values['slider_carrier_separation']
    form_gui.osmod.calcCarrierFrequencies(center_frequency, separation_override)


  def event_plotresults(self, window, values, form_gui):
    sys.stdout.write("event_plotresults\n")
    data = form_gui.osmod.analysis.readDataFromFile()

    chart_type = values['option_analysis_chart_types']

    #form_gui.osmod.modulation_object.drawDotPlotCharts(data, chart_type, window, values, form_gui)
    form_gui.osmod.analysis.drawDotPlotCharts(data, chart_type, window, values, form_gui)

  def event_savedotplotsubset(self, window, values, form_gui):
    sys.stdout.write("event_savedotplotsubset\n")
    try:
      form_gui.osmod.analysis.writeDataToFile2(form_gui.osmod.analysis.dot_plot_subset, "subset.csv", 'a')
    except:
      sys.stdout.write("Exception in event_savedotplotsubset: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_analysisaddmatchtable(self, window, values, form_gui):
    sys.stdout.write("event_analysisaddmatchtable\n")
    try:
      mode_to_match = window['combo_analysis_modes'].get()
      form_gui.osmod.analysis.match_table.append('-')
      index = len(form_gui.osmod.analysis.match_table) - 1
      form_gui.osmod.analysis.match_table[index] = mode_to_match
      form_gui.window['text_analysisfilter1values'].update(str(form_gui.osmod.analysis.match_table))

    except:
      sys.stdout.write("Exception in event_analysisaddmatchtable: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_analysisclearmatchtable(self, window, values, form_gui):
    sys.stdout.write("event_analysisclearmatchtable\n")
    try:
      form_gui.osmod.analysis.match_table = []
      form_gui.window['text_analysisfilter1values'].update(str(form_gui.osmod.analysis.match_table))
    except:
      sys.stdout.write("Exception in event_analysisclearmatchtable: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_plotsimulation(self, window, values, form_gui):
    sys.stdout.write("event_plotsimulation\n")

    chart_type = values['option_simulator_chart_types']
    form_gui.osmod.simulator.drawSimulation(chart_type, window, values, form_gui)


  def event_loadandprocesssampledsignal(self, window, values, form_gui):
    sys.stdout.write("event_loadandprocesssampledsignal\n")
    sampled_signal_name = form_gui.window['in_sampledsignalname'].get()
    audio_array = form_gui.osmod.modulation_object.readFileWav(sampled_signal_name)
    fft_frequency = 1382
    separation_override = values['slider_carrier_separation']
    frequency = form_gui.osmod.calcCarrierFrequenciesFromFFT(fft_frequency, separation_override)
    form_gui.osmod.decoder_callback(audio_array, frequency)


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

    mode = values['combo_main_modem_modes']

    """ start decoder / demodulator """
    form_gui.osmod.startDecoder(mode, window, values)


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


  def event_exit_receive(self, values):

    try:
      sys.stdout.write("IN event_exit_receive\n")

    except:
      sys.stdout.write("Exception in event_exit_receive: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")
      
    return()




  dispatch = {
      'btn_init_ostream' : event_initostream,
      'btn_init_test'    : event_inittest,
      'btn_canvasdrawplotwaveform' : event_canvasdrawplotwaveform,
      'btn_8pskdecoder'  : event_8pskdecoder,
      'btn_stop8pskdecoder'  : event_stop8pskdecoder,
      'btn_testit'       : event_testit,
      'btn_testit2'       : event_testit2,
      'btn_resettestcounter' :  event_resettestcounter,
      'slider_chart1_xmag' :  event_sliderchart1xmag,
      'slider_chart1_ymag' :  event_sliderchart1xmag,
      'slider_chart2_xmag' :  event_sliderchart1xmag,
      'slider_chart2_ymag' :  event_sliderchart1xmag,
      'combo_code_options' :  event_codeoptions,
      'combo_main_modem_modes' : event_mainmodemmodes,
      'btn_save'           : event_save,
      'btn_reset'          : event_reset,
      'btn_reset_all'          : event_reset_all,
      'btn_plot_results'   : event_plotresults,
      'btn_plot_simulation'   : event_plotsimulation,
      'btn_generate_test_data'  : event_generatetestdata,
      'slider_frequency'   : event_sliderfrequency,
      'cb_override_standard48k' : event_overridestandard48k,
      'cb_override_sampling16k' : event_overridesampling16k,
      'cb_override_costasloop'  : event_overridecostasloop,
      'btn_randomstandingwaveoffset' : event_randomstandingwaveoffset,
      'combo_filter1_matchtype'  : event_filter1matchtype,
      'btn_plot_pulsetrainchart'  : event_plotpulsetrainchart,
      'btn_erasechart'            :  event_erasechart,
      'btn_plotsplinechart'            :  event_plotsplinechart,
      'combo_plotpulsetraintype'  : event_plotpulsetraintype,
      'btn_loadandprocesssampledsignal'  : event_loadandprocesssampledsignal,
      'btn_savedotplotsubset'     : event_savedotplotsubset,
      'btn_analysisaddmatchtable' : event_analysisaddmatchtable,
      'btn_analysisclearmatchtable' : event_analysisclearmatchtable,
  }
  
  


def main():

  debug = db.Debug(cn.DEBUG_INFO)

  """ create the main gui controls event handler """
  form_gui = FormGui(None, None)
  window = form_gui.createMainTabbedWindow('', None)
  dispatcher = ReceiveControlsProc()

  form_gui.runReceive(window, dispatcher)

  #t2 = threading.Thread(target=form_gui.runReceive, args=(window, dispatcher,))
  #t2.start()

  while True:
    if form_gui.plotQueue.empty() == False:
      sys.stdout.write("Preparing plot on main thread\n")
      tuple_data = form_gui.plotQueue.get_nowait()
      data = tuple_data[0]
      canvas_name = tuple_data[1]
      chart_name  = tuple_data[2]
      chart_color = tuple_data[3]
      erase       = tuple_data[4]

      #fig = form_gui.plotWaveCanvasPrepare(len(data), data, None)
      #form_gui.drawitQueue.put((fig, canvas_name))

      #form_gui.osmod.demodulation_object.drawPhaseCharts(data, 'phase', window, form_gui, canvas_name, chart_name, chart_color, erase)
      form_gui.osmod.analysis.drawPhaseCharts(data, 'phase', window, form_gui, canvas_name, chart_name, chart_color, erase)

    time.sleep(2)

if __name__ == '__main__':
    main()




