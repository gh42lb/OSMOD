#!/usr/bin/env python


import constant as cn
import osmod_constant as ocn
import debug as db
import sys
import numpy as np
from commpy.channelcoding import Trellis, turbo_encode, turbo_decode
from commpy.channelcoding import RandInterlv, conv_encode, viterbi_decode 
from commpy.channels import awgn
#from commpy.utilities import dec2bin

from viterbi import Viterbi

from pyldpc import make_ldpc, decode, get_message, encode


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




class OsmodFEC(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  ldpc     = None
  viterbi  = None
  turbo    = None
  fec_type = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod
    self.window = window

  def init_params(self, fec_type):
    self.fec_type = fec_type

    if self.fec_type == ocn.FEC_TURBO:
      self.turbo         = OsmodTurbo(self.osmod, self.window)
      self.turbo.init_params()
    elif self.fec_type == ocn.FEC_VITERBI:
      self.viterbi       = OsmodViterbi(self.osmod, self.window)
      self.viterbi.init_params()
    elif self.fec_type == ocn.FEC_VITERBI2:
      self.viterbi       = OsmodViterbi2(self.osmod, self.window)
      self.viterbi.init_params()
    elif self.fec_type == ocn.FEC_LDPC:
      self.ldpc          = OsmodLDPC(self.osmod, self.window)
      self.ldpc.init_params()

  def encodeFEC(self, message):
    if self.fec_type == ocn.FEC_TURBO:
      return self.turbo.encode(message)
    elif self.fec_type == ocn.FEC_VITERBI or self.fec_type == ocn.FEC_VITERBI2:
      return self.viterbi.encode(message)
    elif self.fec_type == ocn.FEC_LDPC:
      return self.ldpc.encode(message)

  def decodeFEC(self, message):
    if self.fec_type == ocn.FEC_TURBO:
      return self.turbo.decode(message)
    elif self.fec_type == ocn.FEC_VITERBI or self.fec_type == ocn.FEC_VITERBI2:
      return self.viterbi.decode(message)
    elif self.fec_type == ocn.FEC_LDPC:
      return self.ldpc.decode(message)


  def combineCodes(self, three_codes):
    combined_code = []
    for code_1, code_2, code_3 in zip(three_codes[0], three_codes[1], three_codes[2]):
      combined_code_item = code_1 + code_2 + code_3
      if combined_code_item >= 2:
        combined_code.append(1)
      else:
        combined_code.append(0)

    return combined_code



class OsmodTurbo(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod


  def init_params(self):
    print("Turbo init_params")
    memory = np.array([2])
    g_matrix = np.array([[0o7, 0o5]])
    print("Turbo init_params1")
   
    feedback = 0o7

    self.trellis1 = Trellis(memory, g_matrix, feedback=feedback, code_type='rsc')
    print("Turbo init_params2")

    self.trellis2 = Trellis(memory, g_matrix, feedback=feedback, code_type='rsc')
    print("Turbo init_params3")


  def encode(self, message):
    #print("Turbo encode")
    self.debug.info_message("Turbo encode" )

    msg_length = len(message)

    self.interleaver = RandInterlv(msg_length, seed=123)

    sys_bits, non_sys_bits_1, non_sys_bits_2 = turbo_encode(message, self.trellis1, self.trellis2, self.interleaver)
    codeword = np.concatenate((sys_bits, non_sys_bits_1, non_sys_bits_2))

    self.sys_bits_len = len(sys_bits)
    self.non_sys_bits_1_len = len(non_sys_bits_1)
    self.non_sys_bits_2_len = len(non_sys_bits_2)

    self.debug.info_message("self.sys_bits_len: " + str(self.sys_bits_len) )
    self.debug.info_message("self.non_sys_bits_1_len: " + str(self.non_sys_bits_1_len) )
    self.debug.info_message("self.non_sys_bits_2_len: " + str(self.non_sys_bits_2_len) )

    self.debug.info_message("sys_bits: " + str(sys_bits) )
    self.debug.info_message("non_sys_bits_1: " + str(non_sys_bits_1) )
    self.debug.info_message("non_sys_bits_2: " + str(non_sys_bits_2) )


    sys.stdout.write("encoded message: " + str(codeword) + "\n")
    return codeword

  def decode(self, message):
    print("Turbo decode")

    Eb_N0_dB = 10
    Es_N0_dB = Eb_N0_dB + 10 * np.log10(1/3)
    noise_variance = 1/(2 * 10 ** (Es_N0_dB / 10 ))

    num_iterations = 10

    print("Turbo decode1")


    received_sys_symbols       = message[0:self.sys_bits_len]
    received_non_sys_sysbols_1 = message[self.sys_bits_len:self.sys_bits_len + self.non_sys_bits_1_len]
    received_non_sys_sysbols_2 = message[self.sys_bits_len + self.non_sys_bits_1_len:self.sys_bits_len + self.non_sys_bits_1_len + self.non_sys_bits_2_len]

    self.debug.info_message("len(received_sys_symbols): " + str(len(received_sys_symbols)) )
    self.debug.info_message("len(received_non_sys_sysbols_1): " + str(len(received_non_sys_sysbols_1)) )
    self.debug.info_message("len(received_non_sys_sysbols_2): " + str(len(received_non_sys_sysbols_2)) )

    self.debug.info_message("received_sys_symbols: " + str(received_sys_symbols) )
    self.debug.info_message("received_non_sys_sysbols_1: " + str(received_non_sys_sysbols_1) )
    self.debug.info_message("received_non_sys_sysbols_2: " + str(received_non_sys_sysbols_2) )



    print("Turbo decode2")


    decoded_bits = turbo_decode(received_sys_symbols, 
                                               received_non_sys_sysbols_1,
                                               received_non_sys_sysbols_2,
                                               self.trellis1,
                                               noise_variance,
                                               num_iterations,
                                               self.interleaver)

    print("Turbo decode3")

    sys.stdout.write("decoded message: " + str(decoded_bits) + "\n")
    return decoded_bits





class OsmodViterbi(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod


  def init_params(self):
    self.debug.info_message("Viterbi init params" )

    polynomial_override_checked = self.osmod.form_gui.window['cb_overridegeneratorpolynomials'].get()
    if polynomial_override_checked:
      gp1 = int(self.osmod.form_gui.window['in_fecgeneratorpolynomial1'].get())
      gp2 = int(self.osmod.form_gui.window['in_fecgeneratorpolynomial2'].get())
      gpdepth = int(self.osmod.form_gui.window['in_fecgeneratorpolynomialdepth'].get())
    else:
      gpdepth = self.osmod.fec_params[0]
      gp1     = self.osmod.fec_params[1]
      gp2     = self.osmod.fec_params[2]

    puncture_code = self.osmod.fec_params[3]
    if len(puncture_code) == 0:
      self.codec = Viterbi(gpdepth, [gp1, gp2])
    else:
      self.codec = Viterbi(gpdepth, [gp1, gp2], puncture_code )


  def encode(self, message):
    self.debug.info_message("Viterbi encode" )

    viterbi_encoded = self.codec.encode(message)

    return viterbi_encoded

  def decode(self, message):
    self.debug.info_message("Viterbi decode" )

    viterbi_decoded = self.codec.decode(message)

    return viterbi_decoded





class OsmodViterbi2(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod


  def init_params(self):
    self.debug.info_message("Viterbi2 init params" )
    self.memory = np.array([2])
    g_matrix = np.array([[0o7, 0o5]])
    self.trellis = Trellis(self.memory, g_matrix)


  def encode(self, message):
    self.debug.info_message("Viterbi2 encode" )
    viterbi_encoded = conv_encode(message, self.trellis)
    return viterbi_encoded

  def decode(self, message):
    self.debug.info_message("Viterbi2 decode" )
    depth=10
    viterbi_decoded = viterbi_decode(message.astype(float), self.trellis, tb_depth=depth*(self.memory[0]+1), decoding_type = 'hard')
    #start_index = 2*memory[0]
    return viterbi_decoded






class OsmodLDPC(object):

  debug  = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod  = None
  window = None
  values = None

  def __init__(self, osmod, window):  
    self.debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
    self.debug.info_message("__init__")
    self.osmod = osmod


  def init_params(self):
    """ set parameters for fixed frame message """
    """ source bits 641...total 800 """
    n = self.osmod.ldpc_params[0] #400
    d_v = self.osmod.ldpc_params[2] #2    # increasing this number redues number of source bits relative to fixed length message
    d_c = self.osmod.ldpc_params[3] #20
    self.seed = np.random.RandomState(42)
    self.H, self.G = make_ldpc(n, d_v, d_c, seed=self.seed, systematic=True, sparse=True)
    n, k = self.G.shape
    print("Number of coded bits:", k)
    self.n_trials = 1  # number of transmissions with different noise


  def encode(self, message):
    count =self.osmod.ldpc_params[1] - len(message)  #    361 - len(message)
    sys.stdout.write("fixed message length difference: " + str(count) + "\n")

    for i in range(0, count):
      message = np.append(message, 0)

    count = len(message)
    sys.stdout.write("fixed message length: " + str(count) + "\n")


    V = np.tile(message, (self.n_trials, 1)).T  # stack v in columns
    sys.stdout.write("fixed message pre ldpc: " + str(message) + "\n")
    sys.stdout.write("len fixed message pre ldpc: " + str(len(message)) + "\n")

    y = encode(self.G, V, 1000, seed=self.seed)
    y_binary = (y > 0).astype(int)

    y_binary_converted = y_binary.flatten()

    sys.stdout.write("converted binary encoded message: " + str(y_binary_converted) + "\n")
    #sys.stdout.write("binary encoded message: " + str(y_binary) + "\n")
    #sys.stdout.write("len binary encoded message: " + str(len(y_binary)) + "\n")

    return y_binary_converted

  def decode(self, y_binary):

    D = decode(self.H, y_binary, 2, self.osmod.ldpc_params[4])   #500)
    x = get_message(self.G, D[:])      #use if n_trials == 1
    sys.stdout.write("decoded message: " + str(x) + "\n")

    return x








