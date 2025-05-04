#!/usr/bin/env python
import sys
import constant as cn
import json
import os
import debug as db

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

class PersistentData(object):

  """
  debug level 0=off, 1=info, 2=warning, 3=error
  """
  def __init__(self, osmod):  
    self.debug = db.Debug(cn.DEBUG_INFO)
    self.main_settings = self.readMainDictionaryFromFile("osmod_main_settings.txt")
    self.osmod = osmod
    return


  def saveMainSettings(self, values):
    self.writeMainDictionaryToFile("osmod_main_settings.txt", values)


  def createMainDictionaryDefaults(self):
    self.debug.info_message("createMainDictionaryDefaults")
    settings = { 'params': {
                           'LB28-256-2-10-I'       : {'amplitude' : 1.0, 'AWGN' : 8.0, 'symbol_block_size': 51200, 'carrier_separation': 10, 'parameters': (600, 0.70, 0.9, 10000, 2, 98)},
               }           }

    return settings


  """
  Main application settings dictionary
  """
  def readMainDictionaryFromFile(self, filename):
    self.debug.info_message("readMainDictionaryFromFile")

    settings_defaults = self.createMainDictionaryDefaults()
    params_default = settings_defaults.get("params")

    try:
      with open(filename) as f:
        data = f.read()
 
      settings = json.loads(data)

      params_read    = settings.get("params")

      for key in params_default: 
        value = params_default.get(key)
        if(key not in params_read):
          """ upgrade the file format set the new parama to a default value"""
          params_read[key]=value
          self.debug.info_message("UPGRADING FILE FORMAT")
          self.debug.info_message("adding key: " + str(key) )

    except:
      self.debug.error_message("Settings file not found. Setting to default values: readMainDictionaryFromFile: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      settings = settings_defaults

    return(settings)



  """ writes the messages to the messages file """
  def writeMainDictionaryToFile(self, filename, values):
    self.debug.info_message("writeMainDictionaryToFile")

    try:
      details = self.main_settings
      params  = details.get("params")

      """ individual fields first """	  
      details = { 'params': {
                             'LB28-2-2-100-N'         : self.osmod.getPersistentData('LB28-2-2-100-N',   values),
                             'LB28-160-2-2-100-N'     : self.osmod.getPersistentData('LB28-160-2-2-100-N',   values),
                             'LB28-240-2-2-100-N'     : self.osmod.getPersistentData('LB28-240-2-2-100-N',   values),
                             'LB28-2-2-100-N'         : self.osmod.getPersistentData('LB28-2-2-100-N',   values),
                             'LB28-160-4-2-100-N'      : self.osmod.getPersistentData('LB28-160-4-2-100-N',    values),
                             'LB28-160-4-2-50-N'      : self.osmod.getPersistentData('LB28-160-4-2-50-N',    values),
                             'LB28-4-2-40-N'          : self.osmod.getPersistentData('LB28-4-2-40-N',    values),
                             'LB28-4-2-20-N'          : self.osmod.getPersistentData('LB28-4-2-20-N',    values),
                             'LB28-320-8-2-50-N'          : self.osmod.getPersistentData('LB28-320-8-2-50-N',    values),
                             'LB28-8-2-10-N'          : self.osmod.getPersistentData('LB28-8-2-10-N',    values),
                             'LB28-16-2-10-I'         : self.osmod.getPersistentData('LB28-16-2-10-I',   values),
                             'LB28-16-2-15-I'         : self.osmod.getPersistentData('LB28-16-2-15-I',   values),
                             'LB28-3200-32-2-15-I'    : self.osmod.getPersistentData('LB28-3200-32-2-15-I',   values),
                             'LB28-32-2-10-I'         : self.osmod.getPersistentData('LB28-32-2-10-I',   values),
                             'LB28-6400-64-2-15-I'    : self.osmod.getPersistentData('LB28-6400-64-2-15-I',   values),
                             'LB28-64-2-15-I'         : self.osmod.getPersistentData('LB28-64-2-15-I',   values),
                             'LB28-64-2-10-I'         : self.osmod.getPersistentData('LB28-64-2-10-I',   values),
                             'LB28-6400-128-2-15-I'        : self.osmod.getPersistentData('LB28-6400-128-2-15-I',  values),
                             'LB28-128-2-15-I'        : self.osmod.getPersistentData('LB28-128-2-15-I',  values),
                             'LB28-128-2-10-I'        : self.osmod.getPersistentData('LB28-128-2-10-I',  values),
                             'LB28-25600-256-2-15-I'  : self.osmod.getPersistentData('LB28-25600-256-2-15-I',  values),
                             'LB28-256-2-15-I'        : self.osmod.getPersistentData('LB28-256-2-15-I',  values),
                             'LB28-256-2-10-I'        : self.osmod.getPersistentData('LB28-256-2-10-I',  values),
                             'LB28-25600-512-2-15-I'        : self.osmod.getPersistentData('LB28-25600-512-2-15-I',  values),
                             'LB28-51200-512-2-15-I'        : self.osmod.getPersistentData('LB28-51200-512-2-15-I',  values),
                             'LB28-512-2-15-I'        : self.osmod.getPersistentData('LB28-512-2-15-I',  values),
                             'LB28-512-2-10-I'        : self.osmod.getPersistentData('LB28-512-2-10-I',  values),
                             'LB28-1024-2-15-I'       : self.osmod.getPersistentData('LB28-1024-2-15-I', values),
                             'LB28-51200-1024-2-15-I' : self.osmod.getPersistentData('LB28-51200-1024-2-15-I', values),
                             'LB28-102400-1024-2-15-I'       : self.osmod.getPersistentData('LB28-102400-1024-2-15-I', values),
                             'LB28-1024-2-10-I'       : self.osmod.getPersistentData('LB28-1024-2-10-I', values),
                             'LB28-2048-2-15-I'       : self.osmod.getPersistentData('LB28-2048-2-15-I', values),
                             'LB28-102400-2048-2-15-I'       : self.osmod.getPersistentData('LB28-102400-2048-2-15-I', values),
                             'LB28-204800-2048-2-15-I'       : self.osmod.getPersistentData('LB28-204800-2048-2-15-I', values),
                             'LB28-2048-2-10-I'       : self.osmod.getPersistentData('LB28-2048-2-10-I', values),
                }           }

      with open(filename, 'w') as convert_file:
                convert_file.write(json.dumps(details))
      return()

    except:
      self.debug.error_message("Exception in test1: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



