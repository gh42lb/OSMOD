#!/usr/bin/env python

import os
import sys
import math
import sounddevice as sd
import numpy as np
import debug as db
import constant as cn
import osmod_constant as ocn
#import matplotlib.pyplot as plt
from numpy import pi
from scipy.signal import butter, filtfilt, firwin, TransferFunction, lfilter, lfiltic
from modem_core_utils import ModemCoreUtils
from scipy import stats
from scipy.fft import fft, fftfreq
from collections import Counter
import ctypes
import cmath
from scipy import signal

from osmod_c_interface import ptoc_float_array, ptoc_double_array, ptoc_float, ctop_int, ptoc_int_array, ptoc_numpy_int_array, ptoc_double

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

class OsmodInterpolator(object):

  debug = db.Debug(ocn.DEBUG_OSMOD_MAIN)
  osmod = None
  window = None

  def __init__(self, osmod):  
    self.debug = db.Debug(ocn.DEBUG_INFO)
    self.debug.info_message("__init__")
    self.osmod = osmod

  def interpolatePulseTrain(self, pulse_train):
    self.debug.info_message("receive_pre_filters_interpolate")
    try:
      interpolated_lower  = []
      interpolated_higher = []
      half = int(self.osmod.pulses_per_block/2)

      """ if either of the interpolated lists is complete, fill out the other interpolated list if incomplete"""
      """
      def interpolateIfOneListComplete():
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        
        if len(interpolated_lower) == half and len(interpolated_higher) < half:
          for i in range(0,self.osmod.pulses_per_block):
            if i not in interpolated_lower and i not in interpolated_higher:
              interpolated_higher.append(i)
        elif len(interpolated_higher) == half and len(interpolated_lower) < half:
          for i in range(0,self.osmod.pulses_per_block):
            if i not in interpolated_lower and i not in interpolated_higher:
              interpolated_lower.append(i)
      """

      def interpolateCorrespondingListItems():
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        #if len(interpolated_lower) > len(interpolated_higher):
        for i in interpolated_lower:
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_higher and partner_index not in interpolated_lower:
            interpolated_higher.append(partner_index)
        #if len(interpolated_higher) > len(interpolated_lower):
        for i in interpolated_higher:
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_lower and partner_index not in interpolated_higher:
            interpolated_lower.append(partner_index)

      def interpolateListsAdjacent():
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        if interpolated_lower[0] == (interpolated_higher[-1] + 1) % self.osmod.pulses_per_block:
          if len(interpolated_lower) < half:
            for i in range(interpolated_lower[0], interpolated_lower[0] + int(self.osmod.pulses_per_block/2)):
              item = (i + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
              if item not in interpolated_lower:
                self.debug.info_message("interpolateListsAdjacent. Adding at 1")
                interpolated_lower.append(item)
                interpolated_higher.remove(item)
          if len(interpolated_higher) < half:
            for i in range(interpolated_higher[-1], interpolated_higher[-1] - int(self.osmod.pulses_per_block/2), -1):
              item = (i + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
              if item not in interpolated_higher:
                self.debug.info_message("interpolateListsAdjacent. Adding at 2")
                interpolated_higher.append(item)
                interpolated_lower.remove(item)

        elif interpolated_lower[-1] + 1 == interpolated_higher[0]:
          if len(interpolated_higher) < half:
            for i in range(interpolated_higher[0], interpolated_higher[0] + int(self.osmod.pulses_per_block/2)):
              item = (i + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
              if item not in interpolated_higher:
                self.debug.info_message("interpolateListsAdjacent. Adding at 3")
                interpolated_higher.append(item)
                interpolated_lower.remove(item)
          if len(interpolated_lower) < half:
            for i in range(interpolated_lower[-1], interpolated_lower[-1] - int(self.osmod.pulses_per_block/2), -1):
              item = (i + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
              if item not in interpolated_lower:
                self.debug.info_message("interpolateListsAdjacent. Adding at 4")
                interpolated_lower.append(item)
                interpolated_higher.remove(item)


      def interpolateSort():
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        interpolated_lower  = self.sort_interpolated(interpolated_lower)
        interpolated_higher = self.sort_interpolated(interpolated_higher)

      def interpolateContiguous():
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        interpolated_lower  = self.interpolate_contiguous_items(pulse_train[0])
        interpolated_higher = self.interpolate_contiguous_items(pulse_train[1])

      def removeDuplicatesInBothLists():
        nonlocal pulse_train
        in_both_lists = []
        for i in pulse_train[0]:
          if i in pulse_train[1] and i not in in_both_lists:
            in_both_lists.append(i)
        for i in pulse_train[1]:
          if i in pulse_train[0] and i not in in_both_lists:
            in_both_lists.append(i)
        self.debug.info_message("in_both_lists: " + str(in_both_lists))

        for i in in_both_lists:
          pulse_train[0].remove(i)
          pulse_train[1].remove(i)

      def calcShiftAmountLower():
        shift_amount_lower = 0
        completed = False
        for i in range(self.osmod.pulses_per_block - 1, half -1, -1):
          if i in interpolated_lower:
            shift_amount_lower = shift_amount_lower + 1
            completed = True
            self.debug.info_message("completed at 1")
          else:
            break

        if completed == False:
          for i in range(half, self.osmod.pulses_per_block):
            if i in interpolated_lower:
              shift_amount_lower = shift_amount_lower - 1
              completed = True
              self.debug.info_message("completed at 2")
            else:
              break

        if completed == False:
          for i in range(half, self.osmod.pulses_per_block):
            if i in interpolated_lower:
              shift_amount_lower = 0 - i
              completed = True
              self.debug.info_message("completed at 3")
              break

        if completed == False:
          for i in range(0, half):
            if i in interpolated_lower:
              shift_amount_lower = 0 - i
              completed = True
              self.debug.info_message("completed at 4")
              break

        if interpolated_lower[0] < half:
          return min(shift_amount_lower, 0-interpolated_lower[0])
        else:
          return shift_amount_lower

      def calcShiftAmountHigher():
        shift_amount_higher = 0
        completed = False
        for i in range(half -1, -1, -1):
          if i in interpolated_higher:
            shift_amount_higher = shift_amount_higher + 1
            completed = True
          else:
            break

        if completed == False:
          for i in range(0, half -1):
            if i in interpolated_higher:
              shift_amount_higher = shift_amount_higher - 1
              completed = True
            else:
              break

        if completed == False:
          for i in range(0, half):
            if i in interpolated_higher:
              shift_amount_higher = 0 - i - half
              completed = True
              break

        if completed == False:
          for i in range(half, self.osmod.pulses_per_block):
            if i in interpolated_higher:
              shift_amount_higher = half - i
              completed = True
              break

        if interpolated_higher[0] >= half:
          return min(shift_amount_higher, half-interpolated_higher[0])
        else:
          return shift_amount_higher

      def getShiftAmount():
        lower  = calcShiftAmountLower()
        higher = calcShiftAmountHigher()
        self.debug.info_message("shift_amount lower: " + str(lower))
        self.debug.info_message("shift_amount higher: " + str(higher))
        return max(lower,higher)

      def rebaseInterpolatedLower(shift_amount):
        rebased_interpolated_lower = []
        for i in interpolated_lower:
          rebased_item = (i + shift_amount + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
          if rebased_item <= half:
            rebased_interpolated_lower.append(rebased_item)
        return rebased_interpolated_lower

      def rebaseInterpolatedHigher(shift_amount):
        rebased_interpolated_higher = []
        for i in interpolated_higher:
          rebased_item = (i + shift_amount + self.osmod.pulses_per_block) % self.osmod.pulses_per_block
          if rebased_item >= half:
            rebased_interpolated_higher.append(rebased_item)
        return rebased_interpolated_higher

      def rebaseInterpolated(shift_amount):
        nonlocal interpolated_lower
        nonlocal interpolated_higher
        interpolated_lower  = rebaseInterpolatedLower(shift_amount)
        interpolated_higher = rebaseInterpolatedHigher(shift_amount)


      self.debug.info_message("Start Interpolate")
      self.debug.info_message("pulse_train[0]: " + str(pulse_train[0]))
      self.debug.info_message("pulse_train[1]: " + str(pulse_train[1]))

      removeDuplicatesInBothLists()
      self.debug.info_message("removeDuplicatesInBothLists")
      self.debug.info_message("pulse_train[0]: " + str(pulse_train[0]))
      self.debug.info_message("pulse_train[1]: " + str(pulse_train[1]))

      interpolateContiguous()
      self.debug.info_message("interpolateContiguous")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      """
      interpolateIfOneListComplete()
      self.debug.info_message("interpolateIfOneListComplete")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      interpolateSort()
      self.debug.info_message("interpolateSort")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))
      """

      interpolateCorrespondingListItems()
      interpolateCorrespondingListItems()
      self.debug.info_message("interpolateCorrespondingListItems")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      interpolateSort()
      self.debug.info_message("interpolateSort")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      interpolateListsAdjacent()
      self.debug.info_message("interpolateListsAdjacent")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      interpolateSort()
      self.debug.info_message("interpolateSort")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      shift_amount = getShiftAmount()
      self.debug.info_message("getShiftAmount")
      self.debug.info_message("shift_amount: "  + str(shift_amount))

      rebaseInterpolated(shift_amount)
      self.debug.info_message("rebaseInterpolated")
      self.debug.info_message("interpolated_lower: "  + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))



      return interpolated_lower, interpolated_higher, shift_amount

    except:
      self.debug.error_message("Exception in receive_pre_filters_interpolate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def derivePersistentLists(self, pulse_start_index, fft_filtered, frequency):
    self.debug.info_message("derivePersistentLists")
    try:
      #max_metric = 0
      #best_list_factor = 0
      # 0.936 is best or 0.93
      #"""

      #for list_factor in np.arange(0.936,0.92,-0.004):
      for list_factor in np.arange(0.89,0.5,-0.1):
        self.test_list_factor = list_factor

        ret_values = self.osmod.detector.detectStandingWavePulseNew(fft_filtered, frequency, pulse_start_index, 0, ocn.LOCATE_PULSE_TRAIN)
        persistent_lower  = ret_values[1]

        ret_values = self.osmod.detector.detectStandingWavePulseNew(fft_filtered, frequency, pulse_start_index, 1, ocn.LOCATE_PULSE_TRAIN)
        persistent_higher = ret_values[1]

        in_both_lists = []
        for i in persistent_lower:
          if i in persistent_higher and i not in in_both_lists:
            in_both_lists.append(i)
        for i in persistent_higher:
          if i in persistent_lower and i not in in_both_lists:
            in_both_lists.append(i)
        self.debug.info_message("in_both_lists: " + str(in_both_lists))

        metric_lower = (len(persistent_lower) - len(in_both_lists))
        metric_upper = (len(persistent_higher) - len(in_both_lists))
        metric = metric_lower + metric_upper

        #if metric > max_metric:
        #  best_list_factor = list_factor
        #  max_metric = metric
        if metric_lower >= 5 and metric_upper >= 5:
          break

      #self.debug.info_message("best_list_factor: " + str(best_list_factor))

      #self.osmod.test_list_factor = best_list_factor
      #"""
      #self.test_list_factor = 0.936

      #ret_values = self.osmod.detector.detectStandingWavePulseNew(fft_filtered, frequency, pulse_start_index, 0, ocn.LOCATE_PULSE_TRAIN)
      #persistent_lower  = ret_values[1]

      #ret_values = self.osmod.detector.detectStandingWavePulseNew(fft_filtered, frequency, pulse_start_index, 1, ocn.LOCATE_PULSE_TRAIN)
      #persistent_higher = ret_values[1]

      return persistent_lower, persistent_higher

    except:
      self.debug.error_message("Exception in derivePersistentLists: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def receive_pre_filters_interpolate(self, pulse_start_index, audio_block, frequency, fft_filtered):
    self.debug.info_message("receive_pre_filters_interpolate")
    try:
      pulse_length      = int((self.osmod.symbol_block_size / self.osmod.pulses_per_block))
      pulse_end_index   = int(pulse_start_index + pulse_length)

      fft_filtered_lower  = fft_filtered[0]
      fft_filtered_higher  = fft_filtered[1]

      self.osmod.getDurationAndReset('bandpass_filter_fft')

      max_occurrences_lower, where_lower    = self.testCodeGetFrequencySectionStart(fft_filtered_lower, pulse_start_index)
      max_occurrences_higher, where_higher  = self.testCodeGetFrequencySectionStart(fft_filtered_higher, pulse_start_index)
      self.debug.info_message("max_occurrences_lower: " + str(max_occurrences_lower))
      self.debug.info_message("max_occurrences_higher: " + str(max_occurrences_higher))

      self.osmod.getDurationAndReset('testCodeGetFrequencySectionStart')

      max_occurrences_lists = self.removeConflictingItemsTwoList([max_occurrences_lower, max_occurrences_higher])
      max_occurrences_lower  = max_occurrences_lists[0]
      max_occurrences_higher = max_occurrences_lists[1]

      self.osmod.getDurationAndReset('removeConflictingItemsTwoList')

      self.debug.info_message("max_occurrences_lower 2: " + str(max_occurrences_lower))
      self.debug.info_message("max_occurrences_higher 2: " + str(max_occurrences_higher))

      interpolated_lower  = self.interpolate_contiguous_items(max_occurrences_lower)
      interpolated_higher = self.interpolate_contiguous_items(max_occurrences_higher)

      self.osmod.getDurationAndReset('interpolate_contiguous_items')

      self.debug.info_message("interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher: " + str(interpolated_higher))

      half = int(self.osmod.pulses_per_block/2)

      """ if either of the inetrpolated lists is complete, fill out the other interpolated list if incomplete"""
      if len(interpolated_lower) == half and len(interpolated_higher) < half:
        for i in range(0,self.osmod.pulses_per_block):
          if i not in interpolated_lower and i not in interpolated_higher:
            interpolated_higher.append(i)
      elif len(interpolated_higher) == half and len(interpolated_lower) < half:
        for i in range(0,self.osmod.pulses_per_block):
          if i not in interpolated_lower and i not in interpolated_higher:
            interpolated_lower.append(i)

      self.debug.info_message("interpolated_lower 3: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher 3: " + str(interpolated_higher))

      """ if either of the inetrpolated lists is greater than the other list, fill out the other interpolated list"""
      interpolated_lower  = self.sort_interpolated(interpolated_lower)
      interpolated_higher = self.sort_interpolated(interpolated_higher)

      if len(interpolated_lower) > len(interpolated_higher):
        for i in range(interpolated_lower[0],interpolated_lower[-1]):
        #for i in interpolated_lower:
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_higher:
            interpolated_higher.append(partner_index)
      if len(interpolated_higher) > len(interpolated_lower):
        for i in range(interpolated_higher[0],interpolated_higher[-1]):
        #for i in interpolated_higher:
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_lower:
            interpolated_lower.append(partner_index)
      """ 
      if len(interpolated_lower) > len(interpolated_higher):
        for i in range(interpolated_lower[0],interpolated_lower[-1]):
          partner_offset = self.osmod.pulses_per_block / self.osmod.num_carriers
          partner_index = int((i + partner_offset) % self.osmod.pulses_per_block)
          if partner_index not in interpolated_higher:
            interpolated_higher.append(partner_index)
      """
      self.debug.info_message("interpolated_lower 4: " + str(interpolated_lower))
      self.debug.info_message("interpolated_higher 4: " + str(interpolated_higher))

      interpolated_lower  = self.sort_interpolated(interpolated_lower)
      interpolated_higher = self.sort_interpolated(interpolated_higher)


      if len(interpolated_lower) == half:
        first_value = self.get_first_interpolated(interpolated_lower)
        self.block_start_candidates.append(first_value)

      self.osmod.getDurationAndReset('interpolate_contiguous_items 2')


      self.debug.info_message("modified interpolated_lower: " + str(interpolated_lower))
      self.debug.info_message("modified interpolated_higher: " + str(interpolated_higher))


      self.osmod.form_gui.window['text_indices_lower'].update(str(interpolated_lower))
      self.osmod.form_gui.window['text_indices_higher'].update(str(interpolated_higher))


    except:
      self.debug.error_message("Exception in receive_pre_filters_interpolate: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return self.sort_interpolated(interpolated_lower), self.sort_interpolated(interpolated_higher)





  """ contiguous by distance for and wrap around"""
  def interpolate_contiguous_items(self, list_items):
    self.debug.info_message("interpolate_contiguous_items")
    try:
      limit_value = int((self.osmod.pulses_per_block / self.osmod.num_carriers) / 2)
      self.debug.info_message("pulses_per_block: " + str(self.osmod.pulses_per_block))
      self.debug.info_message("num_carriers: " + str(self.osmod.num_carriers))
      self.debug.info_message("limit_value: " + str(limit_value))
      saved_list_items = list_items
      have_suspect_items = True
      while have_suspect_items:
        have_suspect_items = False
        median_input_list = int(np.median(np.array(list_items)))
        self.debug.info_message("median_input_list: " + str(median_input_list))
        for i in list_items:
          self.debug.info_message("distance: " + str(abs(median_input_list - i)))
          if abs(median_input_list - i) > limit_value:
            self.debug.info_message("suspect item: " + str(i))
            list_items.remove(i)
            have_suspect_items = True

      median_input_list_truth = median_input_list

      """ now repeat process using final truth value as median """
      list_items = saved_list_items
      for i in list_items:
        self.debug.info_message("distance from truth: " + str(abs(median_input_list_truth - i)))
        if abs(median_input_list_truth - i) > int((self.osmod.pulses_per_block / self.osmod.num_carriers) / 2):
          self.debug.info_message("suspect item from truth: " + str(i))
          list_items.remove(i)


      half = int(self.osmod.pulses_per_block / 2)

      start_value = list_items[0]
      min_value = 0
      max_value = 0
      walk = 0
      for i, j in zip(list_items, list_items[1:]):
        if abs(i - j) < half:
          walk = walk - (i - j)
        elif abs(j - i) < half:
          walk = walk + (j - i)
        elif abs(j + self.osmod.pulses_per_block - i) < half:
          walk = walk + (j + self.osmod.pulses_per_block - i)
        elif abs(i + self.osmod.pulses_per_block - j) < half:
          walk = walk - (i + self.osmod.pulses_per_block - j)

        if walk > max_value:
          max_value = walk
        if walk < min_value:
          min_value = walk

        self.debug.info_message("walk: " + str(walk))

      self.debug.info_message("min: " + str(min_value))
      self.debug.info_message("max: " + str(max_value))

      interpolated_list = []
      for x in range(start_value + min_value, start_value + max_value + 1):
        interpolated_list.append(x % self.osmod.pulses_per_block)

      self.debug.info_message("Original list: " + str(list_items))
      self.debug.info_message("Interpolated list: " + str(interpolated_list))

      return interpolated_list

    except:
      sys.stdout.write("Exception in interpolate_contiguous_items: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def removeConflictingItemsTwoList(self, max_occurrences_lists):
    self.debug.info_message("removeConflictingItemsTwoList")
    try:

      max_occurrences_lower = max_occurrences_lists[0]
      max_occurrences_higher = max_occurrences_lists[1]

      in_both_lists = []
      for i in max_occurrences_lower:
        if i in max_occurrences_higher and i not in in_both_lists:
          in_both_lists.append(i)
      for i in max_occurrences_higher:
        if i in max_occurrences_lower and i not in in_both_lists:
          in_both_lists.append(i)
      self.debug.info_message("in_both_lists: " + str(in_both_lists))

      for i in in_both_lists:
        max_occurrences_lower.remove(i)
        max_occurrences_higher.remove(i)

    except:
      self.debug.error_message("Exception in removeConflictingItemsTwoList: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    return [max_occurrences_lower, max_occurrences_higher]

  """ min can be a higher numeric value but is defined as the first in the series"""
  def get_first_interpolated(self, interpolated_lower):
    self.debug.info_message("get_first_interpolated")
    try:
      """ does the data wrap around? """
      """ no wrap"""
      if self.osmod.pulses_per_block - 1 not in interpolated_lower:
        first_interpolated = min(interpolated_lower)
        self.debug.info_message("first_interpolated: " + str(first_interpolated))
        return (first_interpolated)

      """ the data does wrap around"""
      half = int(self.osmod.pulses_per_block/2)
      normalized_list = []
      for item in interpolated_lower:
        if item < half:
          normalized_list.append(item + self.osmod.pulses_per_block)
        else:
          normalized_list.append(item)

      first_interpolated = min(normalized_list) % self.osmod.pulses_per_block
      self.debug.info_message("first_interpolated: " + str(first_interpolated))
      return (first_interpolated)
    except:
      self.debug.error_message("Exception in get_first_interpolated: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

  def sort_interpolated(self, interpolated_lower):
    #self.debug.info_message("sort_interpolated")
    try:
      half = int(self.osmod.pulses_per_block/2)
      normalized_list = []
      restored_sorted_list = []
      for item in interpolated_lower:
        if item < half:
          normalized_list.append(item + self.osmod.pulses_per_block)
        else:
          normalized_list.append(item)

      #self.debug.info_message("normalized_list: " + str(normalized_list))
      normalized_list.sort()
      #self.debug.info_message("normalized_list sorted: " + str(normalized_list))

      """ does the data wrap around? """
      if self.osmod.pulses_per_block - 1 in normalized_list:
        for item in normalized_list:
          if item < self.osmod.pulses_per_block:
            restored_sorted_list.append(item)
        for item in normalized_list:
          if item >= self.osmod.pulses_per_block:
            restored_sorted_list.append(item % self.osmod.pulses_per_block)
        #self.debug.info_message("restored_sorted_list: " + str(restored_sorted_list))
 
        #self.osmod.getDurationAndReset('sort_interpolated')

        return restored_sorted_list
      else:
        for item in normalized_list:
          if item >= self.osmod.pulses_per_block:
            restored_sorted_list.append(item % self.osmod.pulses_per_block)
        for item in normalized_list:
          if item < self.osmod.pulses_per_block:
            restored_sorted_list.append(item)
        #self.debug.info_message("restored_sorted_list: " + str(restored_sorted_list))

        #self.osmod.getDurationAndReset('sort_interpolated')

        return restored_sorted_list

    except:
      self.debug.error_message("Exception in sort_interpolated: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))



  def testCodeGetFrequencySectionStart(self, signal, pulse_start_index):
    self.debug.info_message("testCodeGetFrequencySectionStart")

    try:
      if self.osmod.use_compiled_c_code == True:
        self.debug.info_message("calling compiled C code")
        self.debug.info_message("signal type is: " + str(signal.dtype))

        half = int(self.osmod.pulses_per_block / 2);
        results = np.array([0] * half, dtype = np.int32)
        c_results = ptoc_numpy_int_array(results)

        self.osmod.compiled_lib.test_code_get_frequency_section_start.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags = 'C'), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.osmod.compiled_lib.test_code_get_frequency_section_start.restype = ctypes.c_int
        num_results = self.osmod.compiled_lib.test_code_get_frequency_section_start(signal, signal.size, self.osmod.parameters[5], self.osmod.symbol_block_size, self.osmod.pulses_per_block, pulse_start_index, c_results)

        return results[:num_results].tolist(), {}

      else:
        return self.testCodeGetFrequencySectionStartInterpretedPython(signal, pulse_start_index)

    except:
      self.debug.error_message("Exception in testCodeGetFrequencySectionStart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  
  def testCodeGetFrequencySectionStartInterpretedPython(self, signal, pulse_start_index):
    self.debug.info_message("testCodeGetFrequencySectionStart" )
    try:
      all_list = []
      all_list_ab = []
      all_dict_where = {}

      pulse_width = self.osmod.symbol_block_size / self.osmod.pulses_per_block

      for i in range(0, int(len(signal) // self.osmod.symbol_block_size)): 
        #self.debug.info_message("getting test peak")
        test_peak = signal[i*self.osmod.symbol_block_size:(i*self.osmod.symbol_block_size) + self.osmod.symbol_block_size]
        test_max = np.max(test_peak)
        test_min = np.min(test_peak)
        self.debug.info_message("python min and max values" + str(test_min) + " " + str(test_max))
        max_indices = np.where((test_peak*(100/test_max)) > self.osmod.parameters[5])
        min_indices = np.where((test_peak*(100/test_min)) > self.osmod.parameters[5])
        self.debug.info_message("python min and max indices " + str(min_indices) + " " + str(max_indices))
        for x in range(0, len(max_indices[0]) ):
          all_list.append(max_indices[0][x] % self.osmod.symbol_block_size)
          sequence_value = int((max_indices[0][x] % self.osmod.symbol_block_size) // (self.osmod.symbol_block_size/self.osmod.pulses_per_block))
          all_list_ab.append(sequence_value)
          """ where == which pulse numerically from the start"""
          all_dict_where[sequence_value] = max_indices[0][x] % (self.osmod.symbol_block_size // self.osmod.pulses_per_block)
        for x in range(0, len(min_indices[0]) ):
          all_list.append(min_indices[0][x] % self.osmod.symbol_block_size)
          sequence_value = int((min_indices[0][x] % self.osmod.symbol_block_size) // (self.osmod.symbol_block_size/self.osmod.pulses_per_block))
          all_list_ab.append(sequence_value)
          """ where == which pulse numerically from the start"""
          all_dict_where[sequence_value] = min_indices[0][x] % (self.osmod.symbol_block_size // self.osmod.pulses_per_block)
      #self.debug.info_message("Frequency Section all indices: " + str(all_list))
      #self.debug.info_message("Frequency Section all ab indices: " + str(all_list_ab))

      half = int(self.osmod.pulses_per_block / 2)

      max_occurrences = []
      for count in range(0,half):
        max_occurrences_item = self.count_max_occurrences(all_list_ab, max_occurrences)
        if max_occurrences_item != []:
          max_occurrences.append(max_occurrences_item[0])

      median_index_all = int(np.median(np.array(all_list)))
      self.debug.info_message("Frequency Section Filter mean all index: " + str(median_index_all))

      sample_start = median_index_all
      self.debug.info_message("Frequency Setion start: " + str(sample_start))

    except:
      sys.stdout.write("Exception in testCodeGetFrequencySectionStart: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

    return max_occurrences, all_dict_where


