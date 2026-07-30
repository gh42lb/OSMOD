#!/usr/bin/env python

import FreeSimpleGUI as sg

import sys
import debug as db
import threading
import json
import constant as cn
import osmod_net_gui
import osmod_net_events
import random
import getopt
import osmod_net_parser
import os
import platform

from datetime import datetime, timedelta
from datetime import time

"""
MIT License

Copyright (c) 2022 Lawrence Byng

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


"""
The main/high level logic for the lb28 net app goes in the OSMOD_Net class
"""
class OSMOD_Net(object):

  roster = [] 
  timeout=300
  max_timeout = 300
  flashtimer=0
  flashstate=0

  def __init__(self, debug, view, window, osmod_client):  
    self.window = window 
    self.view = view
    self.osmod_client = osmod_client
    self.flash_state = 0
    self.flash_win = None
    self.station_call_sign = ""
    self.saved_send_string = ""
    self.auto_checkin = False
    self.ncs_data = None
    self.seqnum=0
    self.outgoing_awaiting_response = False
    self.flashing_state = False
    self.edit_mode = False
    self.edit_state = cn.SAVED
    self.operating_mode = cn.NETCONTROL
    self.net_started = False
    self.utc_time_now = None
    self.time_now = None
    self.debug = debug
    self.main_offset = 0
    self.sidebar_offset = 0
    self.ncs_hostname = ""
    self.rosterStandbyCalls = ""
    self.current_round = ""
    self.manual_frequency = ""
    self.manual_group = ""
    self.freq_from_osmod = False
    self.side_main_offset_boundary = 700
    self.offsets_list = '1337,700,870,1140,1210,750,920,1190,1260,800,970,1240'
    self.update_freq_on_qsy = False
    self.timer_call = ""
    self.parser = osmod_net_parser.NetParser(debug, view, window, osmod_client, self)


  def setUpdateFreqOnQsy(self, value):
    self.update_freq_on_qsy = value

  def getUpdateFreqOnQsy(self):
    return self.update_freq_on_qsy

  def setOffsetsList(self, offsets_list):
    self.offsets_list = offsets_list

  def getOffsetsList(self):
    return self.offsets_list

  def getSideMainOffsetBoundary(self):
    return self.side_main_offset_boundary

  def setSideMainOffsetBoundary(self, boundary):
    self.side_main_offset_boundary = boundary

  def getFreqFromOSMOD(self):
    return self.freq_from_osmod

  def setFreqFromOSMOD(self, yesno):
    self.freq_from_osmod = yesno
    return

  def setManualFrequency(self, freqstr):
    self.manual_frequency = freqstr

  def getManualFrequency(self):
    return self.manual_frequency

  def setManualGroup(self, grpstr):
    self.manual_group = grpstr

  def getManualGroup(self):
    return self.manual_group

  def setCurrentRound(self, roundstr):
    self.current_round = roundstr

  def getCurrentRound(self):
    return self.current_round

  def setRosterStandbyCalls(self, callstring):
    self.rosterStandbyCalls = callstring

  def getRosterStandbyCalls(self):
    return self.rosterStandbyCalls

  def getStationCallSign(self):
    return self.station_call_sign

  def getNcsHostName(self):
    return self.ncs_hostname
  
  def setOperatingMode(self, mode):
    self.operating_mode = mode

  def getOperatingMode(self):
    return self.operating_mode

  def setEditState(self, state):
    self.edit_state = state
    
  def getEditState(self):
    return self.edit_state

  def toggleEditState(self):
    if(self.edit_state == cn.SAVED):
      self.edit_state = cn.EDIT
    else:
      self.edit_state = cn.SAVED
      
  def setEditMode(self, state):
    self.edit_mode = state
    return
    
  def getEditMode(self):
    return self.edit_mode

  def setFlashingState(self, state):
    self.flashing_state = state

  def getFlashingState(self):
    return self.flashing_state

  """ these fields are used for the slider control """
  def setOutgoingAwaitingResponse(self, awaiting_resp):
    self.outgoing_awaiting_response = awaiting_resp

  def getOutgoingAwaitingResponse(self):
    return self.outgoing_awaiting_response

  def setNcsData(self, js):
    self.ncs_data = js

  def getNcsData(self):
    return self.ncs_data
    
  def setSeqnum(self, value):
    self.seqnum=value

  def getSeqnum(self):
    return self.seqnum

  def setAutoCheckin(self, ckin_type):
    self.auto_checkin = ckin_type

  def getAutoCheckin(self):
    return self.auto_checkin

  def incRound(self):
    retval=""	  
    current_round = self.window['option_currentround'].get().strip()
    total_rounds = self.window['input_rounds'].get().strip()

    if(current_round != total_rounds ):
      if(current_round == "ONE"):
        self.window['option_currentround'].Update("TWO")	
        retval = "TWO"
      elif(current_round == "TWO"):
        self.window['option_currentround'].Update("THREE")	
        retval = "THREE"
      elif(current_round == "THREE"):
        self.window['option_currentround'].Update("FOUR")	
        retval = "FOUR"
      elif(current_round == "FOUR"):
        self.window['option_currentround'].Update("FIVE")	
        retval = "FIVE"
      elif(current_round == "FIVE"):
        self.window['option_currentround'].Update("SIX")	
        retval = "SIX"

    return retval
 
  def processBtuOverTo(self):
    my_call = self.getStationCallSign()
    nettype = self.window['option_menu'].get()
    if(self.getOperatingMode() == cn.PARTICIPANT):
      if(nettype == "Directed"):
        return " BTU"			  
      elif(nettype == "Round Table"):
        next_stn = self.getNextStandbyStation(my_call, True, True)
        if(next_stn == ""):     
          return " BACK TO NET"	
        else:
          return " NEXT OVER TO " + next_stn	
      else:
        return ""
    elif(self.getOperatingMode() == cn.NETCONTROL):
      return ""
 
  def setKnownCalls(self, call_list):
    self.knownCalls = call_list

  def getKnownCalls(self):
    return self.knownCalls
    
  def nameFromSavedCalls(self, lookup_call):
    self.debug.info_message("nameFromSavedCalls")

    try:
      allcalls = self.knownCalls
      for x in range(len(allcalls)):
        if(allcalls[x].split(' ')[0]==lookup_call):
          return allcalls[x].split(' ')[1].strip()
    except:
      self.debug.error_message("method: nameFromSavedCalls. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
   
    if(len(lookup_call)>2):     
      alias = lookup_call[len(lookup_call)-3] + lookup_call[len(lookup_call)-2] + lookup_call[len(lookup_call)-1]
      return alias
    else:
      return("-")

  def updateSavedCalls(self, call, name):
    found=-1
    try:
      allcalls = self.knownCalls
      for x in range(len(allcalls)):
        if(allcalls[x].split(' ')[0]==call):
          found = x			
          self.updateSavedCall(x, call, name)
    except:
      self.debug.error_message("method: nameFromSavedCalls. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    if(found == -1):
      self.appendSavedCall(call, name)		
        
    return("-")

  def appendSavedCall(self, call, name):
    self.setKnownCalls( self.getKnownCalls() + [call + " " + name] )
    return()

  def updateSavedCall(self, index, call, name):
    self.knownCalls[index] = call + " " + name
    return()

  """
  This method is used by the client stations to update roster status information
  this is at the heart of the state machine
  status can be <----> <STANDBY> <TALKING> <DONE> <SKIP> <NEXT>  <QRT> <SWL>
  progression is:
  <STANDBY> (empty) --> <NEXT>
  <TALKING> --> <DONE>  + set next on roster as <NEXT>
  <TALKING> --> <QRT>
  <NEXT> --> <TALKING>  or <SKIP>  
  <SKIP> --> <NEXT> or  <QRT>
  there can only be one <NEXT> station
  there can only be one <TALKING> station
  """
  def incrementStatus(self, indexNext, indexTalking, indexTimeout):

    self.debug.info_message("incrementing status")

    if(indexTalking != -1):
      """ only update the field if this is an existing roster station and not a new checkin """
      status = self.roster[indexTalking].split(" ")[2]
      last_call   = self.roster[indexTalking].split(" ")[0]
      if(status != "<HEARD>"):
        self.view.updateCallNameStatus(None, last_call, '', '')
        
    updatedNext = 0
    for x in range(len(self.roster)):
      
      split_string = self.roster[x].split(" ")
      call   = split_string[0]
      name   = split_string[1]
      status = split_string[2]
      
      """ update any talking stations to <DONE> status"""
      if indexNext != -1 and status == "<TALKING>":
        self.debug.info_message("incrementing status to <DONE>")
        self.updateRoster(x, call, name, '<DONE>')

      """ station did not talk so skip """
      if(indexNext != -1 and x != indexNext and status == "<NEXT>" ):
        self.debug.info_message("incrementing status to <SKIP>")
        self.updateRoster(x, call, name, '<SKIP>')
      
      if(x == indexNext and updatedNext == 0):
        if(status == "<STANDBY>" or status == "<SKIP>"):
          self.debug.info_message("incrementing status to <NEXT>")
          self.updateRoster(x, call, name, '<NEXT>')
          updatedNext=1
      """ station with <NEXT> is is the only station of interested at this stage of the proceedings all others are ignored. """    
      if(x == indexTalking and status == "<NEXT>"):
        self.debug.info_message("incrementing status to <TALKING>")
        self.updateRoster(x, call, name, '<TALKING>')
        self.updatePrevNext(call)
      if(status == "<NEXT>" and x == indexTimeout):
        self.debug.info_message("incrementing status to <SKIP>")
        self.updateRoster(x, call, name, '<SKIP>')
        
    return()

  def setAsNext(self, nextcall):
    for x in range(1, len(self.roster)):
      split_string = self.roster[x].split(" ")
      call   = split_string[0]
      name   = split_string[1]
      status = split_string[2]
      if status == "<TALKING>" :
        self.debug.info_message("setAsNext updating status for " + call + " to <DONE>")
        self.updateRoster(x, call, name, '<DONE>')

      if call == nextcall:
        if status == "<STANDBY>" or status == "<SKIP>":
          self.debug.info_message("setAsNext updating status for " + call + " to <NEXT>")
          self.updateRoster(x, call, name, '<NEXT>')
          return
  
  def getNextStandbyStation(self, currentCall, resetTalking, resetNext):
    standbyCall = ""
    skipIndex=-1
    for x in range(1, len(self.roster)):
      split_string = self.roster[x].split(" ")
      call   = split_string[0]
      name   = split_string[1]
      status = split_string[2]
 
      if status == "<TALKING>" and resetTalking and call != currentCall:
        self.debug.info_message("getNextStandbyStation. Updating status for " + call + " to <DONE>")
        self.updateRoster(x, call, name, '<DONE>')
 
      elif status == "<NEXT>"  and resetNext:
        self.debug.info_message("getNextStandbyStation. Updating status for " + call + " to <SKIP>")
        self.updateRoster(x, call, name, '<SKIP>')
 
      elif status == "<SKIP>" and skipIndex == -1:
        """make a note of the first skipped station for later"""		  
        if call != currentCall:
          skipIndex=x
 
      elif status == "<STANDBY>" and standbyCall == "":
        self.debug.info_message("getNextStandbyStation. Updating status for " + call + " to <NEXT>")
        self.updateRoster(x, call, name, '<NEXT>')
        self.view.updatePrevField(currentCall)
        self.view.updateNextField(call)
        standbyCall = call
        self.view.updateCallNameStatus(None, call, name, "<NEXT>")

    if(standbyCall == "" and skipIndex != -1):
      split_string = self.roster[skipIndex].split(" ")
      call   = split_string[0]
      name   = split_string[1]
      standbyCall = call
      self.updateRoster(skipIndex, call, name, '<NEXT>')
      self.view.updateCallNameStatus(None, call, name, "<NEXT>")
      		    
    self.view.updatePrevField(currentCall)
    self.view.updateNextField(standbyCall)
    return standbyCall
  
  def startTimer(self, call, setit):
    if(setit == True):
      self.timer_call = call			  
    self.debug.info_message("startTimer")
    self.timeout=-30
    return()
    
  def stopTimer(self, call, resetit, checkit):
    if(checkit == True):
      if(call== self.timer_call):
        self.timeout=self.max_timeout
        self.view.stopFlash(self.timeout)
        """ specific call based timeout satisfied so reset call in any case"""
        self.timer_call = ""
    else:
      self.timeout=self.max_timeout
      self.view.stopFlash(self.timeout)
      if(resetit == True):
        self.timer_call = ""
 		
    self.debug.info_message("stopTimer")
    return()

  """
  This method updates the previous and next fields on the main display.
  Accurate update of these fields is critical to the functioning of the state machine
  """
  def updatePrevNext(self, last_call):
    """
    update the previous and next fields. If the <NEXT station in line begins talkin that is the trigger for this
    """
    index = self.findRoster(last_call)
    if(index != -1):
      """ ignore the last heard station if not part of the net """
      if(self.roster[index].split(' ')[2] != "<HEARD>" and self.roster[index].split(' ')[2] != "<IGNORE>"):
        self.view.updatePrevField(self.roster[index].split(' ')[0])
        self.debug.info_message("updatePrevNext. Prev Field: " + self.roster[index].split(' ')[0])
        if(index < len(self.roster)-1):
          nextstn_status = self.roster[index+1].split(' ')[2]
          if( nextstn_status != "<HEARD>" and nextstn_status != "<DONE>" and nextstn_status != "<IGNORE>"):
            self.debug.info_message("updatePrevNext. Next Field: " + self.roster[index+1].split(' ')[0])
            self.view.updateNextField(self.roster[index+1].split(' ')[0])
          else:
            """ blank out next station if next in list is not in the net """
            self.debug.info_message("updatePrevNext. Next Field: ' '")
            self.view.updateNextField(' ')
        else:
          """ blank out next station if end of list reached """
          self.debug.info_message("updatePrevNext. Next Field: ' '")
          self.view.updateNextField(' ')

  def whoIsThis(self, dict_obj, text):
    self.debug.info_message("whoIsThis")
    """
    process incoming callsign
    """
    try:
      last_call = self.osmod_client.getNetParam(dict_obj, "CALL")
      if(last_call == None or last_call =="None"):
        if(": " in text):
          last_call = text.split(": ")[0]
          """ update for station talking """
          callindex = self.findRoster(last_call)
          if(callindex != -1):
            self.incrementStatus(-1, callindex, -1)
        else:
          """ need to figure out if this was the last known talking station by the offset value"""
          #try:
          intOffset = int(self.osmod_client.getNetParam(dict_obj, "OFFSET"))

          talkingIndex = self.getTalkingStationIndex()
          if(talkingIndex != -1):
            self.debug.info_message("whoIsThis. talking index: " + str(talkingIndex) )
            lastTalkingOffset = int(self.getStationOffset(talkingIndex))
            if(intOffset > lastTalkingOffset-3 and intOffset < lastTalkingOffset+3):
              last_call = self.getStationCallAlt(talkingIndex)
              self.debug.info_message("whoIsThis. Matched to station: " + last_call)

            else:
              matchedIndex = self.getOffsetMatch(intOffset, 3)
              lastTalkingOffset = int(self.getStationOffset(matchedIndex))
              if(intOffset > lastTalkingOffset-3 and intOffset < lastTalkingOffset+3):
                last_call = self.getStationCallAlt(matchedIndex)
          else:
            matchedIndex = self.getOffsetMatch(intOffset, 3)
            self.debug.info_message("whoIsThis. matched index: " + str(matchedIndex) )
            lastTalkingOffset = int(self.getStationOffset(matchedIndex))
            if(intOffset > lastTalkingOffset-3 and intOffset < lastTalkingOffset+3):
              last_call = self.getStationCallAlt(matchedIndex)
            else:
              last_call = "-"			        
          
    except:
      self.debug.error_message("method: whoIsThis. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))

    self.debug.info_message("whoIsThis returning: " + last_call)
	  
    return last_call

  """
  process the message returned
  """
  def processMsg(self, dict_obj, text):
    self.debug.info_message("processMsg")

    self.debug.info_message("dict_obj: " + str(dict_obj))
    self.debug.info_message("text: " + str(text))

    try:
      last_call = self.whoIsThis(dict_obj, text)

      """ process the received information """
      if(last_call != None and last_call != "None"):

        self.debug.info_message("processMsg. Call: " + last_call)

        last_offset        = self.osmod_client.getNetParam(dict_obj, "OFFSET")
        last_SNR           = self.osmod_client.getNetParam(dict_obj, "SNR")
      
        if(last_SNR != "" and last_SNR != None and last_SNR != "None"):
          last_snr_int       = int(last_SNR)
          last_SNR = '{:+03d}'.format(last_snr_int)
      
        temp_str           = self.osmod_client.getNetParam(dict_obj, "TDRIFT")

        if(temp_str != None and temp_str != "None"):
          last_TimeDelta     = str(int(float(temp_str)*1000)) # + "ms"
        else:
          last_TimeDelta     = "-"

        last_BadFrm        = str(self.osmod_client.areFramesMissing(self.osmod_client.getValue(dict_obj, "value") ))

        """ update the roster with received information """
        found = self.updateRosterData(last_call, last_offset, last_SNR, last_TimeDelta)

        """ process auto checkin """
        if(found == 0 and self.getAutoCheckin() == True):
          self.processAutoCheckin(text, last_call.strip(), last_offset, last_SNR, last_BadFrm, last_TimeDelta)

      """ only stop the timer if the call matches what was set to start timer """
      self.stopTimer(last_call, False, True)
      self.view.refreshRoster(self.getRoster())
      self.debug.info_message("processMsg. complete")
    
      return(last_call)

    except:
      self.debug.error_message("exception in processMsg: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """
  automatically add the new call sign to the roster if not there already
  """
  def processAutoCheckin(self, text, last_call, last_offset, last_SNR, last_BadFrm, last_TimeDelta):

    self.debug.info_message("processAutoCheckin. setting status for '" + last_call + "' to <HEARD>")
    last_status = " <HEARD> "
    """ check to make sure the call sign does not have embedded spaces. if so discard as invalid"""
    if(not (' ' in last_call) ):   
      last_name = self.nameFromSavedCalls(last_call)
      if(last_name != '' and last_name != '-'):
        self.setRoster( self.getRoster() + [last_call + ' ' + last_name + last_status + last_offset + ' ' + last_SNR + ' ' + last_BadFrm + ' ' + last_TimeDelta] )
      elif(last_call != '' and last_call != '-'):
        self.setRoster( self.getRoster() + [last_call + ' -' + last_status + last_offset + ' ' + last_SNR + ' ' + last_BadFrm + ' ' + last_TimeDelta] )
          
  """
  callback function 
  """
  def my_new_callback(self, json_string, txrcv, rigname, riginstance):

    try:
      self.my_new_callback2(json_string, txrcv)
    except:
      self.debug.error_message("method: my_new_callback. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
    return

    
  def my_new_callback2(self, json_string, txrcv):

    line = json_string.split('\n')
    length = len(line)

    for x in range(length-1):
      self.debug.info_message("my_new_callback. LOC 2 ")

      dict_obj = json.loads(line[x])
      text = self.osmod_client.stripEndOfMessage(self.osmod_client.getValue(dict_obj, "value")).decode('utf-8')
      
      type = self.osmod_client.getValue(dict_obj, "type").decode('utf-8')
      last_call = None

      self.debug.info_message("my_new_callback. LOC 3 ")
      
      """ test to see if there are any missing frames """
      self.osmod_client.areFramesMissing(self.osmod_client.getValue(dict_obj, "value") )

      self.debug.info_message("my_new_callback. LOC 4 ")

      if (type == "STATION.CALLSIGN"):
        self.debug.info_message("my_new_callback. STATION.CALLSIGN")
        self.station_call_sign = self.osmod_client.getValue(dict_obj, "value").decode('utf-8')

      elif (type == "RIG.FREQ"):
        dialfreq = int(self.osmod_client.getNetParam(dict_obj, "DIAL"))
        freqstr = str(float(dialfreq)/1000000.0)
        offsetstr = self.osmod_client.getNetParam(dict_obj, "OFFSET")
        if(self.getFreqFromOSMOD() == True):
          self.setManualFrequency(freqstr)
          			
        self.debug.info_message("my_new_callback. RIG.FREQ. Dial: " + freqstr)
        self.debug.info_message("my_new_callback. RIG.FREQ, Offset: " + offsetstr)

      elif (type == "RX.SPOT"):
        self.debug.info_message("my_new_callback. RX.SPOT")
        last_call = self.processMsg(dict_obj, text)

      elif (type == "RX.DIRECTED"):
        self.debug.info_message("my_new_callback. RX.DIRECTED")
        self.debug.info_message("dict_obj: " + str(dict_obj) )
        last_call = self.processMsg(dict_obj, text)
        try:  
          self.parser.decodeTriggers(dict_obj, text, last_call, txrcv)
        except:
          self.debug.error_message("method: my_new_callback. RX.DIRECTED. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
        
      elif (type == "RX.ACTIVITY"):
        self.debug.info_message("my_new_callback. RX.ACTIVITY")
        last_call = self.processMsg(dict_obj, text)
        missing_frames = self.osmod_client.areFramesMissing(text.encode() )
        self.debug.info_message("processMsg. RX.ACTIVITY. missing frames: " + str(missing_frames) )
        if missing_frames>0:
          self.updateRosterMissingFrames(last_call, missing_frames)
          self.view.writeMsgToScreen(dict_obj, text + "MISSING FRAMES: " + str(missing_frames), self.osmod_client, last_call)
        elif self.osmod_client.isEndOfMessage(text.encode()):
          self.view.writeMsgToScreen(dict_obj, text + "EOM", self.osmod_client, last_call)
        else:
          self.view.writeMsgToScreen(dict_obj, text, self.osmod_client, last_call)
        
      elif (type == "RIG.PTT"):
        self.debug.info_message("my_new_callback. RIG.PTT")
        pttstate = self.osmod_client.getNetParam(dict_obj, "PTT")
        if(str(pttstate) =="False" ):
          if(self.getOutgoingAwaitingResponse() == True):
            self.startTimer("", False)

          self.debug.info_message("my_new_callback. RIG.PTT Start Timer")
        if(str(pttstate) =="True" ):
          self.stopTimer("", False, False)
        self.debug.info_message("my_new_callback. RIG.PTT PTT State: " + str(pttstate))
      elif (type == "TX.TEXT"):
        self.debug.info_message("my_new_callback. TX.TEXT")
      elif (type == "TX.FRAME"):
        self.debug.info_message("my_new_callback. TX.FRAME")
      elif (type == "STATION.STATUS"):
        dialfreq = int(self.osmod_client.getNetParam(dict_obj, "DIAL"))
        freqstr = str(float(dialfreq)/1000000.0)
        offsetstr = self.osmod_client.getNetParam(dict_obj, "OFFSET")
        if(self.getFreqFromOSMOD() == True):
          self.setManualFrequency(freqstr)
          self.window['input_netfre'].update(freqstr)
          self.window['text_netfre'].update(text_color = 'black' )
          			
        self.debug.info_message("my_new_callback. STATION.STATUS. Dial: " + freqstr)
        self.debug.info_message("my_new_callback. STATION.STATUS, Offset: " + offsetstr)
      else:
        self.debug.warning_message("my_new_callback. unhandled type: " + str(type) )

  """
  send the message 
  """
  def sendIt(self, message, handoff):

    self.debug.info_message("sendIt. message: " + message)
    
    send_type = self.window['option_postsend'].get()
    send_btu = self.window['cb_autohandoff'].get()
    post_text = ''
    if (send_btu and handoff):
      post_text = self.processBtuOverTo()

    checked = self.window['cb_simulate'].get()
    if(checked == False):
      if(send_type == "Post to OSMOD only"):
        self.debug.info_message("sendIt. Invoking TX.SET_TEXT")
        self.osmod_client.sendMsg("TX.SET_TEXT", message + post_text)
      else:
        self.debug.info_message("sendIt. Invoking TX.SEND_MESSAGE")
        self.osmod_client.sendMsg("TX.SEND_MESSAGE", message + post_text)
    return ()

  def sendItNow(self, message, handoff):

    self.debug.info_message("sendItNow. message: " + message)

    """need to blank the message box first so that send engages correctly"""
    self.osmod_client.sendMsg("TX.SET_TEXT", '')
    send_btu = self.window['cb_autohandoff'].get()
    post_text = ''
    if (send_btu and handoff):
      post_text = self.processBtuOverTo()

    """ if participant stn then need to process my tx message so that fields (station status) on my station update  """
    checked = self.window['cb_simulate'].get()
    if(checked == False):
      self.buildAndProcessSimulatedMessage(message, post_text)
      self.debug.info_message("sendItNow. Invoking TX.SEND_MESSAGE")
      self.osmod_client.sendMsg("TX.SEND_MESSAGE", message + post_text )
    return ()

  def buildAndProcessSimulatedMessage(self, message, post_text):

    try:
      self.debug.info_message("method: buildAndProcessSimulatedMessage. message + post text: " + message + post_text)
      if(message != ""):
        group_name = self.window['input_netgroup'].get().upper().strip()
        if group_name in (message + post_text):
          text = (message + post_text).split(group_name)[1]
        else:
          group_name="@ALLCALL"
          text = (message + post_text).split(group_name)[1]

        #pre_text = self.getStationCallSign().decode('utf-8') + ': ' + self.window['input_netgroup'].get().strip()
        pre_text = self.getStationCallSign() + ': ' + self.window['input_netgroup'].get().upper().strip()
        simulatedstring = '{"params":{},"type":"RX.DIRECTED","value":"' + pre_text + text + '"}\n'
        self.debug.info_message("buildAndProcessSimulatedMessage. simulate string: " + simulatedstring)
        self.my_new_callback(self.parser.replaceFields(simulatedstring, True), cn.TX, '', None)
       
    except:
      self.debug.error_message("message + post text: " + message + " " + post_text)
      self.debug.error_message("method: buildAndProcessSimulatedMessage. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      
    return 

  def getRoster(self):
    return self.roster

  def setRoster(self, roster):
    self.roster = roster

  def appendRoster(self, call, name, status):

    self.debug.info_message("appendRoster. Appending: " + call)

    if(call != "" and name == "-"):
      name = self.nameFromSavedCalls(call)
    if(status == "<NONE>"):
      if(call == self.window['input_ncs'].get() ):
        status = "<NCS>"
      else:
        status = "<CHECKIN>"

    offset = self.window['main_offset'].get().strip()
    self.setRoster( self.getRoster() + [call + " " + name + " " + status + ' ' + offset + ' 0 0 0'] )
    self.view.refreshRoster(self.getRoster())
    return()

  def updateRoster(self, index, call, name, status):

    self.debug.info_message("updateRoster. status: " + status)

    split_string = self.roster[index].split(" ")
    offset   = split_string[3]
    snr      = split_string[4]
    badfr    = split_string[5]
    timedlt  = split_string[6]
	  
    self.roster[index] = call + " " + name + " " + status + " " + offset + " " + snr + " " + badfr + " " + timedlt
    self.view.refreshRoster(self.getRoster())
    return()


  def getRosterStatus(self, index):
    status = "<NONE>"	  
    if(index != -1):
      split_string = self.roster[index].split(" ")
      status = split_string[2]
    return status

  def getRosterName(self, index):
    name = ""
    if(index != -1):
      split_string = self.roster[index].split(" ")
      name = split_string[1]
    return name


  def updateRosterStatus(self, index, status):
    self.debug.info_message("updateRosterStatus. status: " + status)

    split_string = self.roster[index].split(" ")
    call     = split_string[0]
    name     = split_string[1]
    offset   = split_string[3]
    snr      = split_string[4]
    badfr    = split_string[5]
    timedlt  = split_string[6]

    self.debug.info_message("updateRosterStatus. Updating status for " + call + " to " + status)
	  
    self.roster[index] = call + " " + name + " " + status.strip() + " " + offset + " " + snr + " " + badfr + " " + timedlt
    self.view.refreshRoster(self.getRoster())
    return()

  def updateRosterData(self, last_call, last_offset, last_SNR, last_TimeDelta):
    self.debug.info_message("updateRosterData. Call = " + last_call)

    try:
      self.debug.info_message("self.roster: " + str(self.roster))
      for x in range(len(self.roster)):
        self.debug.info_message("updateRosterData. roster line : " + self.roster[x])
        roster_split_string = self.roster[x].split(" ")
        roster_call = roster_split_string[0]     
        self.debug.info_message("roster_split_string: " + str(roster_split_string))

        if(roster_call == last_call):
          self.debug.info_message("updateRosterData. Call match: " + last_call)

          roster_name      = roster_split_string[1]
          roster_status    = roster_split_string[2]
          roster_offset    = roster_split_string[3]
          roster_SNR       = roster_split_string[4]
          roster_BadFrm    = roster_split_string[5]
          roster_TimeDelta = roster_split_string[6]
        
          if(last_offset != "" and last_offset != None and last_offset != "None"):
            roster_offset = last_offset
          if(last_SNR != "" and last_SNR != None and last_SNR != "None"):
            roster_SNR = last_SNR
          if(last_TimeDelta != "" and last_TimeDelta != None and last_TimeDelta != "None" and last_TimeDelta != "-"):
            roster_TimeDelta = last_TimeDelta

          self.debug.info_message("updateRosterData. status: " + roster_status)

          self.roster[x] = roster_call + " " + roster_name + " " + roster_status + " " + roster_offset + " " + roster_SNR + " " + roster_BadFrm + " " + roster_TimeDelta
          return(1)
    except:
      self.debug.error_message("method: updateRosterData. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))
      
    return(0)

  def updateRosterMissingFrames(self, msg_call, missing_frames):
	  
    badfrint = 0
    for x in range(len(self.roster)):
      split_string = self.roster[x].split(" ")
      call     = split_string[0]
      name     = split_string[1]
      status   = split_string[2]
      offset   = split_string[3]
      snr      = split_string[4]
      badfr    = split_string[5]
      timedlt  = split_string[6]
      
      if call == msg_call:
        if(badfr != "-"):
          badfrint = int(badfr)
        badfrint = badfrint + missing_frames

        self.roster[x] = call + " " + name + " " + status + " " + offset + " " + snr + " " + str(badfrint) + " " + timedlt
        self.view.refreshRoster(self.getRoster())
        break
        
    return()


  def updateRosterOffsetsFromPlan(self, new_plan):
    """
    space out the stations based on max signal width and position in the roster
    net control is always at 1337
    """
    self.offsets_list = new_plan
    offset_split_string = self.offsets_list.split(',')
    max_offsets = len(offset_split_string)
    stn_offset = offset_split_string[0]
    offset_count=1

    for x in range(0, len(self.roster)):
      split_string = self.roster[x].split(" ")
      call     = split_string[0]
      name     = split_string[1]
      status   = split_string[2]
      snr      = split_string[4]
      badfr    = split_string[5]
      timedlt  = split_string[6]

      """ update the roster offset value """
      self.roster[x] = call + " " + name + " " + status + " " + stn_offset + " " + snr + " " + badfr + " " + timedlt
           
      """ is this my station if so set the main offset field """
      if(self.getStationCallSign() == call):
        self.window['main_offset'].update(stn_offset)
        self.window['main_offset'].update(disabled = True)
        self.window['cb_presetoffset'].update(value=False)

      if(offset_count<max_offsets):
        stn_offset=offset_split_string[offset_count]
        offset_count = offset_count + 1
      else:
        """ need to skip the first one second time around as that is for net control only """			   
        stn_offset=offset_split_string[1]
        offset_count=2

    self.view.refreshRoster(self.getRoster())

  """
  insert new item after the index line
  """
  def insertRoster(self, index, call, name, status):

    self.debug.info_message("insertRoster. status: " + status)
	  
    offset = self.window['main_offset'].get().strip()
    self.roster.insert(index+1, call + " " + name + " " + status + ' ' + offset + ' 0 0 0')
    self.view.refreshRoster(self.getRoster())
    return()

  def findRoster(self, call):
    found = -1

    for x in range(len(self.roster)):
      roster_call = self.roster[x].split(" ")[0]
      if(roster_call == call):
        found = x
        break

    return(found)
    
  def resetRoster(self):
    self.view.updatePrevField('')
    found = 0
    for x in range(1, len(self.roster)):
      roster_call   = self.roster[x].split(" ")[0]
      roster_status = self.roster[x].split(" ")[2]
      if(roster_status == "<TALKING>" or roster_status == "<NEXT>" or roster_status == "<SKIP>" or roster_status == "<DONE>" or roster_status == "<CHECKIN>"):
        self.updateRosterStatus(x, '<STANDBY>')		  
      roster_status = self.roster[x].split(" ")[2]
      if(found == 0 and roster_status == "<STANDBY>"):
        self.view.updateNextField(roster_call)
        found = x
    if(found == 0):
      self.view.updateNextField('')

    if(len(self.roster) >0 and self.window['input_ncs'].get() == self.roster[0].split(" ")[0]):
      self.updateRosterStatus(0, '<NCS>')		  

    return    

  def getTalkingStationIndex(self):
    for x in range(len(self.roster)):
      roster_status = self.roster[x].split(" ")[2]
      if(roster_status == "<TALKING>"):
        return x
    return -1
  
  def getOffsetMatch(self, intOffset, nearest):
    for x in range(len(self.roster)):
      roster_offset = int(self.roster[x].split(" ")[3])
      if(intOffset > roster_offset-nearest and intOffset < roster_offset+nearest):
        return x
    return -1

  def getStationOffset(self, index):
    roster_offset = self.roster[index].split(" ")[3]
    return roster_offset
    
  def getStationCallAlt(self, index):
    roster_call = self.roster[index].split(" ")[0]
    return roster_call

  def setDialAndOffset(self, dial, offset):
    self.osmod_client.sendMsg("RIG.SET_FREQ", "", params={"DIAL":dial, "OFFSET":offset, "_ID":-1})
    return()

  def getDialAndOffset(self):
    self.osmod_client.sendMsg("RIG.GET_FREQ", "")
    return()
    
  def getStationCall(self):
    self.osmod_client.sendMsg("STATION.GET_CALLSIGN", "")
    return()

def usage():
  sys.exit(2)


#if __name__ == '__main__':
#    main()
