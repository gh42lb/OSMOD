#!/usr/bin/env python

import FreeSimpleGUI as sg

import sys
import threading
import json
import constant as cn

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
This class handles the events from the varius control defined in osmod_net_gui
"""
class ControlsProc(object):
  
  flashstate=0
  flashtimer=0
  
  def __init__(self, view, osmod_net, window):  
    self.view = view
    self.osmod_net = osmod_net
    self.window = window
    self.timer_counter1 = 0
    self.timer_counter2 = 0
    self.window_initialized = False

  def time_delta_string(self):
    time_b = datetime.utcnow()

    self.osmod_net.utc_time_now = time_b
    self.osmod_net.time_now = datetime.now()

    try:
      start = self.window['input_starttime'].get().strip()

      #self.osmod_net.debug.info_message("start: " + str(start) )

      starthour = int(start.split(':')[0])
      startmin  = int(start.split(':')[1])
      time_a = datetime(time_b.year, time_b.month, time_b.day, starthour, startmin, 0, 0) # net start in UTC
    except:
      """set as a default if time field not set correctly"""		
      time_a = datetime(time_b.year, time_b.month, time_b.day, 4, 30, 0, 0) # net start in UTC

    delta = time_a - time_b
    tdSaved = tdSeconds = delta.seconds
    tdMinutes, tdSeconds = divmod(abs(tdSeconds), 60)
    tdHours, tdMinutes = divmod(tdMinutes, 60)

    """
    if(tdHours==0 and tdSeconds == 0):
      if(tdMinutes == 30):
        self.view.startFlashButtons(['button_qst'])
      elif(tdMinutes == 10):
        self.view.startFlashButtons(['button_qst'])
      elif(tdMinutes == 2):
        self.view.stopFlashingButtons(['button_qst'])
        self.view.startFlashButtons(['button_open'])
    """
    
    if (tdSaved>=0 and tdHours<2):
      return "T-{hours:02d}:{minutes:02d}:{seconds:02d}".format(
                 hours=tdHours,
                 minutes=tdMinutes,
                 seconds=tdSeconds)
    elif tdHours>=23:
      return "ON AIR"		
    else:
      return "--:--:--"

  """
  add a station to the roster
  """
  def event_addstn(self, values):
    selected = self.view.getSelected(values)
    call = self.view.getCall(values)

    if( call != "-" and self.view.getName(values) != "-"):
      self.osmod_net.updateSavedCalls(self.view.getCall(values), self.view.getName(values) )

    if( call != "-"):
      if(selected == []):
        index = self.osmod_net.findRoster(call)
        if(index == -1):
          self.osmod_net.appendRoster(call, self.view.getName(values), self.view.getStatus(values))
        else:
          self.osmod_net.updateRoster(index, call, self.view.getName(values), self.view.getStatus(values))
          if(self.view.getStatus(values) == "<NEXT>"):
            self.view.updateNextField(call)
          if(self.view.getStatus(values) == "<TALKING>"):
            self.osmod_net.getNextStandbyStation(call, True, True)
            self.view.updatePrevField(call)
          
      else:
        index=selected[0]
        if(self.osmod_net.roster[index].split(' ')[0] == call):
          self.osmod_net.updateRoster(index, call, self.view.getName(values), self.view.getStatus(values))
          if(self.view.getStatus(values) == "<NEXT>"):
            self.view.updateNextField(call)
          if(self.view.getStatus(values) == "<TALKING>"):
            self.osmod_net.getNextStandbyStation(call, True, True)
            self.view.updatePrevField(call)

          """
          update the entry
          """
        else:
          self.osmod_net.insertRoster(index, call, self.view.getName(values), self.view.getStatus(values))
          """
          insert a new entry in the middle 
          """
    return()

  """
  remove a station from the roster
  """
  def event_removestn(self, values):
    self.osmod_net.roster.remove(self.osmod_net.roster[self.view.getSelectedIndex(values)])
    self.view.removestn(values, self.osmod_net.getRoster())
 
  def event_msgselectside(self, values):
    myrpt = self.view.handle_msgselectside(values)
    return()

  def event_button_QST(self, values):
    self.osmod_net.debug.info_message("event_button_QST")
    try:
      self.osmod_net.setOutgoingAwaitingResponse(False)
      self.view.stopFlashingAllButtons()
      self.view.next_flash_buttons = ['button_ckin', 'button_qst']
      self.view.startFlashButtons(['send_really'])
      myrpt = self.view.handle_button(values, 'QST', 'button_qst')
      return()
    except:
      sys.stdout.write("Exception in event_button_QST: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")
 

  def event_button_Open(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_ckin']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Open', 'button_open')
    return()

  def event_button_CKin(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_roster', 'button_ckin', 'button_gotu', 'button_open']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'CKin', 'button_ckin')
    return()
    
  def event_button_gotu(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_roster', 'button_ckin', 'button_gotu']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Gotu', 'button_gotu')
    return()

  def event_button_ncscust1(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    myrpt = self.view.handle_button(values, 'NCS_Cust1', 'button_ncs_cust1')
    return()
    
  def event_button_ncscust2(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    myrpt = self.view.handle_button(values, 'NCS_Cust2', 'button_ncs_cust2')
    return()

  def event_button_Roster(self, values):
    self.osmod_net.resetRoster()
    self.view.updateCallNameStatus(None, '', '', '<NONE>')
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_first', 'send_announce']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Roster', 'button_roster')
    return()

  def event_button_First(self, values):

    self.osmod_net.resetRoster()

    """ set the current round to the one we are now starting out on"""
    """ update the display combo to reflect the next round but do not update current round until starting out on the next round"""
    if( self.osmod_net.getCurrentRound() == self.window['option_currentround'].get() ):
      self.osmod_net.incRound()

    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_next', 'button_prompt']
    self.view.startFlashButtons(['send_really'])

    call = self.window['next_stn'].get().strip()
    self.osmod_net.setAsNext(call)
    myrpt = self.view.handle_button(values, 'First', 'button_first')
    return()

  def event_button_Next(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()

    self.view.next_flash_buttons = ['button_next', 'button_prompt', 'button_end']
    self.view.startFlashButtons(['send_really'])

    call = self.window['next_stn'].get().strip()
    checked = self.window['cb_further'].get()
    if not checked:
      self.osmod_net.setAsNext(call)
    myrpt = self.view.handle_button(values, 'Next', 'button_next')
    return()

  """ The action of this button relates to the end of the round """
  def event_button_End(self, values):
    self.osmod_net.setCurrentRound( self.window['option_currentround'].get() )
    
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_cmnt', 'send_report', 'button_ckin']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'End', 'button_end')
    return()

  def event_button_skip(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_next', 'button_prompt']
    self.view.startFlashButtons(['send_really'])
    call = self.window['next_stn'].get().strip()
    self.osmod_net.getNextStandbyStation(call, True, True)
    myrpt = self.view.handle_button(values, 'Skip', 'button_skip')
    return()

  def event_button_prompt(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_skip', 'button_next']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Prompt', 'button_prompt')
    return()
  
  def event_button_Cmnt(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(True)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = ['button_close', 'button_cmnt']
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Cmnt', 'button_cmnt')
    return()

  def event_button_Close(self, values):
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button(values, 'Close', 'button_close')
    return()

  """ The following methods are for the client/participant programmable buttons"""
  def event_btncli_clear(self, values):
    self.window['input_rounds'].Update(value='')
    self.window['input_ncs'].Update(value='')
    self.window['input_netgroup'].Update(value=self.osmod_net.getManualGroup())
    self.window['input_netfre'].Update(value=self.osmod_net.getManualFrequency())
    self.window['input_netname'].Update(value='')
    self.window['input_starttime'].Update(value='')
    self.window['option_profile'].Update(value='')
    self.window['option_menu'].Update(value='-')
    self.window['cb_savedetails'].Update(value=False)

    self.view.updatePrevField('')
    self.view.updateNextField('')
    self.view.refreshRoster([])

    self.view.checkDisableAllButtons()
    return()
    
  def event_cbpresetoffset(self, values):
    checked = self.window['cb_presetoffset'].get()
    if(checked):
      self.window['main_offset'].update(disabled=False)		
    else:
      self.window['main_offset'].update(disabled=True)		
    return

  def event_cbprevstn(self, values):
    checked = self.window['cb_prevstn'].get()
    if(checked):
      self.window['prev_stn'].update(disabled=False)		
    else:
      self.window['prev_stn'].update(disabled=True)		
    return

  def event_cbnextstn(self, values):
    checked = self.window['cb_nextstn'].get()
    if(checked):
      self.window['next_stn'].update(disabled=False)		
    else:
      self.window['next_stn'].update(disabled=True)		
    return
    
  def event_btncli_ckmein(self, values):
    self.window['multi_select'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_side'])
    myrpt = self.view.handle_button_common(values, 'CkMeIn', 'btncli_ckmein', cn.PREVIEW_SIDE, False)
    return()
    
  def event_btncli_alert(self, values):
    self.window['multi_select'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_side'])
    myrpt = self.view.handle_button_common(values, 'Alert', 'btncli_alert', cn.PREVIEW_SIDE, False)
    return()
    
  def event_btncli_qrt(self, values):
    self.window['multi_select'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_side'])
    myrpt = self.view.handle_button_common(values, 'QRT', 'btncli_qrt', cn.PREVIEW_SIDE, False)
    return()
    
  def event_btncli_post(self, values):
    self.window['multi_select'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_side'])
    myrpt = self.view.handle_button_common(values, 'POST', 'btncli_post', cn.PREVIEW_SIDE, False)
    return()
    
  def event_btncli_tks(self, values):
    self.window['multi_side'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button_common(values, 'TKS', 'btncli_tks', cn.PREVIEW_MAIN, True)
    return()
   
  def event_btncli_cust1(self, values):
    self.window['multi_select'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_side'])
    myrpt = self.view.handle_button_common(values, 'CliCust1', 'btncli_cust1', cn.PREVIEW_SIDE, False)
    return()
    
  def event_btncli_cust2(self, values):
    self.window['multi_side'].update("")
    self.osmod_net.setOutgoingAwaitingResponse(False)
    self.view.stopFlashingAllButtons()
    self.view.next_flash_buttons = []
    self.view.startFlashButtons(['send_really'])
    myrpt = self.view.handle_button_common(values, 'CliCust2', 'btncli_cust2', cn.PREVIEW_MAIN, True)
    return()

  def event_option_postsend(self, values):
    send_type = self.window['option_postsend'].get()
    if(send_type == "Post to OSMOD only"):
      self.window['send_really'].Update(disabled = False)	
    else:
      self.window['send_really'].Update(disabled = True)	  

  def setDialsMain(self):
    offset = int(self.window['main_offset'].get().split("Hz")[0])
    freq_text = self.window['input_netfre'].get().strip()
    
    if(freq_text != ""):
      freq = float(freq_text.split("MHz")[0])
      int_freq = int(freq * 1000000)
      self.osmod_net.setDialAndOffset(int_freq, offset)
    else:
      self.osmod_net.debug.error_message("setDialsMain. Frequency field not set. Please set frequency.")

  def setDialsSide(self):
    offset = int(self.window['sidebar_offset'].get().split("Hz")[0])
    freq_text = self.window['input_netfre'].get().strip()

    if(freq_text != ""):
      freq = float(freq_text.split("MHz")[0])
      int_freq = int(freq * 1000000)
      self.osmod_net.setDialAndOffset(int_freq, offset)
    else:
      self.osmod_net.debug.error_message("setDialsSide. Frequency field not set. Please set frequency.")
  
  def event_sendside(self, values):
    self.osmod_net.debug.info_message("event_sendside")
    try:
      checked = self.window['cb_simulate'].get()
      if(checked == False):
        if(self.window['input_netgroup'].get().strip() == ""):
          self.osmod_net.debug.error_message("group field is empty. please specify a group name")
        else:		
          text = self.window['multi_side'].get().strip()
        
          self.osmod_net.setSeqnum(0)
          self.view.stopFlashingAllButtons()
          self.view.startFlashButtons(self.view.next_flash_buttons)
          self.view.next_flash_buttons = []
        
          self.window['multi_side'].update("")
          self.setDialsSide()    
          self.uncheckFields()
          self.osmod_net.sendItNow(text, False)
      else:
        """
        simulate a call
        """
        text = self.window['multi_side'].get().strip()
        simulatedstring = '{"params":{"DIAL":7078000,"FREQ":7080341,"OFFSET":890,"SNR":-5,"SPEED":4,"TDRIFT":-0.5,"UTC":1654371086715,"_ID":-1},"type":"RX.DIRECTED","value":"' + text + '"}\n'
        self.osmod_net.debug.info_message("event_sendside. simulated string: " + simulatedstring)
        self.osmod_net.my_new_callback(self.osmod_net.parser.replaceFields(simulatedstring, True), cn.RCV, "NOT USED", "NOT USED")
      
      return()
    except:
      sys.stdout.write("Exception in event_sendside: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")


  def event_send_really(self, values):
    self.osmod_net.debug.info_message("event_send_really")

    self.osmod_net.debug.info_message("window: " + str(self.window))
    self.osmod_net.debug.info_message("view: " + str(self.view))

    try:
      if(self.window['input_netgroup'].get().strip() == ""):
        self.osmod_net.debug.error_message("group field is empty. please specify a group name")
      else:		
        if(self.osmod_net.getRosterStandbyCalls() != ""):
          split_string = self.osmod_net.getRosterStandbyCalls().split(',')
          self.osmod_net.debug.info_message("event_send_really. calls list: " + self.osmod_net.getRosterStandbyCalls())
          for x in range(len(split_string)):
            callindex = self.osmod_net.findRoster(split_string[x])
            self.osmod_net.updateRosterStatus(callindex, '<STANDBY>')
          self.osmod_net.setRosterStandbyCalls("")

        text = self.window['multi_select'].get().strip()
      
        self.osmod_net.setSeqnum(0)
        self.view.stopFlashingAllButtons()
        self.view.startFlashButtons(self.view.next_flash_buttons)
        self.view.next_flash_buttons = []
      
        self.window['multi_select'].update("")
        self.setDialsMain()    
        self.uncheckFields()
        self.osmod_net.sendItNow(text, True)
      return()
    except:
      sys.stdout.write("Exception in event_send_really: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def uncheckFields(self):
    send_type = self.window['option_postsend'].get()
    if(send_type == "Post to OSMOD only"):

      checked = self.window['cb_clronsnd'].get()
      if(checked):
        self.window['cb_rep'].Update(False)
        self.window['cb_snr'].Update(False)
        self.window['cb_further'].Update(False)
        self.window['cb_other'].Update(False)
        self.window['cb_goodeve'].Update(False)
        self.window['cb_welcome'].Update(False)
        self.window['multi_other'].Update('')
      if(self.osmod_net.getOutgoingAwaitingResponse() == True):
        self.osmod_net.startTimer("", False)
    
  def event_cb_autocheckin(self, values):
    value = self.window['cb_autocheckin'].get()
    if(value == True):
      self.osmod_net.setAutoCheckin(True)
    else:
      self.osmod_net.setAutoCheckin(False)

  def event_cb_further(self, values):
    value = self.window['cb_further'].get()
    if(value == True):
      self.window['cb_goodeve'].Update(False)

  def event_flashbtn(self, values):
    value = self.window['cb_flashbtn'].get()
    if(value == True):
      self.osmod_net.setFlashingState(True)
    else:
      self.osmod_net.setFlashingState(False)
      saved = self.osmod_net.flash_win
      self.view.stopFlashingButtons(self.osmod_net.flash_win)
      self.osmod_net.flash_win = saved
      self.window['send_really'].Update(button_color=(self.view.btn_clr1, self.view.btn_clr2))

  def event_cb_goodeve(self, values):
    value = self.window['cb_goodeve'].get()
    if(value == True):
      self.window['cb_further'].Update(False)

  def sendIt(self, text):
    self.osmod_net.saved_send_string = self.osmod_net.parser.replaceFields(text, True)
    self.setDialsSide()    
    self.osmod_net.sendIt(self.osmod_net.saved_send_string, True)

  def event_reset_roster(self, values):
    self.osmod_net.resetRoster()
    self.view.stopFlashingAllButtons()
    self.view.updateCallNameStatus(None, '', '', '<NONE>')
    
    return()

  def event_cbedited(self, values):
    key = self.view.currently_editing
    if(key != None):
     if(self.window['cb_edited'].get() ):
       text = self.view.getEditedMessage(key)
       self.window['multi_select'].update(text)
     else:
       text = self.view.getDefaultMessage(key)
       self.window['multi_select'].update(text)

  def event_multiselect(self, values):
    self.window['cb_edited'].update(True)

  def event_roster(self, values):
    self.view.updateCallNameStatus(values, self.osmod_net.roster[self.view.getSelectedIndex(values)].split()[0],
                                           self.osmod_net.roster[self.view.getSelectedIndex(values)].split()[1],
                                           self.osmod_net.roster[self.view.getSelectedIndex(values)].split()[2])
   
    call = self.osmod_net.roster[self.view.getSelectedIndex(values)].split()[0]
    index = self.osmod_net.findRoster(call)
    if(index != -1):
      checked = self.window['cb_nextstn'].get()
      """use the roster select to update only the next field"""
      if(checked):
        self.view.updateNextField(self.osmod_net.roster[index].split(' ')[0])
        self.window['cb_nextstn'].update(False)
        self.window['next_stn'].update(disabled=True)		

      checked = self.window['cb_prevstn'].get()
      """use the roster select to update only the prev field"""
      if(checked):
        self.view.updatePrevField(self.osmod_net.roster[index].split(' ')[0])
        self.window['cb_prevstn'].update(False)
        self.window['prev_stn'].update(disabled=True)		
        
    return()

  def event_rostername(self, values):
    return()


  def event_save_net(self, values):
    sys.stdout.write("event_save") 
    try:
      self.view.writeDictToFile("osmod_net_save_data.txt")
      #self.osmod_client.stopThreads()
    except:
      self.osmod_net.debug.error_message("method: event_save. " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ))


  """ callsign is the key field. If this is changed then reset the name and status """
  def event_rstrchoose(self, values):
    self.window['rstr_name'].update(' ')  
    self.window['rstr_status'].update(cn.NONE)  
    
  def event_sendannounce(self, values):

    if(self.window['input_netgroup'].get().strip() == ""):
      self.osmod_net.debug.error_message("group field is empty. please specify a group name")
    else:		
      self.view.stopFlashingButtons(['send_announce'])
      self.view.startFlashButtons(['button_first'])
	  
      myrpt = self.window['input_netgroup'].get().strip() + ' ' + self.window['multi_announce'].get().strip() # + "BTN"
    
      self.setDialsMain()    
      self.osmod_net.sendItNow(myrpt, True)

    return()

  def event_sendother(self, values):

    if(self.window['input_netgroup'].get().strip() == ""):
      self.osmod_net.debug.error_message("group field is empty. please specify a group name")
    else:		
      self.view.stopFlashingButtons(['send_report'])
      self.view.startFlashButtons(['button_cmnt'])
	  
      myother = self.window['input_netgroup'].get().strip() + ' ' + self.window['multi_myreport'].get().strip()

      self.setDialsMain()    
      self.osmod_net.sendItNow(myother, True)

    return()

  def event_sendcomment(self, values):
    if(self.window['input_netgroup'].get().strip() == ""):
      self.osmod_net.debug.error_message("group field is empty. please specify a group name")
    else:		
      mycmnt = self.window['input_netgroup'].get().strip() + ' ' + self.window['multi_other'].get().strip()

      self.setDialsMain()    
      self.osmod_net.sendItNow(mycmnt, True)
    
    return()
    
  def event_enablesave(self, values):
    return()

  def event_netgrp(self, values):
    self.view.checkDisableAllButtons()
    return()

  def event_netfre(self, values):
    self.osmod_net.debug.info_message("event_netfre")
    try:
      self.view.checkDisableAllButtons()
      return()
    except:
      sys.stdout.write("Exception in event_netfre: " + str(sys.exc_info()[0]) + str(sys.exc_info()[1] ) + "\n")

  def event_enableother(self, values):
    self.window['cb_other'].Update(True)
    return()

  def event_rstrstatus(self, values):
    self.event_addstn(values)
    return

  def event_option_stntype(self, values):
    type = self.window['option_stntype'].get()
    if type == "NCS":
      self.view.clearall(values)
    else:
      self.view.populateall(values)

  def event_clearall(self, values):
    self.view.clearall(values)
    self.osmod_net.resetRoster()

    """ Make sure NCS always sits at the top of the roster """
    self.osmod_net.appendRoster(self.window['input_ncs'].get(), "-", "<NCS>")

    return
    
  def event_editbtn(self, values):
    checked = self.window['cb_editbtn'].get()
    if(checked):
      self.osmod_net.setEditMode(True)
      self.view.startFlashAllButtons()
      
    else:
      self.view.stopFlashingAllButtons()
      self.osmod_net.setEditMode(False)
    return
    
  def event_catchall(self, window, values):
    #self.window = window
    if(self.osmod_net.timeout<self.osmod_net.max_timeout):
      self.osmod_net.timeout+=1
    if(self.osmod_net.timeout>=0):
      window['slider_timeout'].Update(self.osmod_net.timeout)


    #self.osmod_net.debug.info_message("LOC 1")


    if(self.window_initialized == False):
      if(window['input_netfre'].get() == ""):
        frestr = self.osmod_net.getManualFrequency()
        if(frestr != ""):
          window['input_netfre'].update(frestr)
          window['text_netfre'].update(text_color = 'black' )

      if(window['input_netgroup'].get() == ""):
        grpstr = self.osmod_net.getManualGroup()
        if(grpstr != ""):
          window['input_netgroup'].update(grpstr)
          window['text_netgrp'].update(text_color = 'black' )

      self.window_initialized = True		
      
    """ This code flashes the received text windows during the expiration of the slider timer """
    """
    if(self.osmod_net.timeout == 200):
      self.flashtimer=self.osmod_net.timeout
    if(self.osmod_net.timeout>200 and self.osmod_net.timeout < 300):
      if(self.osmod_net.timeout - self.flashtimer > 5):
        self.flashtimer = self.osmod_net.timeout
        if(self.flashstate ==0):
          self.flashstate=1
          self.view.flashcolor(values, 'red', 'red')
        else:
          self.flashstate=0
          self.view.flashcolor(values, 'SeaGreen1', 'LightBlue1')
    """      

    if(self.view.delayed_send >0):
      self.view.delayed_send  = self.view.delayed_send -1
      if(self.view.delayed_send == 0):
        """transition state achieved """	
        
        """ are there any callsigns from the gotu processing waiting to be set to standby """
        if(self.osmod_net.getRosterStandbyCalls() != ""):
          split_string = self.osmod_net.getRosterStandbyCalls().split(',')
          self.osmod_net.debug.info_message("event_send_really. calls list: " + self.osmod_net.getRosterStandbyCalls())
          for x in range(len(split_string)):
            callindex = self.osmod_net.findRoster(split_string[x])
            self.osmod_net.updateRosterStatus(callindex, '<STANDBY>')
          self.osmod_net.setRosterStandbyCalls("")
        
        """ get the transmit string that was saved earlier """	  
        text = self.osmod_net.saved_send_string
        self.osmod_net.setSeqnum(0)
        self.view.stopFlashingAllButtons()
        self.view.startFlashButtons(self.view.next_flash_buttons)
        self.view.next_flash_buttons = []
        
        text = window['multi_select'].get().strip()
        if(text != ""):
          self.osmod_net.sendItNow(text, True)
          window['multi_select'].update("")
        else:
          text = window['multi_side'].get().strip()
          if(text != ""):
            self.osmod_net.sendItNow(text, False)
            window['multi_side'].update("")
       
        checked = window['cb_clronsnd'].get()
        if(checked):
          window['cb_rep'].Update(False)
          window['cb_snr'].Update(False)
          window['cb_further'].Update(False)
          window['cb_other'].Update(False)
          window['cb_goodeve'].Update(False)
          window['cb_welcome'].Update(False)
          window['multi_other'].Update('')
        if(self.osmod_net.getOutgoingAwaitingResponse() == True):
          self.osmod_net.startTimer("", False)
    


    

    """ flash the main control buttons """
    if(self.osmod_net.flash_win != None and self.osmod_net.getFlashingState()==True):
      if(self.timer_counter1 > 4):
        self.view.flashButtons()

    if(self.timer_counter1 > 4):
      self.timer_counter1 = 0
    else:
      self.timer_counter1 += 1
  
    """
    current_time = self.time_delta_string() 

    if(current_time == "ON AIR"):
      window['clock'].update(current_time, text_color='red')
      self.osmod_net.net_started = True
    else:  
      #self.osmod_net.debug.info_message("updating time")
      #self.osmod_net.debug.info_message("current_time: " + str(current_time))
      window['clock'].update(current_time, text_color='white')
      self.osmod_net.net_started = False
    """

    return()





  def event_exit(self, values):
    self.view.event_exit(values)

  dispatch = {
      'add_stn'          : event_addstn, 
      'remove_stn'       : event_removestn,
      'option_selectside': event_msgselectside,
      'send_side'        : event_sendside,
      'roster'           : event_roster,
      'rstr_name'        : event_rostername,
      'send_announce'    : event_sendannounce,
      'send_report'      : event_sendother,
      'send_comment'     : event_sendcomment,
      'Exit'             : event_exit,
      'multi_announce'   : event_enablesave,
      'multi_other'      : event_enableother,
      'multi_myreport'   : event_enablesave,
      'input_netgroup'   : event_netgrp,
      'input_netfre'     : event_netfre,
      'rstr_status'      : event_rstrstatus,
      'clear_all'        : event_clearall,
      'button_qst'       : event_button_QST,
      'button_open'      : event_button_Open,
      'button_ckin'      : event_button_CKin,
      'button_roster'    : event_button_Roster,
      'button_first'     : event_button_First,
      'button_next'      : event_button_Next,
      'button_end'       : event_button_End,
      'button_cmnt'      : event_button_Cmnt,
      'button_close'     : event_button_Close,
      'send_really'      : event_send_really,
      'option_postsend'  : event_option_postsend,
      'cb_autocheckin'   : event_cb_autocheckin,
      'cb_further'       : event_cb_further,
      'cb_goodeve'       : event_cb_goodeve,
      'option_stntype'   : event_option_stntype,
      'button_skip'      : event_button_skip,
      'button_prompt'    : event_button_prompt,
      'cb_flashbtn'      : event_flashbtn,
      'cb_editbtn'       : event_editbtn,       
      'button_gotu'      : event_button_gotu,
      'button_ncs_cust1' : event_button_ncscust1,
      'button_ncs_cust2' : event_button_ncscust2,
      'reset_roster'     : event_reset_roster,
      'cb_edited'        : event_cbedited,
      'multi_select'     : event_multiselect,
      'btncli_clear'     : event_btncli_clear,
      'btncli_ckmein'    : event_btncli_ckmein,
      'btncli_alert'     : event_btncli_alert,
      'btncli_qrt'       : event_btncli_qrt,
      'btncli_post'      : event_btncli_post,
      'btncli_tks'       : event_btncli_tks,
      'btncli_cust1'     : event_btncli_cust1,
      'btncli_cust2'     : event_btncli_cust2,
      'cb_presetoffset'  : event_cbpresetoffset,
      'cb_prevstn'       : event_cbprevstn,
      'cb_nextstn'       : event_cbnextstn,
      'roster_choose'    : event_rstrchoose,
      'btn_save_net'     : event_save_net,
  }

