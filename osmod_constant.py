
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

""" debug level constants """
DEBUG_VERBOSE = 0
DEBUG_INFO    = 1
DEBUG_WARNING = 2
DEBUG_ERROR   = 3
DEBUG_OFF     = 4

DEBUG_OSMOD           = DEBUG_INFO
DEBUG_OSMOD_MOD       = DEBUG_INFO
DEBUG_OSMOD_DEMOD     = DEBUG_INFO
DEBUG_OSMOD_MAIN      = DEBUG_INFO

OSMOD_MODEM_8PSK      = 1
OSMOD_MODEM_8FSK      = 2
OSMOD_MODEM_OLIVIA    = 3
OSMOD_MODEM_APSK      = 4
OSMOD_MODEM_QPSK      = 5
OSMOD_MODEM_16QAM     = 6

EXTRACT_NORMAL    = 0
EXTRACT_INTERPOLATE   = 1
