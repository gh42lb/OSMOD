#!/usr/bin/env python

import os
import sys
import ctypes
import numpy as np

""" convert int array to ctypes pointer """
def ptoc_int_array(numpy_array):
    ctypes_array = (ctypes.c_int * len(numpy_array))(*numpy_array)
    #pointer = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    #array_size = numpy_array.size * numpy_array.itemsize
    #ctypes_array = (ctypes.c_int * numpy_array.size).from_buffer(numpy_array)
    return ctypes_array

def ptoc_numpy_int_array(numpy_array):
    array_size = numpy_array.size * numpy_array.itemsize
    ctypes_array = (ctypes.c_int  * numpy_array.size).from_buffer(numpy_array)
    return ctypes_array


""" convert numpy float32 array to ctypes pointer """
def ptoc_float_array(numpy_array):
    pointer = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    array_size = numpy_array.size * numpy_array.itemsize
    ctypes_array = (ctypes.c_float * numpy_array.size).from_buffer(numpy_array)
    return ctypes_array

""" convert numpy double array to ctypes pointer """
def ptoc_double_array(numpy_array):
    pointer = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    array_size = numpy_array.size * numpy_array.itemsize
    ctypes_array = (ctypes.c_double * numpy_array.size).from_buffer(numpy_array)
    return ctypes_array

def ptoc_float_list(float_list):
    return (ctypes.c_float * len(float_list))(*float_list)

""" convert float to ctypes float """
def ptoc_float(float_var):
    return ctypes.c_float(float_var)

""" convert double to ctypes double """
def ptoc_double(double_var):
    return ctypes.c_double(double_var)

""" convert ctypes int to int """
def ctop_int(int_var):
    return int_var.value


""" convert numpy complex128 array to ctypes pointer """
#def ptoc_complex_array(complex_array):
#    ctypes_array = (ctypes.c_double * complex_array.size * 2).from_buffer(complex_array)
#    return ctypes_array



def float_data_function(lib):

    numpy_array = np.array([1.0,2.0,3.0,4.0,5.0], dtype=np.float32)
    pointer = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    array_size = numpy_array.size * numpy_array.itemsize
    ctypes_array = (ctypes.c_float * numpy_array.size).from_buffer(numpy_array)

    sys.stdout.write("numpy array before call: " + str(numpy_array) + "\n")

    lib.my_func_with_float_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.my_func_with_float_data(ctypes_array, len(ctypes_array))

    sys.stdout.write("numpy array after call: " + str(numpy_array) + "\n")


def complex_data_function(lib):

    numpy_array = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

    sys.stdout.write("numpy array before call: " + str(numpy_array) + "\n")

    lib.my_func_with_complex_data.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags='C_CONTIGUOUS'), ctypes.c_int]
    lib.my_func_with_complex_data(numpy_array, numpy_array.size)

    sys.stdout.write("numpy array after call: " + str(numpy_array) + "\n")


def main():

    sys.stdout.write("Hello from Python\n")

    #os.chdir('/home/lawrence/mypython/liquiddsp/')
    #sys.stdout.write("current directory: " + str(os.getcwd()) + "\n")

    #lib = ctypes.CDLL("/home/lawrence/mypython/liquiddsp/mytest.dll")
    lib = ctypes.CDLL("/home/pi/mypython/osmod/src/c_code/mytest.dll")




    lib.myfunc()

    """ convert from c types double array to numpy """
    #np.ctypeslib.as_array(x)

    """ not recommended...convert to numpy from ctypes pointer to data"""
    #np.frombuffer(np.core.multiarray.int_asbuffer(ctypes.addressof(y.contents), array_length * np.dtype(float).itemsize))

    #buffer = np.core.multiarray.int_asbuffer(ctypes.addressof(y.contents), 8 * array_length)

    #buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
    #buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
    #buffer_from_memory.restore = ctypes.py_object
    #buffer = buffer_from_memory(y, 8*array_length)
    #my_numpy_array = np.grombuffer(buffer, float)

    """ these two methods should be the best to convert c type to a numpy array"""
    #c_array = (c_float * 8)()
    #np.ctypeslib.as_array(c_array)

    #c_array = (c_float * 8)()
    #ptr = ctypes.pointer(c_array[0])
    #np.ctypeslib.as_array(ptr, shape=(8,))

    """ convert numpy array to c types"""

    float_data_function(lib)

    complex_data_function(lib)

if __name__ == '__main__':
    main()




