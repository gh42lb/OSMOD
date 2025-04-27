
#modify .so name for your platform. name chosen must match name that python detects for platform (see debug output for reference) 
gcc -c osmod_c_code.c
gcc -shared -o lb28_compiled_linux_raspberrypi.so osmod_c_code.o

