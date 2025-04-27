

#gcc -Wall -o mytest mytest.c -lm -lc -lliquid
#gcc -shared -o mytest.dll mytest.o



gcc -c osmod_c_code.c
gcc -shared -o lb28_compiled_linux_raspberrypi.so osmod_c_code.o

