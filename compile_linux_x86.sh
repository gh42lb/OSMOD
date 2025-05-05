

# compile.sh for Linux platform

gcc -c osmod_c_code.c -fPIC
gcc -shared -o lb28_compiled_linux_x86_64.so osmod_c_code.o

