

# compile.sh for Mac platform

gcc -c osmod_c_code.c -fPIC
gcc -shared -o lb28_compiled_darwin_arm64.so osmod_c_code.o

