

# compile.sh for Mac platform

#gcc -c osmod_c_code.c -fPIC
#gcc -shared -o lb28_compiled_darwin_arm64.so osmod_c_code.o



gcc  -Wall -fPIC -Wno-deprecated -Wno-deprecated-declarations -c osmod_c_code.c 

#gcc  -Wall -fPIC -Wno-deprecated -Wno-deprecated-declarations -c lbmod_fft.c 

#gcc  -Wall -fPIC -Wno-deprecated -Wno-deprecated-declarations -c lbmod_I3.c 

gcc  -shared -o lb28_compiled_darwin_arm64.so osmod_c_code.o 

install_name_tool -add_rpath /usr/local/lib  lb28_compiled_darwin_arm64.so


#install_name_tool -add_rpath @executable_path lb28_compiled_darwin_arm64.so
# install_name_tool -rpath   @executable_path  /usr/local/lib   lb28_compiled_darwin_arm64.so


