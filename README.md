# OSMOD v0.2.0 alpha

PSK/FSK modulation and phase extraction using Quantized Time-Scale Holograms

"During development and testing of the LB28 I3 modes that utilize pulse train standing waves, specifically the testing of fixed absolute phase recovery with extrapolation, it has become apparent that not only are these effective techniques for phase recovery, but that the signal produced by the interposed three phase signal generator is in fact a quantized time-scale hologram with some very useful properties. These include:

* increased resilience to phase noise
* characters in the message having the appearance of being locked together or entangled thus eliminating phase drift
* ability to take multiple samples from the hologram to further enhance noise resilience by averaging out the noise. 
"

v0.2.0 includes a live modem for sending and receiving audio via speaker and mic (entangled phonons) and ham radio (entangled photons)


Instructions for running program...

either:-

1) use pre-built .so shared object for apple mac or
2) edit compile.sh or compile_linux_x86.sh and update name of .so for your OS and CPU architecture then run it to generate the .so file

run osmod:- python3 ./osmod_gui.py

view the console debug info to check compiled code loads correctly

testing confirms successful run on Apple Mac Book. 


![osmod4](https://github.com/user-attachments/assets/b5d5b5c2-c3f9-48d6-b11d-a5ee97836be7)


![osmod2](https://github.com/user-attachments/assets/acd51036-0c72-4404-84ea-90960f4e2fbd)


![osmod3](https://github.com/user-attachments/assets/51409883-1495-4ef4-9e19-9d5c1ef4c591)


