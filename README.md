# OSMOD v0.3.1 alpha

PSK/FSK modulation and phase extraction using Quantized Time-Scale Holograms

"During development and testing of the LB28 I3 modes that utilize pulse train standing waves, specifically the testing of fixed absolute phase recovery with extrapolation, it has become apparent that not only are these effective techniques for phase recovery, but that the signal produced by the interposed three phase signal generator is in fact a quantized time-scale hologram with some very useful properties. These include:

* increased resilience to phase noise
* characters in the message having the appearance of being locked together or entangled thus eliminating phase drift
* ability to take multiple samples from the hologram to further enhance noise resilience by averaging out the noise. 
"

v0.3.1 includes a live modem ** for sending and receiving audio via speaker and mic * (entangled phonons) and ham radio (entangled photons)

please note...audio testing is more effective at minimal magnitues i.e. barely audible.
** latency in the audio processing callbacks (instream and ostream) can significantly impact signal and decodes. modem tested successfully on Apple M4 laptop using lowest audio speaker setting; input magnitudes in the range 2 to 4

Instructions for running program...

either:-

1) use pre-built .so shared object for apple mac or
2) edit compile.sh or compile_linux_x86.sh and update name of .so for your OS and CPU architecture then run it to generate the .so file

run osmod:- python3 ./osmod_gui.py

view the console debug info to check compiled code loads correctly

testing confirms successful run on Apple Mac Book. 

<img width="2536" height="2122" alt="image" src="https://github.com/user-attachments/assets/f32e4e6e-c852-4af7-b417-1170dc5a0856" />

<img width="2012" height="1316" alt="performance_data" src="https://github.com/user-attachments/assets/e45081fb-5ce0-48d3-a0fb-f7927d4b61f3" />



