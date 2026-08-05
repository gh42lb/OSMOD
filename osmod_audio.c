#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include "portaudio.h"

int count_max_occurrences(int* data, int size, int* items_to_ignore, int size2);


/*
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
*/


static int audioCallbackOutput(const void *inputBuffer, void *outputBufferCallback, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData);
static int audioCallbackInput(const void *inputBuffer, void *outputBufferCallback, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData);



pthread_mutex_t lock_out_buffer;
pthread_mutex_t lock_in_buffer;
pthread_mutex_t lock_next_inbuffer_start;



/* create circular buffers (for 8k and 48k) for tx and rx data. */
#define MAX_RX_BUFFER 800
#define MAX_TX_BUFFER 800

float inputBuffer[48000 * MAX_RX_BUFFER];
float outputBuffer[48000 * MAX_TX_BUFFER];
int inputBufferStart = 0;
int next_input_buffer_start = 0;
//int outputBufferStart = 0;
int max_ostream_sdblocks = 0;
int max_istream_sdblocks = 60;
int ostream_counter = 0;
int istream_counter = 0;
int sd_blocksize = 0;
PaError err;

PaStream *output_stream;
PaStream *input_stream;
//PortAudioStream *stream;



int get_waterfall_data(double* signal, float frequency, int size) {
    printf("is_signal_present\n");

    /* debug code to monitor phase lock*/
    float sig_mag = 5.0;

    return sig_mag;

}


int get_snr(double* signal, float frequency, int size) {
    printf("is_signal_present\n");

    /* debug code to monitor phase lock*/
    float sig_mag = 5.0;

    return sig_mag;

}



int buffer_tx_data(float* signal, int signal_len) {
    printf("audio_tx_data\n");

    //outputBufferStart = 0;

    pthread_mutex_lock(&lock_out_buffer);
    for(int i = 0; i < signal_len; i++) {
      outputBuffer[i] = signal[i];
    }
    pthread_mutex_unlock(&lock_out_buffer);
    max_ostream_sdblocks = signal_len / sd_blocksize;

    ostream_counter = 0;

}

/* extent is determined either by timing in sync mode or squelch traigger and reset in squelch mode */
int get_rx_data(float* signal, int buffer_start, int num_sd_blocks) {
    printf("get_rx_data\n");

    //buffer_start--;
    //if (buffer_start < 0){
    //    buffer_start = MAX_RX_BUFFER - 1;
    //}

    pthread_mutex_lock(&lock_in_buffer);
    for(int block_count = 0; block_count < num_sd_blocks; block_count++){
        int offset = buffer_start * sd_blocksize;
        int dest_offset = block_count * sd_blocksize;
        for(int i = 0; i < sd_blocksize; i++) {
            signal[dest_offset + i] = inputBuffer[offset + i];
        }
        buffer_start = (buffer_start + 1) % MAX_RX_BUFFER;
    }
    pthread_mutex_unlock(&lock_in_buffer);

}

int get_last_sd_block(float* signal) {
    printf("get_last_sd_block\n");

    pthread_mutex_lock(&lock_next_inbuffer_start);
    int buffer_start = next_input_buffer_start;
    pthread_mutex_unlock(&lock_next_inbuffer_start);
    buffer_start = (buffer_start + MAX_RX_BUFFER - 1) % MAX_RX_BUFFER;

    pthread_mutex_lock(&lock_in_buffer);
    int offset = buffer_start * sd_blocksize;
    for(int i = 0; i < sd_blocksize; i++) {
        signal[i] = inputBuffer[offset + i];
    }
    pthread_mutex_unlock(&lock_in_buffer);
}


int get_intput_buffer_start() {
    printf("get_intput_buffer_start\n");

    pthread_mutex_lock(&lock_next_inbuffer_start);
    int buffer_start = next_input_buffer_start;
    pthread_mutex_unlock(&lock_next_inbuffer_start);

    return buffer_start;
}


int terminate(){

    err = Pa_CloseStream(output_stream);
    if (err != paNoError){
        //printf("Pa_CloseStream ERROR\n");
        printf("Pa_CloseStream ERROR %s\n", Pa_GetErrorText(err));
    }
    err = Pa_CloseStream(input_stream);
    if (err != paNoError){
        //printf("Pa_CloseStream ERROR\n");
        printf("Pa_CloseStream ERROR %s\n", Pa_GetErrorText(err));
    }


    err = Pa_Terminate();
    if (err != paNoError){
        //printf("Pa_Terminate ERROR\n");
        printf("Pa_Terminate ERROR %s\n", Pa_GetErrorText(err));
    }

    pthread_mutex_destroy(&lock_out_buffer);
    pthread_mutex_destroy(&lock_in_buffer);
    pthread_mutex_destroy(&lock_next_inbuffer_start);


}

int initialize(char* strInputDeviceName, char* strOutputDeviceName, int sample_rate, int sd_bksize) {
    printf("initialize\n");

    sd_blocksize = sd_bksize;


    printf("sample_rate: %d\n", sample_rate);
    printf("sd_blocksize: %d\n", sd_blocksize);

    /* debug code to monitor phase lock*/
    int num_devices;
    int i;
    int iInputDeviceId;
    int iOutputDeviceId;


    err = Pa_Initialize();
    if (err != paNoError){
        printf("Pa_Initialize ERROR\n");
    }

    pthread_mutex_init(&lock_out_buffer, NULL);
    pthread_mutex_init(&lock_in_buffer, NULL);
    pthread_mutex_init(&lock_next_inbuffer_start, NULL);


    //set defaults
    iInputDeviceId  = Pa_GetDefaultInputDevice();
    iOutputDeviceId = Pa_GetDefaultOutputDevice();

    // find specific named devices
    num_devices = Pa_GetDeviceCount();
    for (i=0; i < num_devices; i++) {
        const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(i);
        printf("Device %d: %s\n", i, deviceInfo->name);

        // locate named input device
        if (strcmp(deviceInfo->name, strInputDeviceName ) == 0 && deviceInfo->maxInputChannels > 0) {
            printf("setting input device %d, %s\n", i, deviceInfo->name);
            iInputDeviceId = i;
        }
        else if (strcmp(deviceInfo->name, strOutputDeviceName ) == 0 && deviceInfo->maxOutputChannels > 0) {
            printf("setting output device %d, %s\n", i, deviceInfo->name);
            iOutputDeviceId = i;
        }
    }

    PaStreamParameters inputParameters;
    PaStreamParameters outputParameters;

    const PaDeviceInfo* deviceInfoInput  = Pa_GetDeviceInfo(iInputDeviceId);
    const PaDeviceInfo* deviceInfoOutput = Pa_GetDeviceInfo(iOutputDeviceId);

    printf("max input channels %d\n", deviceInfoInput->maxInputChannels);
    printf("default input sample_rate %f\n", deviceInfoInput->defaultSampleRate);

    printf("max output channels %d\n", deviceInfoOutput->maxOutputChannels);
    printf("default output sample_rate %f\n", deviceInfoOutput->defaultSampleRate);

    inputParameters.channelCount     = 1;
    inputParameters.device           = iInputDeviceId;
    inputParameters.sampleFormat     = paFloat32;
    inputParameters.suggestedLatency = deviceInfoInput->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    outputParameters.channelCount     = 1;
    outputParameters.device           = iOutputDeviceId;
    outputParameters.sampleFormat     = paFloat32;
    outputParameters.suggestedLatency = deviceInfoOutput->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;


    err = Pa_OpenStream(&output_stream, NULL, &outputParameters, 8000, sd_blocksize, paNoFlag, audioCallbackOutput, NULL);
    //err = Pa_OpenStream(&output_stream, NULL, &outputParameters, 8000, sd_blocksize, paClipOff, audioCallbackOutput, NULL);
    //err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, paFloat32, 48000, sd_blocksize, audioCallback, NULL);
    //err = Pa_OpenStream(&stream, NULL, &outputParameters, paFloat32, 8000, sd_blocksize, audioCallback, NULL);
    //err = Pa_OpenDefaultStream(&output_stream, 0, 1, paFloat32, 8000, sd_blocksize, audioCallbackOutput, NULL);
    if (err != paNoError){
        printf("Pa_OpenDefaultStream output ERROR %s\n", Pa_GetErrorText(err));
    }

    err = Pa_OpenStream(&input_stream, &inputParameters, NULL, 8000, sd_blocksize, paNoFlag, audioCallbackInput, NULL);
    //err = Pa_OpenDefaultStream(&input_stream, 1, 0, paFloat32, 8000, sd_blocksize, audioCallbackInput, NULL);
    if (err != paNoError){
        printf("Pa_OpenDefaultStream input ERROR %s\n", Pa_GetErrorText(err));
    }


}

int startStream(){
    printf("startStream\n");

    err = Pa_StartStream(output_stream);
    if (err != paNoError){
        printf("Pa_StartStream ERROR %s\n", Pa_GetErrorText(err));
    }
    err = Pa_StartStream(input_stream);
    if (err != paNoError){
        printf("Pa_StartStream ERROR %s\n", Pa_GetErrorText(err));
    }
}

int startOutputStream(){
    printf("startOutputStream\n");

    err = Pa_StartStream(output_stream);
    if (err != paNoError){
        printf("Pa_StartStream ERROR %s\n", Pa_GetErrorText(err));
    }
}

int startInputStream(){
    printf("startInputStream\n");

    err = Pa_StartStream(input_stream);
    if (err != paNoError){
        printf("Pa_StartStream ERROR %s\n", Pa_GetErrorText(err));
    }
}


int stopStream(){
    printf("stopStream\n");

    err = Pa_StopStream(output_stream);
    if (err != paNoError){
        printf("Pa_StopStream ERROR %s\n", Pa_GetErrorText(err));
    }
    err = Pa_StopStream(input_stream);
    if (err != paNoError){
        printf("Pa_StopStream ERROR %s\n", Pa_GetErrorText(err));
    }
}

int stopOutputStream(){
    printf("stopOutputStream\n");

    err = Pa_StopStream(output_stream);
    if (err != paNoError){
        printf("Pa_StopStream ERROR %s\n", Pa_GetErrorText(err));
    }
}

int stopInputStream(){
    printf("stopInputStream\n");

    err = Pa_StopStream(input_stream);
    if (err != paNoError){
        printf("Pa_StopStream ERROR %s\n", Pa_GetErrorText(err));
    }
}


static int audioCallbackOutput(const void *inputBufferCallback, void *outputBufferCallback, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData){
    printf("audioCallbackOutput\n");

    /* write data to output buffer */
    if(ostream_counter < max_ostream_sdblocks){
        int offset = ostream_counter * sd_blocksize;
        pthread_mutex_lock(&lock_out_buffer);
        for(int i = 0; i < sd_blocksize; i++) {
            ((float*)outputBufferCallback)[i] = outputBuffer[i + offset];
        }
        pthread_mutex_unlock(&lock_out_buffer);
        ostream_counter ++;
    }
    else{
        for(int i = 0; i < sd_blocksize; i++) {
            ((float*)outputBufferCallback)[i] = 0.0; // silence
        }
    }

    return 0;  //paContinue;
}

static int audioCallbackInput(const void *inputBufferCallback, void *outputBufferCallback, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData){
    printf("audioCallbackInput\n");

    /* read data from input buffer */
    
    int offset = istream_counter * sd_blocksize;

    pthread_mutex_lock(&lock_in_buffer);
    for(int i = 0; i < sd_blocksize; i++) {
        inputBuffer[i + offset] = ((float*)inputBufferCallback)[i];
    }
    pthread_mutex_unlock(&lock_in_buffer);

    istream_counter = (istream_counter + 1) % MAX_RX_BUFFER;
    //if (istream_counter == MAX_RX_BUFFER) {
    //  istream_counter = 0;
    //}
    pthread_mutex_lock(&lock_next_inbuffer_start);
    next_input_buffer_start = istream_counter;
    pthread_mutex_unlock(&lock_next_inbuffer_start);

    return 0;  //paContinue;
}

