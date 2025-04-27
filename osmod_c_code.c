#include <stdio.h>
#include <complex.h>
#include <math.h>

int mytestmain() {

    float phase_offset      = 0.8f;
    float frequency_offset  = 0.01f;
    unsigned int n          = 40;

    float complex x         = 0;
    float phase_error       = 0;
    float  phi_hat          = 0;
    float  complex y        = 0;

    unsigned int i;
    for (i=0; i<n; i++)  {
        x = cexpf(_Complex_I*(phase_offset + i * frequency_offset));
        y = cexpf(_Complex_I*phi_hat);
        phase_error = cargf(x*conjf(y));
        printf("%3u : phase = %12.8f, error = %12.8f\n", i, phi_hat, phase_error);
    }

    printf("done.\n");
    return 0;

}


int myfunc() {
    printf("Hello from C.\n");
}

int my_func_with_float_data(float*arr, int size) {
    printf("My C Func With Float Data.\n");

    for(int i = 0; i < size; i++) {
      printf("%f ", arr[i]);
      arr[i] = 0.0;

    }

    printf("\n");
}


int my_func_with_complex_data(complex double *arr, int size) {
    printf("My C Func With Complex Data.\n");

    for(int i = 0; i < size; i++) {
      printf("Element %d: real = %f, imag = %f\n", i, creal(arr[i]), cimag(arr[i]));
      arr[i] = (double complex){2.0, 3.0};
    }

    printf("\n");
}




int costas_loop_8psk(double* signal, double* recovered_signal1a, double* recovered_signal1b, float frequency, double* t, int size) {
    printf("Costas Loop\n");

    double complex recovered_signal2a;
    double complex recovered_signal2b;
    double complex recovered_signal2;
    double I45;
    double Q45;
    double I0;
    double Q0;
    double max1=0.0;
    double max2=0.0;
    double phase_normal;
    double phase_45;
    double phase_error;
    double  ratio;

    double term1;
    double term2;
    double term3;
    double term4;

    double frequency_estimate = 0.0;
    double phase_estimate = 0.0;
    double damping_factor = 1.0 / sqrt(2.0);
    double loop_bandwidth = 10.0;
    double K1 = 2.0 * damping_factor * loop_bandwidth ;
    double K2 = pow(loop_bandwidth, 2.0) ;

    for(int i = 0; i < size; i++) {
      recovered_signal1a[i] = signal[i] * cos(2 * M_PI * (frequency + frequency_estimate) * t[i] + phase_estimate);
      recovered_signal1b[i] = signal[i] * sin(2 * M_PI * (frequency + frequency_estimate) * t[i] + phase_estimate);
      recovered_signal2a    = signal[i] * cos(2 * M_PI * (frequency + frequency_estimate) * t[i] + phase_estimate + M_PI * 2 / 8);
      recovered_signal2b    = signal[i] * sin(2 * M_PI * (frequency + frequency_estimate) * t[i] + phase_estimate + M_PI * 2 / 8);
      recovered_signal2     = recovered_signal2a - recovered_signal2b;

      I0   = recovered_signal1a[i];
      Q0   = recovered_signal1b[i];
      I45 = creal(recovered_signal2);
      Q45 = cimag(recovered_signal2);

      phase_normal = I0 * Q0;
      phase_45     = I45 * Q45;
      if (abs(phase_normal) < abs(phase_45))
        phase_error  = phase_normal;
      else
        phase_error  = phase_45;

      frequency_estimate += K2 * phase_error ;
      phase_estimate += (K1 * phase_error) + frequency_estimate;

      max1 = fmax(max1, fabs(recovered_signal1a[i]));
      max2 = fmax(max2, fabs(recovered_signal1b[i]));
    }

    ratio = fmax(max1, max2) / 10000.0;   //parameters[3]
    for(int i = 0; i < size; i++) {
      recovered_signal1a[i] = recovered_signal1a[i]/ratio;
      recovered_signal1b[i] = recovered_signal1b[i]/ratio;
    }

    printf("Completed Costas Loop\n");

    return 0;
}















































