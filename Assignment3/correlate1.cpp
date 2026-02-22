#include <cmath>
#include <cstdlib>

void correlate(int ny, int nx, const float *data, float *result) {
    double *norm = (double *)malloc(ny * nx * sizeof(double));

    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++)
            sum += data[y * nx + x];
        double mean = sum / nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = data[y * nx + x] - mean;
            norm[y * nx + x] = val;
            sq_sum += val * val;
        }

        double inv_len = 1.0 / sqrt(sq_sum);
        for (int x = 0; x < nx; x++)
            norm[y * nx + x] *= inv_len;
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[i * nx + x] * norm[j * nx + x];
            result[i + j * ny] = (float)dot;
        }
    }

    free(norm);
}
