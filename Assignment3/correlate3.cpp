#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

void correlate(int ny, int nx, const float *data, float *result) {
    int padded_nx = (nx + 7) & ~7;
    double *norm = (double *)aligned_alloc(64, (size_t)ny * padded_nx * sizeof(double));
    memset(norm, 0, (size_t)ny * padded_nx * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++)
            sum += data[y * nx + x];
        double mean = sum / nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = data[y * nx + x] - mean;
            norm[y * padded_nx + x] = val;
            sq_sum += val * val;
        }

        double inv_len = 1.0 / sqrt(sq_sum);
        #pragma omp simd
        for (int x = 0; x < padded_nx; x++)
            norm[y * padded_nx + x] *= inv_len;
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ny; i++) {
        const double *row_i = &norm[i * padded_nx];

        int j = 0;
        for (; j + 3 <= i; j += 4) {
            double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;
            const double *r0 = &norm[(j + 0) * padded_nx];
            const double *r1 = &norm[(j + 1) * padded_nx];
            const double *r2 = &norm[(j + 2) * padded_nx];
            const double *r3 = &norm[(j + 3) * padded_nx];

            #pragma omp simd reduction(+:dot0,dot1,dot2,dot3)
            for (int x = 0; x < padded_nx; x++) {
                double vi = row_i[x];
                dot0 += vi * r0[x];
                dot1 += vi * r1[x];
                dot2 += vi * r2[x];
                dot3 += vi * r3[x];
            }

            result[i + (j + 0) * ny] = (float)dot0;
            result[i + (j + 1) * ny] = (float)dot1;
            result[i + (j + 2) * ny] = (float)dot2;
            result[i + (j + 3) * ny] = (float)dot3;
        }

        for (; j <= i; j++) {
            double dot = 0.0;
            const double *row_j = &norm[j * padded_nx];
            #pragma omp simd reduction(+:dot)
            for (int x = 0; x < padded_nx; x++)
                dot += row_i[x] * row_j[x];
            result[i + j * ny] = (float)dot;
        }
    }

    free(norm);
}
