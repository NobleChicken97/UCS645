

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define THREADS 256
#define N_DEFAULT (1 << 18)

int allclose_f(const float* a, const float* b, int N, float atol)
{
    for (int i = 0; i < N; i++)
        if (fabsf(a[i] - b[i]) > atol) return 0;
    return 1;
}





__global__ void relu(const float* x, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(0.0f, x[i]);
}



__global__ void softmax(const float* logits, float* probs, int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row = logits + n * C;
    float*      out  = probs  + n * C;

    
    float maxVal = -1e30f;
    for (int c = 0; c < C; c++) maxVal = fmaxf(maxVal, row[c]);

    
    float sumExp = 0.0f;
    for (int c = 0; c < C; c++) sumExp += expf(row[c] - maxVal);

    
    for (int c = 0; c < C; c++) out[c] = expf(row[c] - maxVal) / sumExp;
}

void run_provided_softmax(void)
{
    int N = 4, C = 10;
    float *h_logits = (float*)malloc(N * C * sizeof(float));
    float *h_probs  = (float*)malloc(N * C * sizeof(float));
    for (int i = 0; i < N * C; i++) h_logits[i] = (float)rand() / RAND_MAX;

    float *d_logits, *d_probs;
    CUDA_CHECK(cudaMalloc(&d_logits, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs,  N * C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, N * C * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    softmax<<<blocks, threads>>>(d_logits, d_probs, N, C);
    CUDA_CHECK(cudaMemcpy(h_probs, d_probs, N * C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    
    int ok = 1;
    for (int n = 0; n < N; n++) {
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += h_probs[n * C + c];
        if (fabsf(s - 1.0f) > 1e-5f) { ok = 0; break; }
    }
    printf("  [A2-Softmax] Row sums = 1.0: %s\n", ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_logits); cudaFree(d_probs);
    free(h_logits); free(h_probs);
}





__global__ void sigmoid(const float* x, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void diy_sigmoid(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_x   = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i]  = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        h_ref[i] = 1.0f / (1.0f + expf(-h_x[i]));
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    sigmoid<<<blocks, threads>>>(d_x, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_ref, N, 1e-5f);
    printf("  [B1-Sigmoid] %s\n", ok ? "[PASS]" : "[FAIL] -- check expf formula");

    cudaFree(d_x); cudaFree(d_out);
    free(h_x); free(h_out); free(h_ref);
}



__global__ void tanhKernel(const float* x, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        out[i] = tanhf(x[i]);
    }
}

void diy_tanh(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_x   = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i]  = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;
        h_ref[i] = tanhf(h_x[i]);
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    tanhKernel<<<blocks, threads>>>(d_x, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_ref, N, 1e-5f);
    printf("  [B2-Tanh] %s\n", ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_x); cudaFree(d_out);
    free(h_x); free(h_out); free(h_ref);
}



__global__ void leakyRelu(const float* x, float* out, float alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        out[i] = (x[i] > 0.0f) ? x[i] : alpha * x[i];
    }
}

void diy_leaky_relu(int N, float alpha)
{
    size_t bytes = N * sizeof(float);
    float *h_x   = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i]  = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        h_ref[i] = h_x[i] > 0.0f ? h_x[i] : alpha * h_x[i];
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    leakyRelu<<<blocks, threads>>>(d_x, d_out, alpha, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_ref, N, 1e-5f);
    printf("  [B3-LeakyReLU] alpha=%.2f  %s\n", alpha, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_x); cudaFree(d_out);
    free(h_x); free(h_out); free(h_ref);
}



__global__ void reluBackward(const float* dOut, const float* x_fwd,
                             float* dIn, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        dIn[i] = (x_fwd[i] > 0.0f) ? dOut[i] : 0.0f;
    }
}

void diy_relu_backward(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_x    = (float*)malloc(bytes);
    float *h_dOut = (float*)malloc(bytes);
    float *h_dIn  = (float*)malloc(bytes);
    float *h_ref  = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i]    = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        h_dOut[i] = (float)rand() / RAND_MAX;
        h_ref[i]  = h_x[i] > 0.0f ? h_dOut[i] : 0.0f;
    }

    float *d_x, *d_dOut, *d_dIn;
    CUDA_CHECK(cudaMalloc(&d_x,    bytes));
    CUDA_CHECK(cudaMalloc(&d_dOut, bytes));
    CUDA_CHECK(cudaMalloc(&d_dIn,  bytes));
    CUDA_CHECK(cudaMemcpy(d_x,    h_x,    bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dOut, h_dOut, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dIn, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    reluBackward<<<blocks, threads>>>(d_dOut, d_x, d_dIn, N);
    CUDA_CHECK(cudaMemcpy(h_dIn, d_dIn, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_dIn, h_ref, N, 1e-5f);
    printf("  [B4-ReLUBackward] %s\n", ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_x); cudaFree(d_dOut); cudaFree(d_dIn);
    free(h_x); free(h_dOut); free(h_dIn); free(h_ref);
}





__global__ void bceLoss(const float* pred, const float* target,
                        float* loss, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        float p = fmaxf(fminf(pred[i], 1.0f - 1e-7f), 1e-7f);
        loss[i] = -(target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p));
    }
}

void diy_bce_loss(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_pred   = (float*)malloc(bytes);
    float *h_target = (float*)malloc(bytes);
    float *h_loss   = (float*)malloc(bytes);
    float *h_ref    = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_pred[i]   = (float)rand() / RAND_MAX;
        h_target[i] = (rand() % 2) ? 1.0f : 0.0f;
        float p     = fmaxf(fminf(h_pred[i], 1.0f - 1e-7f), 1e-7f);
        h_ref[i]    = -(h_target[i] * logf(p) + (1.0f - h_target[i]) * logf(1.0f - p));
    }

    float *d_pred, *d_target, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_pred,   bytes));
    CUDA_CHECK(cudaMalloc(&d_target, bytes));
    CUDA_CHECK(cudaMalloc(&d_loss,   bytes));
    CUDA_CHECK(cudaMemcpy(d_pred,   h_pred,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_loss, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    bceLoss<<<blocks, threads>>>(d_pred, d_target, d_loss, N);
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_loss, h_ref, N, 1e-4f);
    printf("  [C1-BCE-Loss] %s\n", ok ? "[PASS]" : "[FAIL] -- check clip + log formula");

    cudaFree(d_pred); cudaFree(d_target); cudaFree(d_loss);
    free(h_pred); free(h_target); free(h_loss); free(h_ref);
}



__global__ void crossEntropyLoss(const float* logits, const int* labels,
                                 float* loss, int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row = logits + n * C;

    
    float maxVal = -1e30f;
    for (int c = 0; c < C; c++) maxVal = fmaxf(maxVal, row[c]);

    float sumExp = 0.0f;
    for (int c = 0; c < C; c++) sumExp += expf(row[c] - maxVal);

    int label = labels[n];
    loss[n] = -(row[label] - maxVal) + logf(sumExp);
}

void diy_cross_entropy(int N, int C)
{
    size_t logit_bytes = (size_t)N * C * sizeof(float);
    size_t label_bytes = N * sizeof(int);
    size_t loss_bytes  = N * sizeof(float);

    float *h_logits = (float*)malloc(logit_bytes);
    int   *h_labels = (int*)malloc(label_bytes);
    float *h_loss   = (float*)malloc(loss_bytes);
    float *h_ref    = (float*)malloc(loss_bytes);

    for (int n = 0; n < N; n++) {
        float maxV = -1e30f, sumE = 0.0f;
        for (int c = 0; c < C; c++) {
            h_logits[n*C+c] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
            if (h_logits[n*C+c] > maxV) maxV = h_logits[n*C+c];
        }
        for (int c = 0; c < C; c++) sumE += expf(h_logits[n*C+c] - maxV);
        h_labels[n] = rand() % C;
        h_ref[n]    = -(h_logits[n*C + h_labels[n]] - maxV) + logf(sumE);
    }

    float *d_logits; int *d_labels; float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_logits, logit_bytes));
    CUDA_CHECK(cudaMalloc(&d_labels, label_bytes));
    CUDA_CHECK(cudaMalloc(&d_loss,   loss_bytes));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, logit_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, label_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_loss, 0, loss_bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    crossEntropyLoss<<<blocks, threads>>>(d_logits, d_labels, d_loss, N, C);
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, loss_bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_loss, h_ref, N, 1e-4f);
    printf("  [C2-CrossEntropy] N=%d C=%d  %s\n", N, C, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_logits); cudaFree(d_labels); cudaFree(d_loss);
    free(h_logits); free(h_labels); free(h_loss); free(h_ref);
}





__global__ void adamUpdate(float* w, const float* g, float* m, float* v,
                           float lr, float beta1, float beta2, float eps,
                           float b1t, float b2t, int N)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        float mhat = m[i] / (1.0f - b1t);
        float vhat = v[i] / (1.0f - b2t);
        w[i] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

void stretch_adam(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_w = (float*)malloc(bytes);
    float *h_g = (float*)malloc(bytes);
    float *h_m = (float*)calloc(N, sizeof(float));
    float *h_v = (float*)calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) { h_w[i] = (float)rand() / RAND_MAX; h_g[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f; }

    float *d_w, *d_g, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_w, bytes)); CUDA_CHECK(cudaMalloc(&d_g, bytes));
    CUDA_CHECK(cudaMalloc(&d_m, bytes)); CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));

    float lr=1e-3f, b1=0.9f, b2=0.999f, eps=1e-8f;
    int threads = THREADS, blocks = (N + threads - 1) / threads;
    int ok = 1;

    for (int t = 1; t <= 5; t++) {
        float b1t = powf(b1, t), b2t = powf(b2, t);
        adamUpdate<<<blocks, threads>>>(d_w, d_g, d_m, d_v, lr, b1, b2, eps, b1t, b2t, N);

        
        for (int i = 0; i < N; i++) {
            h_m[i] = b1 * h_m[i] + (1.0f - b1) * h_g[i];
            h_v[i] = b2 * h_v[i] + (1.0f - b2) * h_g[i] * h_g[i];
            float mhat = h_m[i] / (1.0f - b1t);
            float vhat = h_v[i] / (1.0f - b2t);
            h_w[i] -= lr * mhat / (sqrtf(vhat) + eps);
        }
    }

    float *h_w_gpu = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_w_gpu, d_w, bytes, cudaMemcpyDeviceToHost));
    if (!allclose_f(h_w_gpu, h_w, N, 1e-5f)) ok = 0;

    printf("  [D1-Adam] 5 steps  %s\n", ok ? "[PASS]" : "[FAIL]");
    cudaFree(d_w); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    free(h_w); free(h_g); free(h_m); free(h_v); free(h_w_gpu);
}



int main(void)
{
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 3: ML Primitives\n");
    printf("========================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n\n", prop.name);

    printf("[Section A] Reference:\n");
    run_provided_softmax();

    printf("\n[Section B] DIY Activations:\n");
    diy_sigmoid(N_DEFAULT);
    diy_tanh(N_DEFAULT);
    diy_leaky_relu(N_DEFAULT, 0.01f);
    diy_relu_backward(N_DEFAULT);

    printf("\n[Section C] DIY Loss Functions:\n");
    diy_bce_loss(N_DEFAULT);
    diy_cross_entropy(512, 10);

    printf("\n[Section D] Stretch — Adam Optimizer:\n");
    stretch_adam(1 << 16);

    printf("\n========================================================\n");
    printf("  All [PASS] = ready for Exercise 4!\n");
    printf("========================================================\n\n");
    return 0;
}
