"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = expf(a[idx]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        out[r] = matrix[r * cols];
        for (int i = 0; i < cols; ++i) {
            out[r] = fmaxf(matrix[r * cols + i], out[r]);
        }
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sm = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sm += matrix[row * cols + i];
        }
        out[row] = sm;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b
    float dot_product = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot_product += a[i] * b[i];
    }
    return dot_product;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float result = 0.0f;
        for (int i = 0; i < k; ++i) {
            result += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = result;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col]; 
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] = dot_product(q + row * head_dim, k + col * head_dim, head_dim) / sqrtf(head_dim);
    }
}

# Step 10 - softmax_rows
#include <cfloat>

__global__ void softmax_rows(float* matrix, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    float *row_ptr = matrix + row * cols;
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float max_row = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += expf(row_ptr[i] - max_row);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) 
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float sum_row = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] = expf(row_ptr[i] - max_row) / sum_row;
    __syncthreads();
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= seq_len || col >= head_dim) return;
    
    float tmp_val = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        tmp_val += p[row * seq_len + i] * v[i * head_dim + col];
    }
    out[row * head_dim + col] = tmp_val;
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    int BLOCK_SIZE = 32;
    int GRID_SIZE_ROW = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int GRID_SIZE_COL =  (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_scores;
    cudaMalloc(&d_scores, (size_t)seq_len * seq_len * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_qk(GRID_SIZE_ROW, GRID_SIZE_ROW);
    dim3 grid(GRID_SIZE_COL, GRID_SIZE_ROW);

    qk_scores<<<grid_qk, threads>>>(d_q, d_k, d_scores, seq_len, head_dim);
    softmax_rows<<<seq_len, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_scores, seq_len, seq_len);
    pv_matmul<<<grid, threads>>>(d_scores, d_v, d_out, seq_len, head_dim);
}

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

