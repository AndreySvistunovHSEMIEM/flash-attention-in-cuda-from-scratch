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

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    // TODO: combine the rescaled old sum with the new block sum
    return correction * old_sum + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // TODO: multiply each of the head_dim entries of out_row by correction in place
	for (int idx = 0; idx < head_dim; ++idx) {
		out_row[idx] *= correction;
	}
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // TODO: cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    int tile_size = tile_rows * tile_cols;
    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int tile_row = idx / tile_cols;
        int tile_col = idx % tile_cols;

        int row = src_row_start + tile_row;
        int col = src_col_start + tile_col;

        if (row < src_rows && col < src_cols) {
            shared_dst[idx] = src[row * src_cols + col];
        } else {
            shared_dst[idx] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    int tile_size = tile_q * tile_k;
    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int row = idx / tile_k;
        int col = idx % tile_k;
        float dot_product = 0.0f;
        for (int j = 0; j < head_dim; ++j) {
            dot_product += q_tile[row * head_dim + j] * k_tile[col * head_dim + j];
        }
        s_tile[row * tile_k + col] = scale * dot_product;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write row_max_out[r] = max over c of s_tile[r, c]
    for (int idx = thread_id; idx < tile_q; idx += num_threads) {
        int row_idx = idx * tile_k;
        float max_value = s_tile[row_idx];
        for (int j = 1; j < tile_k; ++j) {
            max_value = fmaxf(max_value, s_tile[row_idx + j]);
        }
        row_max_out[idx] = max_value;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // TODO: for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    int tile_size = tile_q * tile_k;
    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int row = idx / tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[row]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill row_sum_out[r] with the sum of p_tile row r
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float sum_result = 0.0f;
        for (int col = 0; col < tile_k; ++col) {
            sum_result += p_tile[row * tile_k + col];
        }
        row_sum_out[row] = sum_result;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    // TODO: cooperatively add P_tile * V_tile into out_acc
    int tile_size = tile_q * head_dim;
    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        float dot_product = 0.0f;
        for (int j = 0; j < tile_k; ++j) {
            dot_product += p_tile[row * tile_k + j] * v_tile[j * head_dim + col];
        }
        out_acc[row * head_dim + col] += dot_product;
    }
}

# Step 23 - flash_attention_kernel
#include <cfloat>

__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    // TODO: tiled fused attention using shared memory and online softmax.
    extern __shared__ float smem[];

    float* q_tile = smem;
    float* k_tile = q_tile + tile_q * head_dim;
    float* v_tile = k_tile + tile_k * head_dim;
    float* scores_tile = v_tile + tile_k * head_dim;
    float* row_max_tile = scores_tile + tile_q * tile_k;
    float* row_block_max_tile = row_max_tile + tile_q;
    float* exp_sum_tile = row_block_max_tile + tile_q;
    float* accumulative_tile = exp_sum_tile + tile_q;

    int tidx = threadIdx.x;
    int num_threads = blockDim.x;
    int q_row_start = blockIdx.x * tile_q;

    load_tile(q, q_tile, q_row_start, 0,
        seq_len, head_dim, tile_q, head_dim, tidx, num_threads);
    for (int idx = tidx; idx < tile_q; idx += num_threads) {
        row_max_tile[idx] = -FLT_MAX;
        row_block_max_tile[idx] = -FLT_MAX;
        exp_sum_tile[idx] = 0.0f;
    }
    for (int idx = tidx; idx < tile_q * head_dim; idx += num_threads) {
        accumulative_tile[idx] = 0.0f;
    }
    __syncthreads();

    int kv_iterations = (seq_len + tile_k - 1) / tile_k;
    for (int kt = 0; kt < kv_iterations; ++kt) {
        int k_row_start = kt * tile_k;
        load_tile(k, k_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tidx, num_threads);
        load_tile(v, v_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tidx, num_threads);
        __syncthreads();

        tile_scores(q_tile, k_tile, scores_tile, tile_q, tile_k, head_dim, scale, tidx, num_threads);
        __syncthreads();

        for (int idx = tidx; idx < tile_q * tile_k; idx += num_threads) {
            int row = idx % tile_k;
            if (k_row_start + row >= seq_len) {
                scores_tile[idx] = -FLT_MAX;
            }
        }
        __syncthreads();

        tile_rowmax(scores_tile, row_max_tile, tile_q, tile_k, tidx, num_threads);
        __syncthreads();

        for (int idx = tidx; idx < tile_q; idx += num_threads) {
            float m_new = online_max(row_block_max_tile[idx], row_max_tile[idx]);
            float alpha = correction_factor(row_block_max_tile[idx], m_new);
            rescale_output(&accumulative_tile[idx * head_dim], head_dim, alpha);
            row_block_max_tile[idx] = m_new;
            row_max_tile[idx] = m_new;
            exp_sum_tile[idx] *= alpha;
        }
        __syncthreads();

        tile_exp(scores_tile, row_max_tile, tile_q, tile_k, tidx, num_threads);
        __syncthreads();

        tile_rowsum(scores_tile, row_max_tile, tile_q, tile_k, tidx, num_threads);
        __syncthreads();

        for (int idx = tidx; idx < tile_q; idx += num_threads) {
            exp_sum_tile[idx] += row_max_tile[idx];
        }
        __syncthreads();

        accumulate_pv(scores_tile, v_tile, accumulative_tile, tile_q, tile_k, head_dim, tidx, num_threads);
        __syncthreads();
    }
    for (int idx = tidx; idx < tile_q * head_dim; idx += num_threads) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        int global_row = q_row_start + row;
        if (global_row < seq_len) {
            out[global_row * head_dim + col] = accumulative_tile[idx] / exp_sum_tile[row];
        }
    }
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    // TODO: configure grid/block/shared memory and launch flash_attention_kernel
    int BLOCK_SIZE = 256;
    int GRID_SIZE = (seq_len * head_dim - 1 + BLOCK_SIZE) / BLOCK_SIZE;
    size_t shared_memory_capacity = (2 * tile_q * head_dim + 2 * tile_k * head_dim + tile_q * tile_k + tile_q * 3) * sizeof(float);

    float scale = 1 / sqrtf(head_dim);
    flash_attention_kernel<<<GRID_SIZE, BLOCK_SIZE, shared_memory_capacity>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale
    );
}

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

