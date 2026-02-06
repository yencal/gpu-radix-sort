// scan_primitives.cuh
// Warp and block level scan primitives:
// - WarpScanInclusive
// - WarpScanExclusive
// - BlockScanInclusive
// - BlockScanExclusive

#pragma once

#include <cuda_runtime.h>

template <typename T>
static __device__ __forceinline__ T WarpScanInclusive(T value)
{
    const int lane = threadIdx.x % 32;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T tmp = __shfl_up_sync(0xFFFFFFFF, value, offset);
        if (lane >= offset) {
            value += tmp;
        }
    }
    return value;
}

template <typename T>
static __device__ __forceinline__ T WarpScanExclusive(T value)
{
    T inclusive = WarpScanInclusive(value);
    return inclusive - value;
}

template<int BLOCK_SIZE, typename T>
static __device__ __forceinline__ T BlockScanInclusive(T value)
{
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of warp size (32)");

    const int warp_idx = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    // Step 1: Warp-level inclusive scan
    T warp_scan = WarpScanInclusive(value);

    // Step 2: Last lane of each warp writes its total to shared memory
    __shared__ T warp_totals[NUM_WARPS];
    if (lane == 31) {
        warp_totals[warp_idx] = warp_scan;
    }
    __syncthreads();

    // Step 3: First warp scans the warp totals
    if (warp_idx == 0) {
        T warp_total = (lane < NUM_WARPS) ? warp_totals[lane] : 0;
        warp_total = WarpScanInclusive(warp_total);
        if (lane < NUM_WARPS) {
            warp_totals[lane] = warp_total;
        } 
    }
    __syncthreads();

    // Step 4: Add prefix from previous warps (exclusive)
    T warp_prefix = (warp_idx > 0) ? warp_totals[warp_idx - 1] : 0;
    return warp_scan + warp_prefix;
}

template<int BLOCK_SIZE, typename T>
static __device__ __forceinline__ T BlockScanExclusive(T value)
{
    T inclusive = BlockScanInclusive<BLOCK_SIZE>(value);
    return inclusive - value;
}