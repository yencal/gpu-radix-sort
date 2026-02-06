#pragma once

#include "utils.cuh"
#include "scan_primitives.cuh"

// ============================================================================
// Kernel 1: Upsweep
// Computes two things simultaneously:
// - g_global_hist[digit]: total count of each digit across all keys
// - g_block_hist[digit][block]: count of each digit within each block
// ============================================================================

template<int RADIX = 256, int BLOCK_THREADS = 256, int ITEMS_PER_THREAD = 8>
__global__ void Upsweep(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ g_global_hist,
    uint32_t* __restrict__ g_block_hist,
    uint32_t num_keys,
    uint32_t radix_shift)
{
    constexpr int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    __shared__ uint32_t smem_hist[RADIX];

    uint32_t items[ITEMS_PER_THREAD];
    const int block_offset = blockIdx.x * ITEMS_PER_BLOCK;
    const int block_valid_items = min(ITEMS_PER_BLOCK, (int)(num_keys - block_offset));

    LoadStripedVectorized<uint32_t, uint4, BLOCK_THREADS, ITEMS_PER_THREAD>(
        input + block_offset, items, block_valid_items);

    for (int i = threadIdx.x; i < RADIX; i += BLOCK_THREADS) {
        smem_hist[i] = 0;
    }
    __syncthreads();

    // Use striped indexing for bounds check
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (threadIdx.x + i * BLOCK_THREADS < block_valid_items) {
            atomicAdd(&smem_hist[(items[i] >> radix_shift) & (RADIX - 1)], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < RADIX; i += BLOCK_THREADS) {
        uint32_t count = smem_hist[i];
        g_block_hist[i * gridDim.x + blockIdx.x] = count;
        if (count > 0) atomicAdd(&g_global_hist[i], count);
    }
}

// ============================================================================
// Kernel 2: Scan
// Exclusive prefix sum on g_block_hist (per digit) and g_global_hist
// ============================================================================

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockExclusiveScan(
    T (&items)[ITEMS_PER_THREAD],
    T carry,
    T* s_block_total)
{
    // Thread-local inclusive scan
    #pragma unroll
    for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
        items[i] += items[i - 1];
    }

    // Block scan on thread totals
    T thread_total = items[ITEMS_PER_THREAD - 1];
    T thread_prefix = BlockScanExclusive<BLOCK_THREADS>(thread_total);

    // Last thread stores block total
    if (threadIdx.x == BLOCK_THREADS - 1) {
        *s_block_total = thread_prefix + thread_total;
    }

    // Convert inclusive to exclusive + add carry
    #pragma unroll
    for (int i = ITEMS_PER_THREAD - 1; i > 0; --i) {
        items[i] = items[i - 1] + thread_prefix + carry;
    }
    items[0] = thread_prefix + carry;
}

template<int RADIX = 256, int BLOCK_THREADS = 256, int ITEMS_PER_THREAD = 8>
__global__ void Scan(
    uint32_t* __restrict__ g_global_hist,
    uint32_t* __restrict__ g_block_hist,
    uint32_t num_blocks)
{
    static_assert(BLOCK_THREADS * ITEMS_PER_THREAD >= RADIX, "Tile must fit RADIX for global hist scan");

    constexpr int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    __shared__ uint32_t smem[ITEMS_PER_BLOCK];
    __shared__ uint32_t s_block_total;
    
    uint32_t items[ITEMS_PER_THREAD];
    uint32_t* digit_hist = g_block_hist + blockIdx.x * num_blocks;
    uint32_t carry = 0;

    for (uint32_t tile_offset = 0; tile_offset < num_blocks; tile_offset += ITEMS_PER_BLOCK) {
        uint32_t valid_items = min(ITEMS_PER_BLOCK, num_blocks - tile_offset);
        
        LoadStripedToBlocked<uint32_t, uint2, BLOCK_THREADS, ITEMS_PER_THREAD>(
            digit_hist + tile_offset, items, smem, valid_items);
        __syncthreads();

        BlockExclusiveScan<BLOCK_THREADS, ITEMS_PER_THREAD>(items, carry, &s_block_total);
        __syncthreads();
        
        carry = s_block_total;

        StoreBlockedToStriped<uint32_t, uint2, BLOCK_THREADS, ITEMS_PER_THREAD>(
            digit_hist + tile_offset, items, smem, valid_items);
    }

    // Global histogram (block 0 only)
    if (blockIdx.x == 0) {
        __syncthreads();
        
        LoadStripedToBlocked<uint32_t, uint2, BLOCK_THREADS, ITEMS_PER_THREAD>(
            g_global_hist, items, smem, RADIX);
        __syncthreads();

        BlockExclusiveScan<BLOCK_THREADS, ITEMS_PER_THREAD>(items, 0, &s_block_total);
        __syncthreads();

        StoreBlockedToStriped<uint32_t, uint2, BLOCK_THREADS, ITEMS_PER_THREAD>(
            g_global_hist, items, smem, RADIX);
    }
}

// ============================================================================
// Kernel 3: Downsweep
// Reorders keys using digit offsets from Upsweep/Scan
// Uses WLMS (Warp-Level Multi-Split) for ranking
// Two-phase scatter: registers -> shared memory -> global memory
// ============================================================================

template<
    int RADIX = 256,
    int RADIX_LOG = 8,
    int BLOCK_THREADS = 256,
    int ITEMS_PER_THREAD = 8>
__global__ void Downsweep(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ g_global_hist,
    const uint32_t* __restrict__ g_block_hist,
    uint32_t num_keys,
    uint32_t radix_shift)
{
    static_assert(RADIX == (1 << RADIX_LOG), "RADIX must match RADIX_LOG");
    static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of 32");
    static_assert(BLOCK_THREADS >= RADIX, "Need BLOCK_THREADS >= RADIX for Phase 4");

    constexpr int ITEMS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    constexpr int RADIX_MASK = RADIX - 1;

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t lane_mask_lt = (1U << lane_idx) - 1;
    const uint32_t block_offset = blockIdx.x * ITEMS_PER_BLOCK;

    // Shared memory allocations
    constexpr int SMEM_HIST_SIZE = NUM_WARPS * RADIX;
    constexpr int SMEM_EXCHANGE_SIZE = ITEMS_PER_BLOCK;
    constexpr int SMEM_SIZE = (SMEM_HIST_SIZE > SMEM_EXCHANGE_SIZE) ? SMEM_HIST_SIZE : SMEM_EXCHANGE_SIZE;
    __shared__ uint32_t s_hist_exchange[SMEM_SIZE];

    __shared__ union {
        uint32_t digit_total[RADIX];    // digit totals for scan
        uint32_t digit_start[RADIX];    // cross-digit offsets
    } smem;

    // === Phase 1: Clear warp histograms ===
    for (int i = threadIdx.x; i < SMEM_SIZE; i += BLOCK_THREADS) {
        s_hist_exchange[i] = 0;
    }
    __syncthreads();

    // === Phase 2: Load keys (striped) ===
    uint32_t keys[ITEMS_PER_THREAD];
    uint8_t digits[ITEMS_PER_THREAD];
    uint16_t warp_offsets[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint32_t idx = block_offset + threadIdx.x + i * BLOCK_THREADS;
        // Load key or sentinel (0xFFFFFFFF sorts last, won't be stored)
        keys[i] = (idx < num_keys) ? input[idx] : 0xFFFFFFFF;
        digits[i] = (keys[i] >> radix_shift) & RADIX_MASK;
    }

    // === Phase 3: WLMS - compute warp-local offsets ===
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint32_t digit = digits[i];
        
        // Match lanes with same digit
        uint32_t match = 0xFFFFFFFF;
        #pragma unroll
        for (int bit = 0; bit < RADIX_LOG; ++bit) {
            uint32_t digit_bit = (digit >> bit) & 1;
            uint32_t ballot = __ballot_sync(0xFFFFFFFF, digit_bit);
            match &= digit_bit ? ballot : ~ballot;
        }

        // Rank within matching lanes
        uint32_t lane_rank = __popc(match & lane_mask_lt);
        uint32_t warp_count = __popc(match);

        // First matching lane does atomic add
        uint32_t pre_increment = 0;
        if (lane_rank == 0) {
            pre_increment = atomicAdd(&s_hist_exchange[warp_idx * RADIX + digit], warp_count);
        }

        // Broadcast pre_increment to all matching lanes
        uint32_t first_lane = __ffs(match) - 1;
        pre_increment = __shfl_sync(0xFFFFFFFF, pre_increment, first_lane);

        warp_offsets[i] = pre_increment + lane_rank;
    }
    __syncthreads();

    // === Phase 4: Compute warp prefix + cross-digit prefix ===
    if (threadIdx.x < RADIX) {
        uint32_t digit = threadIdx.x;
        uint32_t warp_prefix = 0;

        // Exclusive scan across warps for this digit
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            uint32_t count = s_hist_exchange[w * RADIX + digit];
            s_hist_exchange[w * RADIX + digit] = warp_prefix;
            warp_prefix += count;
        }
        
        // warp_prefix now = total count for this digit in this block
        smem.digit_total[digit] = warp_prefix;
    }
    __syncthreads();

    // Cross-digit exclusive scan
    if (threadIdx.x < RADIX) {
        uint32_t digit = threadIdx.x;
        uint32_t cross_digit_offset = BlockScanExclusive<RADIX>(smem.digit_total[digit]);
        smem.digit_start[digit] = cross_digit_offset;
    }
    __syncthreads();

    // Accumulate full offset into registers while s_hist_exchange still holds warp prefixes
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        warp_offsets[i] += smem.digit_start[digits[i]]
                        + s_hist_exchange[warp_idx * RADIX + digits[i]];
    }
    __syncthreads();  // s_hist_exchange is now dead, safe to reuse

    // === Phase 5: Scatter to shared memory ===
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        s_hist_exchange[warp_offsets[i]] = keys[i];
    }
    __syncthreads();

    // === Phase 6: Compute global base with subtraction trick and Scatter to global memory ===
    uint32_t valid_items = min(ITEMS_PER_BLOCK, (int)(num_keys - block_offset));
    for (int i = threadIdx.x; i < valid_items; i += BLOCK_THREADS) {
        uint32_t key = s_hist_exchange[i];
        uint32_t digit = (key >> radix_shift) & RADIX_MASK;
        uint32_t dst = g_global_hist[digit] 
                     + g_block_hist[digit * gridDim.x + blockIdx.x]
                     - smem.digit_start[digit]
                     + i;
        output[dst] = key;
    }
}