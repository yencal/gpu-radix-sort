#pragma once

#include "radix_sort_multikernel.cuh"

template<int RADIX_LOG = 8, int BLOCK_THREADS = 256, int ITEMS_PER_THREAD = 8>
struct RadixSortMultikernel {
    static constexpr int RADIX = 1 << RADIX_LOG;
    static constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr int NUM_PASSES = (32 + RADIX_LOG - 1) / RADIX_LOG;  // ceil(32 / RADIX_LOG)

    static_assert(RADIX_LOG >= 1 && RADIX_LOG <= 16, "RADIX_LOG must be in [1, 16]");
    static_assert(32 % RADIX_LOG == 0 || NUM_PASSES * RADIX_LOG >= 32, 
                  "RADIX_LOG should evenly divide 32 for clean passes");

    static size_t GetTempSize(uint32_t n) {
        const uint32_t num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
        return (RADIX + RADIX * num_blocks) * sizeof(uint32_t);
    }

    static void Run(uint32_t* d_input, uint32_t* d_output, uint32_t n, void* d_temp) {
        const uint32_t num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;

        uint32_t* g_global_hist = static_cast<uint32_t*>(d_temp);
        uint32_t* g_block_hist = g_global_hist + RADIX;

        uint32_t* d_src = d_input;
        uint32_t* d_dst = d_output;

        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            uint32_t radix_shift = pass * RADIX_LOG;

            CHECK_CUDA(cudaMemsetAsync(g_global_hist, 0, RADIX * sizeof(uint32_t)));

            Upsweep<RADIX, BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<num_blocks, BLOCK_THREADS>>>(
                    d_src, g_global_hist, g_block_hist, n, radix_shift);

            Scan<RADIX, BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<RADIX, BLOCK_THREADS>>>(
                    g_global_hist, g_block_hist, num_blocks);

            Downsweep<RADIX, RADIX_LOG, BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<num_blocks, BLOCK_THREADS>>>(
                    d_src, d_dst, g_global_hist, g_block_hist, n, radix_shift);

            std::swap(d_src, d_dst);
        }

        // Copy to d_output if needed
        if (d_src != d_output) {
            CHECK_CUDA(cudaMemcpy(d_output, d_src, n * sizeof(uint32_t), 
                cudaMemcpyDeviceToDevice));
        }
    }
};