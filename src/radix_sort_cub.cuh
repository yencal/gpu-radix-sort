#pragma once

#include <cub/cub.cuh>

struct RadixSortCUB {
    static size_t GetTempSize(uint32_t n) {
        size_t temp_size = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_size,
            (uint32_t*)nullptr, (uint32_t*)nullptr, n);
        return temp_size;
    }

    static void Run(uint32_t* d_input, uint32_t* d_output, uint32_t n, void* d_temp) {
        size_t temp_size = GetTempSize(n);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_size,
            d_input, d_output, n);
    }
};