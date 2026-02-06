// utils.cuh
// Error checking and benchmark utilities

#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CHECK_CUDA(val) CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t err, const char* const func, 
                      const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================================
// DEVICE INFO
// ============================================================================

inline float GetPeakBandwidth()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    int memory_clock_khz;
    int memory_bus_width_bits;
    
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, 
                                      cudaDevAttrMemoryClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_bus_width_bits, 
                                      cudaDevAttrGlobalMemoryBusWidth, device));
    
    // DDR: multiply by 2
    float peak_bandwidth_gbs = 2.0f * memory_clock_khz * 
                               (memory_bus_width_bits / 8.0f) / 1e6f;
    
    return peak_bandwidth_gbs;
}

inline int GetNumSMs()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    int num_SMs = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_SMs,
                                      cudaDevAttrMultiProcessorCount, device));
    return num_SMs;
}

// ============================================================================
// TEST DATA INITIALIZATION AND VERIFICATION
// ============================================================================

inline void FillRandom(uint32_t* d_data, int num_elements)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerate(gen, reinterpret_cast<unsigned int*>(d_data), num_elements);
    curandDestroyGenerator(gen);
}

inline bool VerifySorted(const uint32_t* h_data, int num_elements)
{
    for (int i = 1; i < num_elements; ++i) {
        if (h_data[i] < h_data[i-1]) {
            return false;
        }
    }
    return true;
}