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
// Device Helper Functions
// ============================================================================

template <
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void LoadStripedVectorized(
    const T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    int valid_items)
{
    constexpr int VECTOR_WIDTH = sizeof(VectorT) / sizeof(T);
    constexpr int VECTORS_PER_THREAD = ITEMS_PER_THREAD / VECTOR_WIDTH;
    
    static_assert(sizeof(VectorT) % sizeof(T) == 0, 
        "VectorT size must be multiple of T size");
    static_assert(ITEMS_PER_THREAD % VECTOR_WIDTH == 0,
        "ITEMS_PER_THREAD must be multiple of VECTOR_WIDTH");

    const VectorT* vec_ptr = reinterpret_cast<const VectorT*>(block_ptr);

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        const int vec_idx = threadIdx.x + v * BLOCK_THREADS;
        const int base_idx = vec_idx * VECTOR_WIDTH;

        if (base_idx + VECTOR_WIDTH - 1 < valid_items) {
            VectorT loaded = vec_ptr[vec_idx];
            const T* scalars = reinterpret_cast<const T*>(&loaded);
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; i++) {
                items[v * VECTOR_WIDTH + i] = scalars[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; i++) {
                int idx = base_idx + i;
                items[v * VECTOR_WIDTH + i] = (idx < valid_items) ? block_ptr[idx] : T{};
            }
        }
    }
}

template <
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void StoreStripedVectorized(
    T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    int valid_items)
{
    constexpr int VECTOR_WIDTH = sizeof(VectorT) / sizeof(T);
    constexpr int VECTORS_PER_THREAD = ITEMS_PER_THREAD / VECTOR_WIDTH;

    VectorT* vec_ptr = reinterpret_cast<VectorT*>(block_ptr);

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        const int vec_idx = threadIdx.x + v * BLOCK_THREADS;
        const int base_idx = vec_idx * VECTOR_WIDTH;

        if (base_idx + VECTOR_WIDTH - 1 < valid_items) {
            VectorT result;
            T* scalars = reinterpret_cast<T*>(&result);
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; i++) {
                scalars[i] = items[v * VECTOR_WIDTH + i];
            }
            vec_ptr[vec_idx] = result;
        } else {
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; i++) {
                const int idx = base_idx + i;
                if (idx < valid_items) {
                    block_ptr[idx] = items[v * VECTOR_WIDTH + i];
                }
            }
        }
    }
}

template<
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void StripedToBlocked(
    T (&items)[ITEMS_PER_THREAD],
    T* smem)
{
    constexpr int VECTOR_WIDTH = sizeof(VectorT) / sizeof(T);
    constexpr int VECTORS_PER_THREAD = ITEMS_PER_THREAD / VECTOR_WIDTH;

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        const int smem_idx = (threadIdx.x + v * BLOCK_THREADS) * VECTOR_WIDTH;
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; i++) {
            smem[smem_idx + i] = items[v * VECTOR_WIDTH + i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = smem[threadIdx.x * ITEMS_PER_THREAD + i];
    }
}

template < 
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockedToStriped(
    T (&items)[ITEMS_PER_THREAD],
    T* smem)
{
    constexpr int VECTOR_WIDTH = sizeof(VectorT) / sizeof(T);
    constexpr int VECTORS_PER_THREAD = ITEMS_PER_THREAD / VECTOR_WIDTH;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        smem[threadIdx.x * ITEMS_PER_THREAD + i] = items[i];
    }
    __syncthreads();

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        const int smem_idx = (threadIdx.x + v * BLOCK_THREADS) * VECTOR_WIDTH;
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; i++) {
            items[v * VECTOR_WIDTH + i] = smem[smem_idx + i];
        }
    }
}

template<
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void LoadStripedToBlocked(
    const T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    T* smem,
    int valid_items)
{
    LoadStripedVectorized<T, VectorT, BLOCK_THREADS, ITEMS_PER_THREAD>(
        block_ptr, items, valid_items);
    StripedToBlocked<T, VectorT, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, smem);
}

template<
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void StoreBlockedToStriped(
    T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    T* smem,
    int valid_items)
{
    BlockedToStriped<T, VectorT, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, smem);
    StoreStripedVectorized<T, VectorT, BLOCK_THREADS, ITEMS_PER_THREAD>(
        block_ptr, items, valid_items);
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
