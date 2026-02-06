#pragma once

template< 
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void LoadStripedVectorized(
    const T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    int valid_items)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        items[i] = (idx < valid_items) ? block_ptr[idx] : T{};
    }
}

template< 
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void StoreStripedVectorized(
    T* block_ptr,
    T (&items)[ITEMS_PER_THREAD],
    int valid_items)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx < valid_items) {
            block_ptr[idx] = items[i];
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
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        smem[threadIdx.x + i * BLOCK_THREADS] = items[i];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = smem[threadIdx.x * ITEMS_PER_THREAD + i];
    }
}

template<
    typename T,
    typename VectorT,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockedToStriped(
    T (&items)[ITEMS_PER_THREAD],
    T* smem)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        smem[threadIdx.x * ITEMS_PER_THREAD + i] = items[i];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = smem[threadIdx.x + i * BLOCK_THREADS];
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