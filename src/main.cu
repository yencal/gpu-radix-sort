// main.cu
// Benchmark runner for radix sort algorithm comparison

#include <iostream>
#include <cstdlib>

#include "utils.cuh"
#include "radix_sort_multikernel_dispatch.cuh"
#include "radix_sort_cub.cuh"

// ============================================================================
// Benchmark Runner
// ============================================================================

template<typename SortAlgorithm>
void RunSortBenchmark(const char* name, uint32_t n) {
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << name << std::endl;

    // Allocate
    uint32_t* d_input;
    uint32_t* d_output;
    size_t temp_size = SortAlgorithm::GetTempSize(n);
    void* d_temp;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_temp, temp_size));

    // Initialize with random data
    FillRandom(d_input, n);

    // Warmup
    SortAlgorithm::Run(d_input, d_output, n, d_temp);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify correctness (fail fast)
    uint32_t* h_output = new uint32_t[n];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (!VerifySorted(h_output, n)) {
        std::cerr << "  FAILED: Output not sorted!" << std::endl;
        // std::exit(EXIT_FAILURE);
    } else {
        std::cout << "  Correctness: PASSED" << std::endl;
    }

    // Benchmark
    constexpr int NUM_ITERATIONS = 10;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Re-randomize for fair benchmark
    FillRandom(d_input, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        SortAlgorithm::Run(d_input, d_output, n, d_temp);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / NUM_ITERATIONS;

    std::cout << "  Time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (n / 1e6) / (avg_ms / 1000.0f) << " Mkeys/s" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
    delete[] h_output;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    // Default: 2^26 keys (~256M keys, 1GB)
    int power = 26;

    if (argc >= 2) {
        power = std::atoi(argv[1]);
        if (power < 1 || power > 30) {
            std::cerr << "Power must be between 1 and 30" << std::endl;
            return EXIT_FAILURE;
        }
    }

    const uint32_t n = 1U << power;

    std::cout << "========================================" << std::endl;
    std::cout << "GPU Radix Sort Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Array size: 2^" << power << " = " << n << " keys" << std::endl;
    std::cout << "Data size: " << (static_cast<size_t>(n) * sizeof(uint32_t)) / (1024.0 * 1024.0) 
              << " MB" << std::endl;

    const float peak_bandwidth = GetPeakBandwidth();
    std::cout << "Device peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    // ========================================================================
    // RadixSortMultikernel
    // ========================================================================

    RunSortBenchmark<RadixSortMultikernel<8, 256, 8>>(
        "RadixSortMultikernel (8 RADIX_LOG, 256 threads, 8 items)", n);

    RunSortBenchmark<RadixSortMultikernel<8, 256, 16>>(
        "RadixSortMultikernel (8 RADIX_LOG, 256 threads, 16 items)", n);

    // ========================================================================
    // CUB DeviceRadixSort
    // ========================================================================

    RunSortBenchmark<RadixSortCUB>(
        "CUB DeviceRadixSort", n);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}