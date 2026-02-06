#include <iostream>
#include <vector>

using namespace std;

template<int BITS>
void RadixSortK(vector<uint32_t>& keys, vector<uint32_t>& output)
{
    constexpr uint32_t K = 1 << BITS; // Number of Bins/Buckets
    vector<uint32_t> histogram(K);

    for (size_t shift = 0; shift < sizeof(uint32_t)*8; shift += BITS) {

        // Build histogram
        fill(histogram.begin(), histogram.end(), 0);
        for (size_t i = 0; i < keys.size(); ++i) {
            uint32_t d = (keys[i] >> shift) & ((1 << BITS) - 1);
            histogram[d]++;
        }

        // Exclusive prefix sum of histogram to get offset
        uint32_t sum = 0;
        for (size_t bin = 0; bin < histogram.size(); ++bin) {
            uint32_t count = histogram[bin];
            histogram[bin] = sum;
            sum += count;
        }

        // Scatter (stable)
        for (size_t i = 0; i < keys.size(); ++i) {
            uint32_t d = (keys[i] >> shift) & ((1 << BITS) - 1);
            output[histogram[d]++] = keys[i];
        }

        swap(keys, output);

    }
}

void PrintVector(const vector<uint32_t>& vec)
{
    for (const auto elem : vec) {
        cout << elem << ", ";
    }
    cout << "\n";
}

int main(int argc, char** argv)
{
    constexpr size_t NUM_ELEMS = 32;

    vector<uint32_t> keys(NUM_ELEMS);
    vector<uint32_t> output(NUM_ELEMS);

    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        keys[i] = rand() % 16;
    }

    PrintVector(keys);
    RadixSortK<16>(keys, output);
    PrintVector(output);
}