ঠিক আছে! তোমার আগের দুটো প্রশ্নের কোডে আমি **প্রশ্নগুলো সরাসরি কমেন্ট হিসেবে লিখে দিচ্ছি** যাতে কোডের মাথাতেই বোঝা যায় কী করা হচ্ছে।  

নিচে দুটো প্রোগ্রামের **আপডেটেড ভার্সন** দিলাম, যেখানে প্রথম লাইনগুলোতে **প্রশ্নটা কমেন্ট হিসেবে আছে**।

### ১. MPI Word Count Program

```cpp
/*
প্রশ্ন:
Write a program using MPI to count the words in a file and sort it in descending order of frequency of words 
i.e. highest occurring word must come first and the least occurring word must come last.
Input:
No. of processes, (Text input from file)
Output:
Total searching time, top 10 occurrences of string
*/

// MPI Word Count Program with Frequency Sort (Descending)
// Compile: mpicxx wordcount_mpi.cpp -o wordcount_mpi -std=c++11
// Run on Ubuntu: mpirun -np 4 ./wordcount_mpi input.txt

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace std;

// Function to tokenize string into words (lowercase, remove punctuation)
void tokenize(const string& text, map<string, int>& freq) {
    stringstream ss(text);
    string word;
    while (ss >> word) {
        string clean_word;
        for (char c : word) {
            if (isalpha(c)) {
                clean_word += tolower(c);
            }
        }
        if (!clean_word.empty()) {
            freq[clean_word]++;
        }
    }
}

// Comparator for sorting pairs by value descending
bool compare_freq(const pair<string, int>& a, const pair<string, int>& b) {
    return a.second > b.second;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: mpirun -np N ./wordcount_mpi input.txt" << endl;
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    string full_text;
    int text_len = 0;

    double start_time = MPI_Wtime();  // Start timing

    // Root reads the file
    if (rank == 0) {
        ifstream file(filename);
        if (!file) {
            cout << "File not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        stringstream buffer;
        buffer << file.rdbuf();
        full_text = buffer.str();
        text_len = full_text.size();
    }

    // Broadcast text length
    MPI_Bcast(&text_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate chunk sizes
    int chunk_size = text_len / size;
    int remainder = text_len % size;
    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Local chunk
    string local_text(sendcounts[rank], ' ');
    MPI_Scatterv(full_text.data(), sendcounts.data(), displs.data(), MPI_CHAR,
                 &local_text[0], sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // Local frequency map
    map<string, int> local_freq;
    tokenize(local_text, local_freq);

    // Gather maps at root
    int num_entries = local_freq.size();
    vector<int> all_num_entries(size);
    MPI_Gather(&num_entries, 1, MPI_INT, all_num_entries.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    map<string, int> global_freq;
    if (rank == 0) {
        global_freq = local_freq;
    } else {
        for (auto& p : local_freq) {
            int word_len = p.first.size();
            MPI_Send(&word_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(p.first.data(), word_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&p.second, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        for (int src = 1; src < size; src++) {
            for (int j = 0; j < all_num_entries[src]; j++) {
                int word_len;
                MPI_Recv(&word_len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string word(word_len, ' ');
                MPI_Recv(&word[0], word_len, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int count;
                MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_freq[word] += count;
            }
        }

        // Sort by frequency descending
        vector<pair<string, int>> sorted_freq(global_freq.begin(), global_freq.end());
        sort(sorted_freq.begin(), sorted_freq.end(), compare_freq);

        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;

        // Output
        cout << "Total computing time: " << total_time << " seconds" << endl;
        cout << "Top 10 words:" << endl;
        for (int i = 0; i < min(10, (int)sorted_freq.size()); i++) {
            cout << sorted_freq[i].first << ": " << sorted_freq[i].second << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
```

### ২. CUDA Phonebook Search Program

```cuda
/*
প্রশ্ন:
Consider a phonebook is given as text file and a person name P. 
Write a program using CUDA to search for a person P (given as a string) in the phonebook. 
The program will return the names of all persons containing the substring P from the phonebook, 
along with their corresponding contact numbers.
Input:
No. of CPU core, (phonebook from file), person name P
Output:
Total searching time, matching names and contact numbers
*/

// CUDA Substring Search in Phonebook
// Compile: nvcc phonebook_cuda.cu -o phonebook_cuda
// Run: ./phonebook_cuda phonebook.txt Alice

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Kernel: each thread checks one line for substring in name
__global__ void searchKernel(const char* d_text, const size_t* d_offsets, int num_lines,
                             const char* d_pattern, int pat_len, int* d_result_indices, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;

    size_t start = d_offsets[idx];
    size_t end = d_offsets[idx + 1];

    size_t space_pos = start;
    while (space_pos < end && d_text[space_pos] != ' ') space_pos++;
    if (space_pos == end) return;

    size_t name_len = space_pos - start;

    for (size_t i = 0; i <= name_len - pat_len; ++i) {
        bool match = true;
        for (int j = 0; j < pat_len; ++j) {
            if (d_text[start + i + j] != d_pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            int pos = atomicAdd(d_count, 1);
            d_result_indices[pos] = idx;
            return;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./phonebook_cuda phonebook.txt P" << endl;
        return 1;
    }

    string filename = argv[1];
    string pattern = argv[2];
    int pat_len = pattern.size();

    ifstream file(filename);
    if (!file) {
        cout << "File not found!" << endl;
        return 1;
    }

    vector<string> lines;
    string line;
    while (getline(file, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    int num_lines = lines.size();
    if (num_lines == 0) return 1;

    string full_text;
    vector<size_t> offsets(num_lines + 1, 0);
    for (int i = 0; i < num_lines; ++i) {
        full_text += lines[i] + '\n';
        offsets[i + 1] = offsets[i] + lines[i].size() + 1;
    }
    size_t text_len = full_text.size();

    char *d_text, *d_pattern;
    size_t *d_offsets;
    int *d_result_indices, *d_count;

    cudaMalloc(&d_text, text_len);
    cudaMalloc(&d_offsets, (num_lines + 1) * sizeof(size_t));
    cudaMalloc(&d_pattern, pat_len);
    cudaMalloc(&d_result_indices, num_lines * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_text, full_text.data(), text_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (num_lines + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.data(), pat_len, cudaMemcpyHostToDevice);

    int h_count = 0;
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads_per_block = 256;
    int blocks = (num_lines + threads_per_block - 1) / threads_per_block;
    searchKernel<<<blocks, threads_per_block>>>(d_text, d_offsets, num_lines, d_pattern, pat_len, d_result_indices, d_count);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    vector<int> h_result_indices(h_count);
    if (h_count > 0) {
        cudaMemcpy(h_result_indices.data(), d_result_indices, h_count * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cout << "Total searching time: " << ms / 1000.0 << " seconds" << endl;
    cout << "Matching names and contacts:" << endl;
    for (int idx : h_result_indices) {
        cout << lines[idx] << endl;
    }

    cudaFree(d_text);
    cudaFree(d_offsets);
    cudaFree(d_pattern);
    cudaFree(d_result_indices);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

এখন কোডের একদম শুরুতেই প্রশ্নটা কমেন্ট আকারে আছে।  
চাইলে এগুলো কপি করে সরাসরি VS Code বা Colab-এ পেস্ট করে চালাতে পারো।  

কোনো অংশে আরও পরিবর্তন বা ব্যাখ্যা লাগলে বলো!
