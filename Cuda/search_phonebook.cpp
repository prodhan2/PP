
#!nvcc -arch=sm_75 search_phonebook.cu -o search_phonebook
#!time ./search_phonebook ROY 128 asc


%%writefile search_phonebook.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

using namespace std;

#define MAX_STR_LEN 50

__device__ bool check(char* str1, char* str2, int len) {
    for (int i = 0; str1[i] != '\0'; i++) {
        int j = 0;
        while (str1[i + j] != '\0' &&
               str2[j] != '\0' &&
               str1[i + j] == str2[j]) {
            j++;
        }
        if (j == len - 1)
            return true;
    }
    return false;
}

__global__ void searchPhonebook(
    char* d_names,
    int num_contacts,
    char* search_name,
    int search_len,
    int* d_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_contacts) {
        char* current_name = d_names + idx * MAX_STR_LEN;
        d_results[idx] = check(current_name, search_name, search_len) ? 1 : 0;
    }
}

struct Contact {
    string name;
    string number;
};

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cerr << "Usage: " << argv[0]
             << " <search_text> <threads> <asc|desc>\n";
        return 1;
    }

    string search_text = argv[1];
    int threads = atoi(argv[2]);
    string order = argv[3];

    string file_name = "/content/sample_data/phonebook1.txt";

    vector<string> names, numbers;
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Cannot open file\n";
        return 1;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        int pos = line.find(",");
        names.push_back(line.substr(1, pos - 2));
        numbers.push_back(line.substr(pos + 2, line.size() - pos - 3));
    }
    file.close();

    int n = names.size();

    char* h_names = (char*)malloc(n * MAX_STR_LEN);
    int* h_results = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        strncpy(h_names + i * MAX_STR_LEN, names[i].c_str(), MAX_STR_LEN - 1);
        h_names[i * MAX_STR_LEN + MAX_STR_LEN - 1] = '\0';
    }

    char *d_names, *d_search;
    int* d_results;
    int search_len = search_text.length() + 1;

    cudaMalloc(&d_names, n * MAX_STR_LEN);
    cudaMalloc(&d_results, n * sizeof(int));
    cudaMalloc(&d_search, search_len);

    cudaMemcpy(d_names, h_names, n * MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_search, search_text.c_str(), search_len, cudaMemcpyHostToDevice);

    int blocks = (n + threads - 1) / threads;
    searchPhonebook<<<blocks, threads>>>(d_names, n, d_search, search_len, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost);

    vector<Contact> results;
    for (int i = 0; i < n; i++) {
        if (h_results[i]) {
            results.push_back({names[i], numbers[i]});
        }
    }

    if (order == "asc") {
        sort(results.begin(), results.end(),
             [](const Contact& a, const Contact& b) {
                 return a.name < b.name;
             });
    } else {
        sort(results.begin(), results.end(),
             [](const Contact& a, const Contact& b) {
                 return a.name > b.name;
             });
    }

    cout << "\nSearch Results (" << order << "):\n";
    for (auto& c : results) {
        cout << c.name << " " << c.number << endl;
    }

    free(h_names);
    free(h_results);
    cudaFree(d_names);
    cudaFree(d_results);
    cudaFree(d_search);

    return 0;
}

