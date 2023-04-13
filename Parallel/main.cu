#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include <cmath>
 __global__ void update_centroids(double** X, double** C, double** memberships, const size_t* counter, size_t M,
                                 size_t N, size_t K, size_t width) {
    /**
     * @param X: matrix of size M * N
     * @param C: matrix of size K * N
     * @param memberships: matrix of size M * K; each row should have only one non-zero element with value 1
     * @param counter: vector of size K; store the number of element per cluster
     * @details: update the centroids stored in C using the membership matrix
     *
     */
     printf("uc\n");
     auto w = threadIdx.x + blockIdx.x * blockDim.x * width;
     auto h = threadIdx.y + blockIdx.y * blockDim.y;
     for(auto offset = 0; offset < width; offset++) {
         auto w1 = w + offset;
          if(w1 < K && h < N) {
              double accumulator = 0;
              for(size_t i = 0; i < M; i++) {
                  accumulator += X[i][h] * memberships[i][w1];
              }
              C[w1][h] = accumulator / (double) counter[w1];
          }
     }
}

__global__ void compute_distance(double** X, double** C, double** distances, size_t M, size_t N, size_t K,
                                 size_t width) {
    /**
     * @param X: matrix of size M * N
     * @param C: matrix of size K * N
     * @param distances: matrix of size M * K; ij-th element is the distance betweeen i-th sample from j-th cluster
     * @details: update the centroids stored in C using the membership matrix
     *
     */

    printf("cd\n");
    auto w = threadIdx.x + blockIdx.x * blockDim.x;
    auto h = threadIdx.y + blockIdx.y * blockDim.y;
    for(auto offset = 0; offset < width; offset++) {
        auto w1 = w + offset;
        if (w1 < M && h < K) {
            double distance = 0;
            for (size_t p = 0; p < N; p++) {
                distance += pow(X[w1][p] - C[h][p], 2);
            }
            distances[w1][h] = sqrt(distance);
        }
    }
}

__global__ void assign_closest_cluster(double** distances, size_t M, size_t K, size_t* counter, size_t width) {
    auto w = threadIdx.x + blockIdx.x * blockDim.x;
    printf("CLOSEST\n");
    for(auto offset = 0; offset < width; offset++) {
        auto w1 = w + offset;
        if (w1 < M) {
            // step 1 identify min index
            double minDist = MAXFLOAT;
            size_t minIdx = 0;
            for (size_t j = 0; j < K; j++) {
                if (distances[w1][j] < minDist) {
                    minDist = distances[w1][j];
                    minIdx = j;
                } else distances[w1][j] = 0; // surely not the closest one
            }
            for (size_t j = 0; j < minIdx; j++) distances[w1][j] = 0;
            distances[w1][minIdx] = 1;
            counter[minIdx] = 0;
        }
    }
}

void initMatrix(double*** matrix, size_t M, size_t N) {
    *matrix = (double**) malloc(sizeof(double*) * M);
    for(size_t i = 0; i < M; i++) (*matrix)[i] = (double *) malloc(sizeof (double ) * N);
}
size_t count(const char* path) {
    size_t  count = 0;
    for(auto const& entry: std::filesystem::directory_iterator(path)) {
        // std::cout << "Reading - " << entry.path() << " -" << std::endl;
        std::ifstream file(entry.path());
        std::string line;
        while(std::getline(file, line)) count++;
        file.close();
    }
    return count;
}

double** readFiles(const char* path, size_t* M, size_t N) {
    // step 1 count how many elements are present in the files
    *M = count(path);
    printf("Total number of vectors is %lu", M);
    auto data = (double**) malloc(sizeof (double *) * (*M));
    size_t idx = 0;
    size_t pos = 0;
    size_t prev = 0;
    for(auto const& entry: std::filesystem::directory_iterator(path)) {
        std::cout << "Reading - " << entry.path() << " -" << std::endl;
        std::ifstream file(entry.path());
        std::string line;
        while(std::getline(file, line)) {
            data[idx] = (double *) malloc(sizeof(double) * N);
            size_t p = 0;
            while(pos != std::string::npos) {
                pos = line.find(',', prev + 1);
                data[idx][p++] = std::stod(line.substr(prev == 0 ? prev : prev + 1, pos != std::string::npos ? pos - prev : std::string::npos));
                prev = pos;
            }
            idx++;
            prev = 0; pos = 0;
        }
        file.close();
    }
    return data;
}

int main() {
    size_t K = 2;
    size_t N = 2;
    size_t M = 0;
    auto counter = (size_t*) malloc(sizeof(size_t) * K);
    // data read ok!
    auto data = readFiles("./files/", &M, N);
    auto clusters = (double **) malloc(sizeof(double*) * 2);
    double** distanceM;
    clusters[0] = (double *) malloc(sizeof(double) * 2);
    clusters[0][0] = 100;
    clusters[0][1] = 100;
    clusters[1] = (double *) malloc(sizeof(double) * 2);
    clusters[1][0] = -100;
    clusters[1][1] = -100;
    initMatrix(&distanceM, M, K);
    // 2048 = w * k
    size_t width = 2048 / K;
    for(int i = 0; i < 2; i++) {
        printf("1\n");
        compute_distance<<<M / width,K>>>(data, clusters, distanceM, M, N, K, width);
        cudaDeviceReset();
        printf("2\n");
        assign_closest_cluster<<<M / width, 1>>>(distanceM, M, K, counter, width);
        cudaDeviceSynchronize();
        printf("3\n");
        update_centroids<<<M / width, K>>>(data, clusters, distanceM, counter, M, N, K, width);
    }


    cudaDeviceReset();
    //getchar();
    /*distances(distanceM, data, clusters, M, N, K);
    closestClusterMatrix(distanceM, M, K);
    printf("%f %f\n", data[0][0], data[0][1]);
    printf("%f %f\n", clusters[0][0], clusters[0][1]);
    printf("%f %f\n", clusters[1][0], clusters[1][1]);

    printf("%f %f", distanceM[0][0], distanceM[0][1]);
*/
    return 0;
}
