#include <stdio.h>
#include <time.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

// #define M 62
#define N 2
#define TPB 32
#define K 3
#define K 4
#define K 5
#define K 6
#define K 7
#define K 8
#define K 9
#define K 10
#define K 11
#define K 12
#define K 13
#define K 14
#define K 15
#define K 16
#define K 17
#define K 18
#define K 19
#define K 20
#define K 7
#define MAX_ITER 100

__device__ __host__ double distance(const float* point, const float* cluster, size_t pointNumber, size_t clusterNumber)
{
    double sum = 0;
    for(auto i = 0; i < N; i++) {
        sum += pow(point[pointNumber * N + i] - cluster[clusterNumber * N + i], 2);
    }
    return sum;
}

__global__ void clearCounts(int* count) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < K) {
        count[idx] = 0;
    }
}

__global__ void KMeansAssignCluster(const float* X, const float* C, unsigned* assignment, int* membershipCount, size_t M) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M) {
        double minDist = INFINITY;
        size_t minIdx = 0;
        for(int i = 0; i < K; i++) {
            double dist = distance(X, C, idx, i);
            if(dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
        // clearCounts(membershipCount);
        __syncthreads();
        assignment[idx] = minIdx;
        atomicAdd(&(membershipCount[minIdx]), 1u);
    }
}

__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids, int* cluster_size, size_t M)
{
    //get idx for this datapoint
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= M) return;

    //find the closest centroid to this datapoint
    float min_dist = INFINITY;
    int closest_centroid = 0;

    for(int c = 0; c<K;c++)
    {
        float dist = distance(d_datapoints,d_centroids, idx, c);

        if(dist < min_dist)
        {
            min_dist = dist;
            closest_centroid=c;
        }
    }
    //assign closest cluster id for this datapoint/thread
    d_clust_assn[idx]=closest_centroid;
    atomicAdd(&cluster_size[closest_centroid], 1);
}

__global__ void KMeansNormalize(float* C, const int* clusterSize) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < K) {
        for(int p = 0; p < N; p++) {
            C[idx * N + p] = C[idx * N + p] / (float) clusterSize[idx];
        }
    }
}

__global__ void KMeansCentroidUpdate(const float* X, float* C, const unsigned* assignment, const int* clusterSize, unsigned M) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M) {

        auto sharedIdx = threadIdx.x;
        // step 1: assign thread data to a shared memory location: then thread 0 will gather all the information
        extern __shared__ float sharedX[];
        __shared__ unsigned sharedAssignments[TPB];
        sharedAssignments[sharedIdx] = assignment[idx];
        for(int p = 0; p < N; p++) {
            sharedX[sharedIdx * N + p] = X[idx * N + p];
        }
        __syncthreads();

        if(sharedIdx == 0) {
            // we are inside the thread 0 of each block
            // gather the shared data into a single array and load it under the global array,
            // assumed to be zeroed beforehand

            // (i,j) -> i * N + j element is the partial sum of the block of
            // the j-th element of the i-th cluster
            float blockPartialSums[K*N];
            for(int i = 0; i < K*N; i++) blockPartialSums[i] = 0;
            //printf("\n%f sx", sharedX[TPB * N - 1]);
            //printf("\n%f bs", blockPartialSums[K*N - 1]);
            // for(int index = 0; index < K*N; index++) { //blockPartialSums[index] = 0 printf("\n%d", index); }
            for(int t = 0; t < blockDim.x; t++) {
                unsigned i = sharedAssignments[t];
                for(int j = 0; j < N; j++) {
                    blockPartialSums[i * N + j] += sharedX[t * N + j];
                }
            }
            // each block now will add its values to the main one
            for(int i = 0; i < K; i++) {
                for(int j = 0; j < N; j++) {
                    atomicAdd(&C[i * N + j], blockPartialSums[i * N + j]);
                }
            }
        }
        /*
        __syncthreads();
        // now the first K threads will divide the clusters sums by their member count
        if(idx < K) {
            for(int p = 0; p < N; p++) {
                C[idx * N + p] = C[idx * N + p] / (float) clusterSize[idx];
            }
        }
*/
    }

}


__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes,
                                     size_t M)
{
    //get idx of thread at grid level
    auto idx = blockIdx.x*blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= M) return;

    //get idx of thread at the block level
    auto s_idx = threadIdx.x;

    //put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
    extern __shared__ float s_datapoints[]; // TPB * N
    __shared__ int s_clust_assn[TPB];
    s_clust_assn[s_idx] = d_clust_assn[idx];
    for(auto p = 0; p < N; p++) {
        s_datapoints[s_idx * N + p] = d_datapoints[idx * N + p];
    }

    __syncthreads();

    //it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
    if(s_idx==0)
    {

        float b_clust_datapoint_sums[K*N];
        int b_clust_sizes[K];

        for(int i = 0; i < K; i++) {
            b_clust_sizes[i] = 0;
        }
        // iterate over the blocks
        for(int j=0; j< blockDim.x; ++j)
        {
            int clust_id = s_clust_assn[j];
            for(auto p = 0; p < N; p++) {
                b_clust_datapoint_sums[clust_id * N + p] += s_datapoints[j * N + p];
            }
            b_clust_sizes[clust_id] += 1;
        }
        //Now we add the sums to the global centroids and add the counts to the global counts.
        for(int z=0; z < K; ++z)
        {
            for(auto p = 0; p < N; p++) {
                atomicAdd(&d_centroids[z * N + p],b_clust_datapoint_sums[z*N + p]);
            }
            // atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
        }

    }

    __syncthreads();

    for(int i = 0; i < K; i++) {
        printf("BCluster size %d:%d\n", i, d_clust_sizes[i]);

    }
    //currently centroids are just sums, so divide by size to get actual centroids
    if(idx < K){
        for(auto p = 0; p < N; p ++) {
            d_centroids[idx * N + p] = d_clust_sizes[idx] > 0 ? d_centroids[idx * N + p] / (float) d_clust_sizes[idx] : 0;
        }
    }

}

void printVec(float* vec, size_t idx) {
    printf("\nCluster #%u: [", idx);
    for(int i =0; i < N; ++i){
        printf("%f\t", vec[idx * N + i]);
    }
    printf("]");
}
unsigned count( char* path) {
    unsigned count = 0;
    for(auto & entry: std::filesystem::directory_iterator(path)) {
        // std::cout << "Reading - " << entry.path() << " -" << std::endl;
        std::ifstream file(entry.path());
        std::string line;
        while(std::getline(file, line)) count++;
        file.close();
    }
    return count;
}

float *readFiles( char *path, unsigned *M) {
// step 1 count how many elements are present in the files
    *M = count(path);
    printf("Total number of vectors is %u\n", *M);
    auto data = (float *) malloc(sizeof(float) * (*M * N));
    size_t idx = 0;
    size_t pos = 0;
    size_t prev = 0;
    for (auto  &entry: std::filesystem::directory_iterator(path)) {
        std::cout << "Reading - " << entry.path() << " -" << std::endl;
        std::ifstream file(entry.path());
        std::string line;
        while (std::getline(file, line)) {
            size_t p = 0;
            while (pos != std::string::npos) {
                pos = line.find(',', prev + 1);
                data[idx * N + p++] = std::stof(line.substr(prev == 0 ? prev : prev + 1,
                                                            pos != std::string::npos ? pos - prev : std::string::npos));
                prev = pos;
            }
            idx++;
            prev = 0;
            pos = 0;
        }
        file.close();
    }
    return data;
}
/*

int main() {
    auto M = (size_t*) malloc(sizeof(size_t));
    //allocate memory on the device for the data points
    float *d_datapoints = 0;
    //allocate memory on the device for the cluster assignments
    int *d_clust_assn = 0;
    //allocate memory on the device for the cluster centroids
    float *d_centroids = 0;
    //allocate memory on the device for the cluster sizes
    int *d_clust_sizes=0;

    float *h_datapoints = readFiles("files/", M);

    cudaMalloc(&d_datapoints,  *M*N*sizeof(float));
    cudaMalloc(&d_clust_assn, *M*sizeof(int));
    cudaMalloc(&d_centroids, K*N*sizeof(float));
    cudaMalloc(&d_clust_sizes,K*sizeof(float));


    float *h_centroids = (float*)malloc(K*N*sizeof(float));

    int *h_clust_sizes = (int*)malloc(K*sizeof(int));

    srand(time(0));

    //initialize centroids
    for(int c=0;c<K;++c)
    {
        auto dataIdx = (size_t) rand() % *M;
        for(auto p = 0; p < N; p++) {
            h_centroids[c * N + p] = h_datapoints[dataIdx * N + p];
        }
        h_clust_sizes[c]=0;
    }


    cudaMemcpy(d_centroids,h_centroids,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints,h_datapoints,*M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

    int cur_iter = 1;
    int *h_clust_assign = (int*) malloc(sizeof (int) * *M);
    while(cur_iter < MAX_ITER)
    {
        cudaMemset(d_clust_sizes,0,K*sizeof(int));
        //call cluster assignment kernel
        kMeansClusterAssignment<<<(*M+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids, d_clust_sizes, *M);
        //copy new centroids back to host
        cudaMemcpy(h_centroids,d_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clust_assign,d_clust_assn,*M*sizeof(int),cudaMemcpyDeviceToHost);



        for(int i =0; i < K; ++i){
            printf("\nIteration %d: centroid %d:",cur_iter,i);
            printVec(h_centroids, i);
        }

        //reset centroids and cluster sizes (will be updated in the next kernel)
        cudaMemset(d_centroids,0.0,K*sizeof(float));
        //call centroid update kernel
        kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB, (TPB * N) * sizeof(float)>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes,
                                                                               *M);

        cur_iter+=1;
    }

    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_clust_sizes);

    free(h_centroids);
    free(h_datapoints);
    free(h_clust_sizes);

    return 0;
}
*/

int main() {
    auto M = (unsigned *) malloc(sizeof(unsigned));
    //allocate memory on the device for the data points
    float *deviceData = nullptr;
    //allocate memory on the device for the cluster assignments
    unsigned *deviceClusterAssignment = nullptr;
    //allocate memory on the device for the cluster centroids
    float *deviceClusters = nullptr;
    //allocate memory on the device for the cluster sizes
    int *deviceMembershipCount = nullptr;

    float *hostData = readFiles("files/", M);

    cudaMalloc(&deviceData, *M * N * sizeof(float));
    cudaMalloc(&deviceClusterAssignment, *M * sizeof(int));

    cudaMalloc(&deviceClusters, K * N * sizeof(float));
    cudaMalloc(&deviceMembershipCount, K * sizeof(float));


    auto *hostClusters = (float *) malloc(K * N * sizeof(float));

    auto *hostMembershipCount = (int *) malloc(K * sizeof(int));

    srand(time(0));
    //initialize centroids
    std::vector<int> used = std::vector<int>();
    for (int c = 0; c < K; c++) {


        int dataIdx = (int) rand() % (*M);
        while (std::find(used.begin(), used.end(), dataIdx) != used.end()) {
            dataIdx++;
        }
        for (auto p = 0; p < N; p++) {
            hostClusters[c * N + p] = hostData[dataIdx * N + p];
        }
        used.emplace_back(dataIdx);
        hostMembershipCount[c] = 0;
    }
    /*
//initialize centroids
    int dataIdx = (int) rand() % (*M);
    for(auto p = 0; p < N; p++) {
        hostClusters[p] = hostData[dataIdx * N + p];
    }
    for(int c=1;c<K;c++)
    {
        int maxId = 0;
        for(int i = 0; i < *M; i++) {
            double maxDist = -INFINITY;
            bool newPoint = true;
            double distance = 0;
            for(int j = 0; j < i; j++) {
                if(hostClusters[j * N] == hostData[i * N]) newPoint = false;
                for(int p = 0; p < N; p++) {
                    distance += pow(hostData[i*N + p] - hostClusters[j*N+p], 2);
                }
            if(newPoint && distance>maxDist) {
                maxDist = distance;
                maxId = j;
            }

            }
        }
        for(int p = 0; p < N; p++) {
            hostClusters[c*N+p] = hostData[maxId * N + p];
        }
        hostMembershipCount[c]=0;
    }
*/

    cudaMemcpy(deviceClusters, hostClusters, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData, hostData, *M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusterAssignment, hostMembershipCount, K * sizeof(int), cudaMemcpyHostToDevice);

    int cur_iter = 0;
    auto *hostClusterAssignment = (int *) malloc(sizeof(int) * *M);
    int width = 40000;
    for(int ms = 1; ms <= 12; ms++) {
    auto begin = std::chrono::high_resolution_clock::now();
        int fakeM = ms * width;
        for(unsigned it = 0; it < MAX_ITER; it++) {
            clearCounts<<<(fakeM + TPB - 1) / TPB, TPB>>>(deviceMembershipCount);
            cudaDeviceSynchronize();
            //call cluster assignment kernel
            KMeansAssignCluster<<<(fakeM + TPB - 1) / TPB, TPB>>>(deviceData, deviceClusters,
                                                               deviceClusterAssignment, deviceMembershipCount,
                                                                  fakeM);
            cudaDeviceSynchronize();
            //copy new centroids back to host
            cudaMemcpy(hostClusters, deviceClusters, N * K * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < K; ++i) {
                printVec(hostClusters, i);
            }

            cudaMemset(deviceClusters, 0, K * N * sizeof(float));
            KMeansCentroidUpdate<<<(fakeM + TPB - 1) / TPB, TPB,
            TPB * N * sizeof(float)>>>(deviceData, deviceClusters,
                                       deviceClusterAssignment,deviceMembershipCount, fakeM);
            cudaDeviceSynchronize();
            KMeansNormalize<<<K, 1>>>(deviceClusters, deviceMembershipCount);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        auto totalTime = elapsed.count() * 1e-9f;
        printf("Configuration M=%d, K=7 Time measured: %.3f seconds, %.5f per iteration.\n", fakeM, totalTime, totalTime / MAX_ITER);
    }

    auto begin = std::chrono::high_resolution_clock::now();
    for(unsigned it = 0; it < MAX_ITER; it++) {
        clearCounts<<<(*M + TPB - 1) / TPB, TPB>>>(deviceMembershipCount);
        cudaDeviceSynchronize();
        //call cluster assignment kernel
        KMeansAssignCluster<<<(*M + TPB - 1) / TPB, TPB>>>(deviceData, deviceClusters,
                                                           deviceClusterAssignment, deviceMembershipCount,
                                                           *M);
        cudaDeviceSynchronize();
        //copy new centroids back to host
        cudaMemcpy(hostClusters, deviceClusters, N * K * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < K; ++i) {
            printVec(hostClusters, i);
        }

        cudaMemset(deviceClusters, 0, K * N * sizeof(float));
        KMeansCentroidUpdate<<<(*M + TPB - 1) / TPB, TPB,
        TPB * N * sizeof(float)>>>(deviceData, deviceClusters,
                                   deviceClusterAssignment,deviceMembershipCount, *M);
        cudaDeviceSynchronize();
        KMeansNormalize<<<K, 1>>>(deviceClusters, deviceMembershipCount);
        cudaDeviceSynchronize();
    }

    // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto totalTime = elapsed.count() * 1e-9f;
    printf("Configuration M=%d, K=7 Time measured: %.3f seconds, %.5f per iteration.\n", *M, totalTime, totalTime / MAX_ITER);

    cudaFree(deviceClusterAssignment);
    cudaFree(deviceClusters);
    cudaFree(deviceMembershipCount);
    cudaFree(deviceData);

    free(hostClusters);
    free(hostMembershipCount);
    free(hostData);
    free(hostClusterAssignment);

    return 0;
}
