#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#define MAXITER 2


std::vector<double>* mean(std::vector<std::vector<double>*>& points) {
    // assert(!points.empty());
    auto p = points[0]->size();
    auto result = new std::vector<double>(p, 0);
    auto n = points.size();
    for(const auto& point: points) {
        for(auto i = 0; i < p; i++) {
            (*result)[i] += (*point)[i];
        }
    }
    for(auto i = 0; i < p; i++) {
        (*result)[i] /= double(n);
    }
    return result;
}

double distance(std::vector<double>* x, std::vector<double>* y) {
    auto p = x->size();
    double distance = 0;
    for(auto i = 0; i < p; i++) {
        distance += pow((*x)[i] - (*y)[i], 2);
    }
    return distance >= 0 ? distance : INFINITY;
}

size_t assignCluster(std::vector<std::vector<double>*>& clusters, std::vector<double>* point) {
    double minDistance = INFINITY;
    size_t closest = 0;
    for(auto i = 0; i < clusters.size(); i++) {
        double d = distance(point, clusters[i]);
        if(d < minDistance) {
            minDistance = d;
            closest = i;
        }
    }
    return closest;
}

std::vector<std::vector<double>*> KMeans(std::vector<std::vector<double>*>& points,
                                         size_t K = 5, unsigned maxSamples = INT32_MAX) {
    auto n = points.size();
    n = n < maxSamples ? n : maxSamples;
    // assert(n > K);
    // initialize clusters centroids
    auto clusters = std::vector<std::vector<double>*>(K);
    auto used = std::vector<size_t>();
    for(auto i = 0; i < K; i++) {
        size_t id = rand() % n;
        while(std::find(used.begin(), used.end(), id) != used.end()) id++;
        used.emplace_back(id);
        clusters[i] = new std::vector<double>(*points[i]);
    }
    // auto clusters = init(points, K);
    // until convergence


    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();

    for(int numIterations = 0; numIterations < MAXITER; numIterations++) {
        // assign to closest centroid
        auto assignments = std::vector<std::vector<size_t>>(K);
        for(size_t i = 0; i < n; i++) {
            auto closest = assignCluster(clusters, points[i]);
            assignments[closest].emplace_back(i);
        }
        for(size_t k = 0; k < K; k++) {
            for(size_t p = 0; p < clusters[k]->size(); p++) (*clusters[k])[p] = 0;
            auto s = (float) assignments[k].size();
            for(auto element: assignments[k]) {
                for(size_t p = 0; p < clusters[k]->size(); p++) (*clusters[k])[p] += (*points[element])[p] / s;
            }
        }

    }

    // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto totalTime = elapsed.count() * 1e-9f;
    printf("Time measured: %.3f seconds, %.5f per iteration.\n", totalTime, totalTime / MAXITER);

    return clusters;
}

void printVectors(std::vector<std::vector<double>*>& vectors) {
    for(int i = 0; i < vectors.size(); i++) {
        std::cout << "Vector #" << i + 1 << ": \t";
        for(const auto& val: *vectors[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}



std::vector<std::vector<double>*> readFiles2(const std::string& path, size_t K) {
    double number;
    auto result = std::vector<std::vector<double>*>();
    size_t pos = 0, prev = 0;
    for(auto const& entry: std::filesystem::directory_iterator(path)) {
        std::cout << "Reading - " << entry.path() << " -" << std::endl;
        std::ifstream file(entry.path());
        std::string line;
        while(std::getline(file, line)) {
            auto point = new std::vector<double>();
            while(pos != std::string::npos) {
                pos = line.find(',', prev + 1);
                number = std::stod(line.substr(prev == 0 ? prev : prev + 1, pos != std::string::npos ? pos - prev : std::string::npos));
                point->emplace_back(number);
                prev = pos;
            }
            prev = 0; pos = 0;
            result.emplace_back(point);
        }
    }
    return result;
}



int main() {
    size_t K = 7;
    auto data = readFiles2("./files/", K);
    unsigned width = 40000;
    for(unsigned M = 1; M <= 12; M++) {
        unsigned nSamples = M * width;
        printf("\nTest with %u samples, K=%lu\n", nSamples, K);
        auto r = KMeans(data, K);
        printVectors(r);
    }
    printf("\nTest with all the samples (4853516), K=%lu\n", K);
    auto r = KMeans(data, K);
    printVectors(r);


    // test over K
    for(int i = 2; i <= 20; i++) {
        printf("\nTest with all the samples, K=%lu\n", i);
        r = KMeans(data, K);
        printVectors(r);
    }



    return 0;
}
