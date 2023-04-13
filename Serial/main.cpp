#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <vector>
#include <random>

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
    double r = sqrt(distance);
    return r;// really necessary?
}

size_t assignCluster(std::vector<std::vector<double>*>& clusters, std::vector<double>* point) {
    double minDistance = MAXFLOAT;
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

void clearVector(const std::vector<size_t*>& toFree) {
    for(auto el: toFree) {
        delete el;
    }
}

void clearVector(const std::vector<std::vector<double>*>& toFree) {
    for(auto el: toFree) {
        delete el;
    }
}

bool find(std::vector<double>* element, std::vector<std::vector<double>*>& target) {
    for(const auto& vec: target) {
        bool found = true;
        for(size_t i = 0; i < vec->size(); i++) {
            if((*element)[i] != (*vec)[i]) {
                found = false;
                break;
            }
        }
        if(found) return true;
    }
    return false;
}
std::vector<std::vector<double>*> init(std::vector<std::vector<double>*>& points, size_t K) {
    std::vector<std::vector<double>*> clusters;
    std::random_device r;
    auto e = std::default_random_engine(r());
    std::uniform_int_distribution<size_t> points_dist(0, points.size() - 1);
    // first cluster sampled randomly
    auto cluster = new std::vector<double>(*points[points_dist(e)]);
    clusters.emplace_back(cluster);
    for(size_t i = 1; i < K; i++) {
        size_t maxIndex = 0;
        double maxDist = 0;
        for(size_t j = 0; j < points.size(); j++) {
            double d = 0;
            for(const auto& c: clusters) {
                d += distance(c, points[j]);
            }
            if(d >= maxDist && !find(points[j], clusters)) {
                // check if not already present
                maxDist = d;
                maxIndex = j;
            }
        }
        clusters.emplace_back(new std::vector<double>(*points[maxIndex]));
    }
    return clusters;
}

std::vector<double>* centroid(std::vector<std::vector<double>*>& points, std::vector<double>* mean) {
    size_t index = 0;
    double minDist = MAXFLOAT;
    for(size_t i = 0; i < points.size(); i++) {
        double d = distance(mean, points[i]);
        if(d < minDist) {
           index = i;
           minDist = d;
        }
    }
    return new std::vector<double>(*points[index]);
}


std::vector<std::vector<double>*> KMeans(std::vector<std::vector<double>*>& points, std::vector<size_t*>& assignements,
                                         size_t K = 5) {
    auto n = points.size();
    // assert(n > K);
    // initialize clusters centroids
    auto clusters = std::vector<std::vector<double>*>(K);
    // todo kmeans++ init
    for(auto i = 0; i < K; i++) {
        clusters[i] = new std::vector<double>(*points[i]);
    }
    // auto clusters = init(points, K);
    // until convergence
    bool converged;
    size_t numIterations = 0;
    do{
        converged = true;
        // assign to closest centroid
        for(size_t i = 0; i < points.size(); i++) {
            auto closest = assignCluster(clusters, points[i]);
            if(closest != (*assignements[i])) {
                *(assignements[i]) = closest;
                converged = false;
            }
        }
        // compute new mean
        for(size_t i = 0; i < K; i++) {
            std::vector<std::vector<double>*> membership;
            for(size_t j = 0; j < n; j++) {
                if((*assignements[j]) == i) {
                    membership.emplace_back(points[j]);
                }
            }
            delete clusters[i];
            clusters[i] = centroid(points, mean(membership));
        }
        numIterations++;
    } while(!converged);
    std::cout << "Convergence in " << numIterations << " steps" << std::endl;
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
    size_t K = 10;
    auto data = readFiles2("./files/", K);
    auto assignements = std::vector<size_t*>();
    for(int i = 0; i < data.size(); i++) assignements.emplace_back(new size_t(0));
    auto r = KMeans(data, assignements, K);
    printVectors(r);
    return 0;
}
