#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "Data.hpp"

class KMeans {
private:
    size_t numberOfClusters;
    std::vector<Data*> points = std::vector<Data*>();
    std::vector<Data> centroids = std::vector<Data>();
    void fit();
    static double distance(Data& x, Data& y);
    size_t closestCluster(Data& x);
    void clearClusters();
    void KMeansPlusPlusInit();
    int getMaxSeparatedCentroid();
public:
    KMeans(std::vector<Data*> data, size_t K);
    std::vector<Data> computeClusterCentroids();
    std::vector<Data*> getClusterAssignements();
    ~KMeans();
};
