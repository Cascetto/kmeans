#include "KMeans.hpp"

KMeans::KMeans(std::vector<Data*> data, size_t K) : points(data), numberOfClusters(K) {}

KMeans::~KMeans()
{
}

double KMeans::distance(Data& X, Data& Y) {
    /// squared distance
    double distance = 0;
    auto x = X.getValue();
    auto y = Y.getValue();
    for(size_t i = 0; i < x.size(); i++) {
        auto delta = x[i] - y[i];
        distance += delta * delta;
    }
    return distance;
}

void KMeans::clearClusters() {
    for(auto& centroid: centroids) {
        for(size_t i = 0; i < centroid.getValue().size(); i++)
            centroid.getValue()[i] = 0;
    }
}

size_t KMeans::closestCluster(Data& x) {
    size_t closest = 0;
    double closestDistance = MAXFLOAT;
    for(auto& centroid: centroids) {
        auto distance = KMeans::distance(x, centroid);
        if(distance < closestDistance) {
            closestDistance = distance;
            closest = centroid.getCluster();
        }
    }
    return closest;
}

void KMeans::fit() {
    // initialize the centroids to match the data size
    auto len = points[0]->getValue().size();
    centroids.clear();
    std::vector<double> count(numberOfClusters, 0);
    KMeansPlusPlusInit();
    size_t currentIteration = 1;
    bool updated;
    do
    {
        updated = false;
        // std::cout << "KMeans iteration #" << currentIteration++ << ": cluster assignements..." << std::endl;
        // step 2, ri-assegna i
        for(auto& point: points) {
            updated = updated || point->setCluster(closestCluster(*point));
        }
        count = std::vector<double>(numberOfClusters, 0);
        // std::cout << "KMeans iteration #" << currentIteration++ << ": updating centroids..." << std::endl;
        // step 1, compute new cluster centers
        clearClusters();
        for(const auto& data: this->points) {
            for(size_t index = 0; index < len; index++) {
                centroids[data->getCluster()].getValue()[index] += data->getValue()[index];
            }
            count[data->getCluster()]++;
        }

        for(size_t i = 0; i < numberOfClusters; i++) {
            for(size_t index = 0; index < len; index++) {
                (centroids[i]).getValue()[index] /= count[i];
            }
        }
        for(size_t i = 0; i < numberOfClusters; i++)
            centroids[i].print();

    } while (updated);

}

std::vector<Data> KMeans::computeClusterCentroids() {
    fit();
    return centroids;
}

std::vector<Data*> KMeans::getClusterAssignements() {
    return points;
}



void KMeans::KMeansPlusPlusInit() {
    std::random_device r;
    auto e = std::default_random_engine(r());
    std::uniform_int_distribution<size_t> points_dist(0, points.size() - 1);
    centroids.clear();
    auto centroid = Data(0, true);
    centroid.setValue(points[points_dist(e)]->getValue());
    centroids.emplace_back(centroid); // first centroid
    for(size_t i = 1; i < numberOfClusters; i++) {
        auto index = getMaxSeparatedCentroid();
        centroid.setValue(points[index]->getValue());
        centroids.emplace_back(centroid); // first centroid
    }
}

int KMeans::getMaxSeparatedCentroid() {
    double maxDist = 0;
    size_t maxDistIndex = 0;
    for(size_t i = 0; i < points.size(); i++) {
        double dist = 0;
        for(auto& centroid: centroids) {
            dist += distance(*points[i], centroid);
        }
        if(dist >= maxDist) {
            maxDist = dist;
            maxDistIndex = i;
        }
    }
    return maxDistIndex;
}
