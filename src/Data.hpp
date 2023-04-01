#include <vector>
#include <random>
#include <iostream>

class Data {
private:
    std::vector<double> value;
    size_t assignedCluster;
public:
    Data(size_t K, bool isCentroid = false) {
        if(isCentroid) 
            assignedCluster = K;
        else {
            // assegna un cluster casuale
            std::random_device r;
            std::default_random_engine e = std::default_random_engine(r());
            std::uniform_int_distribution<size_t> uniform_dist(0, K - 1);
            assignedCluster = uniform_dist(e);
        }
    }
    size_t getCluster() { return assignedCluster; }
    bool setCluster(size_t cluster) { 
        // return true if the cluster is changed
        if(assignedCluster == cluster) return false;
        assignedCluster = cluster; 
        return true;
    }
    std::vector<double>& getValue() { return value; }
    void setValue(const std::vector<double>& val) {
        value = val;
    }
    void print() const {
        std::string msg = "(";
        for(const auto& val: value) {
            msg += std::to_string(val) + ", "; 
        }
        msg = msg.substr(0, msg.npos - 2) + ")";
        std::cout << msg << std::endl; 
    } 
};