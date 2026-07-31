#pragma once

#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <utility>

using namespace std;

struct PositionResult {
    char vote;   
    double prob; 
};

struct ProbPair {
    double p0;   
    double p1;   
};

void sparsify_group(PositionResult group[5], ProbPair output[4]);
std::vector<std::pair<double,double>> computeClusterProbProduct(const std::vector<std::vector<double>>& llrgroup);
unordered_map<string, int> readRefIndexMap(const string& filePath);
std::vector<std::string> readStandardDataBit(const std::string &filePath, size_t expectedLength);
vector<std::string> readFastaFile(const string& filePath);
vector<pair<string, string>> readSequencesFromFASTA(ifstream& file);
extern const std::vector<std::string> v_orig;
extern const std::vector<std::string> v_map;
extern std::map<std::string, std::string> sparse_map;
