#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>
#include <vector>
#include <bitset>
#include <cmath>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <mutex>
#include "edlib.h"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <ctime>

using namespace std;


struct PositionResult {
    char vote;   
    double prob; 
};

struct ProbPair {
    double p0;   
    double p1;   
};


void sparsify_group(PositionResult group[5], ProbPair output[4]) {
    const char combinations[16][5] = {
        {'0','0','0','0','0'}, {'0','0','0','0','1'}, {'0','0','0','1','0'}, {'0','0','0','1','1'},
        {'0','0','1','0','0'}, {'0','0','1','0','1'}, {'0','0','1','1','0'}, {'1','1','0','0','0'},
        {'0','1','0','0','0'}, {'0','1','0','0','1'}, {'0','1','0','1','0'}, {'1','0','1','0','0'},
        {'0','1','1','0','0'}, {'1','0','0','1','0'}, {'1','0','0','0','1'}, {'1','0','0','0','0'}
    };

    double p[16] = {0};
    for (int i = 0; i < 16; i++) {
        double prob = 1.0;
        for (int j = 0; j < 5; j++) {
            double adjusted_prob = group[j].prob;
            if (adjusted_prob == 0.0) adjusted_prob = 1e-6;
            else if (adjusted_prob == 1.0) adjusted_prob = 1.0 - 1e-6;

            if (combinations[i][j] == '0') {
                if (group[j].vote == '0') prob *= adjusted_prob;
                else prob *= (1 - adjusted_prob);
            } else {
                if (group[j].vote == '1') prob *= adjusted_prob;
                else prob *= (1 - adjusted_prob);
            }
        }
        p[i] = prob;
    }


    double sum_p = 0.0;
    for (int i = 0; i < 16; i++) sum_p += p[i];
    for (int i = 0; i < 16; i++) p[i] /= sum_p;


    output[0].p0 = p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7];
    output[0].p1 = 1 - output[0].p0;

    output[1].p0 = p[0]+p[1]+p[2]+p[3]+p[8]+p[9]+p[10]+p[11];
    output[1].p1 = 1 - output[1].p0;

    output[2].p0 = p[0]+p[1]+p[4]+p[5]+p[8]+p[9]+p[12]+p[13];
    output[2].p1 = 1 - output[2].p0;

    output[3].p0 = p[0]+p[2]+p[4]+p[6]+p[8]+p[10]+p[12]+p[14];
    output[3].p1 = 1 - output[3].p0;
}

std::vector<std::pair<double,double>> computeClusterProbProduct(
    const std::vector<std::vector<double>>& llrgroup)
{
    int N = llrgroup.size();
    int L = llrgroup[0].size();

    std::vector<std::pair<double,double>> prob(L);

    for (int j = 0; j < L; j++) {
        long double logP1 = 0.0;
        long double logP0 = 0.0;

        for (int i = 0; i < N; i++) {
            double p1 = llrgroup[i][j];
            double p0 = 1.0 - p1;

            if (p1 < 1e-15) p1 = 1e-15;
            if (p0 < 1e-15) p0 = 1e-15;

            logP1 += log(p1);
            logP0 += log(p0);
        }

        long double maxLog = std::max(logP1, logP0);
        long double P1 = expl(logP1 - maxLog);
        long double P0 = expl(logP0 - maxLog);

        double norm = P1 + P0;
        prob[j].second = (double)(P1 / norm);
        prob[j].first  = (double)(P0 / norm);
    }

    return prob;
}


vector<pair<string, string>> readSequencesFromFASTA(ifstream& file) {
    vector<pair<string, string>> sequences;

    string line;
    while (getline(file, line)) {
        if (!line.empty() && line[0] == '>') {
            string id = line.substr(1);
            getline(file, line); 
            sequences.push_back({id, line});
        }
    }
    return sequences;
}


std::vector<std::string> readStandardDataBit(const std::string &filePath, size_t expectedLength) {
    std::vector<std::string> data;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Ensure the line is the expected length
        if (line.length() == expectedLength) {
            data.push_back(line);
        } else {
            std::cerr << "Warning: Line length does not match expected " << expectedLength << " characters." << std::endl;
        }
    }

    file.close();
    return data;
}


std::vector<std::string> readFastaFile(const std::string& filePath) {

    std::ifstream inputFile(filePath);
    if (!inputFile) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    std::vector<std::string> sequences;
    std::string line;
    std::string currentSequence;

    while (std::getline(inputFile, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '>') {
            if (!currentSequence.empty()) {
                sequences.push_back(currentSequence);
                currentSequence.clear();
            }
        } else {
            currentSequence += line;
        }
    }
    if (!currentSequence.empty()) {
        sequences.push_back(currentSequence);
    }

    inputFile.close();

    return sequences;
}


map<string, string> sparse_map = {
    {"00000", "0000"},
    {"00001", "0001"},
    {"00010", "0010"},
    {"00011", "0011"},
    {"00100", "0100"},
    {"00101", "0101"},
    {"00110", "0110"},
    {"11000", "0111"},
    {"01000", "1000"},
    {"01001", "1001"},
    {"01010", "1010"},
    {"10100", "1011"},
    {"01100", "1100"},
    {"10010", "1101"},
    {"10001", "1110"},
    {"10000", "1111"}};
