#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <map>
#include <cmath>
#include <cassert>
#include <time.h>
#include <chrono>
#include <bitset>
#include <unordered_map>
#include <set>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <omp.h>
#include "edlib.h"
#include "fba_for_index.h"
#include "order.h"

using namespace std;
using namespace std::chrono;

#define index_length 25
#define Q_LENGTH 15
#define R_LENGTH 11

const int DEFAULT_THRESHOLD_SHW = 3;
const int index_nums = 29988;
const int CODE_SPACE = 1 << Q_LENGTH;   // 15-bit space: 0..32767

extern "C" {
    void order5(int G[K][N], double R[N], double out_D[N]);
}

/*
Output file format:
[0] id
[1] alignLength
[2] code_decimal
*/

vector<pair<string, string>> readSequencesFromFASTQ(ifstream& file) {
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

string reverseComplement(const string& dnaSequence) {
    string complement;
    complement.reserve(dnaSequence.size());
    for (char nucleotide : dnaSequence) {
        switch (nucleotide) {
            case 'A': complement += 'T'; break;
            case 'T': complement += 'A'; break;
            case 'C': complement += 'G'; break;
            case 'G': complement += 'C'; break;
            default:  complement += nucleotide; break;
        }
    }
    reverse(complement.begin(), complement.end());
    return complement;
}

// Constant generator matrix shared across threads
int G[K][N] = {
    {1,1,1,0,1,1,0,1,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,0,1,1,0,1,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0},
    {0,0,1,1,1,0,1,1,0,1,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,1,0,1,1,0,1,0,0,1,0, 0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,1,1,0,1,1,0,1,0,0,1, 0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,1,1,0,1,1,0,1,0,0, 1,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,1,1,0,1,1,0,1,0, 0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,1,1,1,0,1,1,0,1, 0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,1,1,0,1,1,0, 1,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,1,1, 0,1,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,0,1, 1,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0, 1,1,0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1, 0,1,1,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1, 1,0,1,1,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 1,1,0,1,1,0,1,0,0,1}
};

// Generator polynomial
int g[] = {1,1,1,0,1,1,0,1,0,0,1};

// Per-read output record
struct ResultLine {
    int alignLength;
    int code_decimal;
};

// Convert a base to the layer-2 codeword bit
static inline int base_to_cw_bit(char base) {
    if (base == 'A' || base == 'G') return 0;
    if (base == 'T' || base == 'C') return 1;
    return 0;
}

// Extract 25 bases from sequence[startPos] and convert to a 25-bit codeword
static inline bool extract_cw25_from_seq_at(const std::string& seq, int startPos, int cw25[25]) {
    if (startPos < 0) return false;
    if (startPos + 25 > (int)seq.size()) return false;
    for (int i = 0; i < 25; ++i) {
        cw25[i] = base_to_cw_bit(seq[startPos + i]);
    }
    return true;
}

// Check whether cw(x) mod g(x) == 0
static inline bool bch25_syndrome_is_zero(const int cw25[25]) {
    int q_temp[R_LENGTH] = {0};
    int r_temp[R_LENGTH] = {0};

    for (int i = 0; i < R_LENGTH; ++i) q_temp[i] = cw25[i];

    for (int i = 0; i < Q_LENGTH; ++i) {
        if (q_temp[0] == 1) {
            for (int j = 0; j < R_LENGTH; ++j) r_temp[j] = q_temp[j] ^ g[j];
        } else {
            for (int j = 0; j < R_LENGTH; ++j) r_temp[j] = q_temp[j];
        }

        if (i == Q_LENGTH - 1) break;

        for (int j = 1; j < R_LENGTH; ++j) q_temp[j - 1] = r_temp[j];
        q_temp[10] = cw25[i + 11];
    }

    for (int j = 1; j < R_LENGTH; ++j) {
        if (r_temp[j] != 0) return false;
    }
    return true;
}

// Extract the 15-bit message from a valid 25-bit codeword
static inline void bch25_extract_message_q(const int cw25[25], int q_out[Q_LENGTH]) {
    int q_temp[R_LENGTH] = {0};
    int r_temp[R_LENGTH] = {0};

    for (int i = 0; i < R_LENGTH; ++i) q_temp[i] = cw25[i];

    for (int i = 0; i < Q_LENGTH; i++) {
        if (q_temp[0] == 1) {
            q_out[i] = 1;
            for (int j = 0; j < R_LENGTH; ++j) r_temp[j] = q_temp[j] ^ g[j];
        } else {
            q_out[i] = 0;
            for (int j = 0; j < R_LENGTH; ++j) r_temp[j] = q_temp[j];
        }

        if (i == Q_LENGTH - 1) break;

        for (int j = 1; j < R_LENGTH; ++j) q_temp[j - 1] = r_temp[j];
        q_temp[10] = cw25[i + 11];
    }
}

// Convert the 15-bit message to decimal using the original bit order
static inline int q_to_decimal_same_as_yours(const int q[Q_LENGTH]) {
    int code_dec = 0;
    for (int i = 0; i < Q_LENGTH; i++) {
        code_dec += q[14 - i] * (1 << (14 - i));
    }
    return code_dec;
}

int main(int argc, char* argv[])
{
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_filename> <output_path> <data_path> [index_align_threshold]\n";
        return 1;
    }

    const int MAX_SEQ_LEN = 64;

    string input_filename = argv[1];
    string output_path = argv[2];
    string data_path = argv[3];
    int threshold_shw = DEFAULT_THRESHOLD_SHW;
    if (argc >= 5) {
        threshold_shw = atoi(argv[4]);
        if (threshold_shw < 0) {
            cerr << "Error: index_align_threshold must be non-negative." << endl;
            return 1;
        }
    }

    string output_filename = output_path + "/index_identification_results.txt";
    string stat_filename = output_path + "/stat.txt";
    
    string map_filename = data_path + "/index_permutation";
    string pn_filename = data_path + "/watermark_sequence_length_25";

    int un_match = 0;
    int un_match_align = 0;
    int un_match_map_miss = 0;
    int match = 0;
    int total = 0;
    float match_rate = 0.0;
    float unmatch_rate = 0.0;
    float un_match_align_rate = 0.0;
    float un_match_map_miss_rate = 0.0;

    int wm_edit0 = 0;
    int syndrome0 = 0;

    string wm1, wm2;
    ifstream pn_file(pn_filename);
    if (!pn_file.is_open()) {
        cerr << "Error: Could not open file " << pn_filename << endl;
        return 1;
    }

    if (!getline(pn_file, wm1) || !getline(pn_file, wm2)) {
        cerr << "Error: failed to read two watermark sequences from " << pn_filename << endl;
        return 1;
    }
    pn_file.close();

    auto start_time = chrono::high_resolution_clock::now();

    ifstream inputFile4(input_filename);
    if (!inputFile4) {
        printf("Error: unable to open file %s\n", input_filename.c_str());
        return 1;
    }
    vector<pair<string, string>> sequences = readSequencesFromFASTQ(inputFile4);
    inputFile4.close();

    ifstream infile(map_filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << map_filename << endl;
        return 1;
    }

    vector<int> dictionary(CODE_SPACE, -1);
    string line2;
    int ref_number = 1;
    while (getline(infile, line2)) {
        stringstream ss(line2);
        int value;
        if (ss >> value) {
            if (value >= 0 && value < CODE_SPACE) {
                dictionary[value] = ref_number;
            } else {
                cerr << "Warning: value " << value
                     << " out of range [0, " << CODE_SPACE - 1 << "]\n";
            }
            ref_number++;
        } else {
            cerr << "Warning: invalid line in file: " << line2 << endl;
        }
    }
    infile.close();

    vector<ResultLine> results(sequences.size(), {-1, -1});

    #pragma omp parallel
    {
        int match_local = 0;
        int un_match_local = 0;
        int unmatch_align_local = 0;
        int unmatch_map_miss_local = 0;

        int wm_edit0_local = 0;
        int syndrome0_local = 0;

        char wm_buff[MAX_SEQ_LEN];

        #pragma omp for
        for (int j = 0; j < (int)sequences.size(); ++j)
        {
            const auto& sequence_pair = sequences[j];
            string sequence = sequence_pair.second;

            int seq_length = (int)sequence.size();
            seq_length = 30;
            sequence = sequence.substr(0, seq_length);

            int local_code_decimal = -1;
            int local_editDistance_wm = -1;
            int local_alignLength = -1;
            int local_startPos = -1;

            for (int l = 0; l < seq_length; ++l)
            {
                char b = sequence[l];
                if (b == 'A' || b == 'T') wm_buff[l] = '0';
                else if (b == 'G' || b == 'C') wm_buff[l] = '1';
                else wm_buff[l] = '0';
            }

            EdlibAlignResult align_result1 = edlibAlign(
                wm1.c_str(), index_length, wm_buff, seq_length,
                edlibNewAlignConfig(-1, EDLIB_MODE_SHW, EDLIB_TASK_LOC, NULL, 0));

            EdlibAlignResult align_result2 = edlibAlign(
                wm2.c_str(), index_length, wm_buff, seq_length,
                edlibNewAlignConfig(-1, EDLIB_MODE_SHW, EDLIB_TASK_LOC, NULL, 0));

            int editDistance1 = align_result1.editDistance;
            int editDistance2 = align_result2.editDistance;

            bool ok1 = (align_result1.status == EDLIB_STATUS_OK &&
                        align_result1.editDistance >= 0 &&
                        align_result1.numLocations > 0 &&
                        align_result1.startLocations != NULL &&
                        align_result1.endLocations != NULL);
            bool ok2 = (align_result2.status == EDLIB_STATUS_OK &&
                        align_result2.editDistance >= 0 &&
                        align_result2.numLocations > 0 &&
                        align_result2.startLocations != NULL &&
                        align_result2.endLocations != NULL);

            if (!ok1 && !ok2) {
                results[j] = {-1, -1};
                un_match_local++;
                unmatch_align_local++;
                edlibFreeAlignResult(align_result1);
                edlibFreeAlignResult(align_result2);
                continue;
            }

            int startPos1 = ok1 ? align_result1.startLocations[0] : -1;
            int endPos1   = ok1 ? align_result1.endLocations[0]   : -1;
            int alignLength1 = ok1 ? (endPos1 - startPos1 + 1) : 0;

            int startPos2 = ok2 ? align_result2.startLocations[0] : -1;
            int endPos2   = ok2 ? align_result2.endLocations[0]   : -1;
            int alignLength2 = ok2 ? (endPos2 - startPos2 + 1) : 0;

            bool choose1 = false;
            bool choose2 = false;

            if (ok1 && ok2) {
                if (editDistance1 <= editDistance2 &&
                    alignLength1 > 0 && editDistance1 <= threshold_shw) {
                    choose1 = true;
                } else if (editDistance2 <= editDistance1 &&
                           alignLength2 > 0 && editDistance2 <= threshold_shw) {
                    choose2 = true;
                }
            } else if (ok1) {
                if (alignLength1 > 0 && editDistance1 <= threshold_shw) {
                    choose1 = true;
                }
            } else if (ok2) {
                if (alignLength2 > 0 && editDistance2 <= threshold_shw) {
                    choose2 = true;
                }
            }

            if (choose1)
            {
                local_editDistance_wm = editDistance1;
                local_alignLength = alignLength1;
                local_startPos = startPos1;

                string index_fb = sequence.substr(startPos1, alignLength1);
                char index_fb_char[64];
                for (int i = 0; i < alignLength1; i++) index_fb_char[i] = index_fb[i];
                index_fb_char[alignLength1] = '\0';

                bool fast_done = false;
                if (local_editDistance_wm == 0) {
                    wm_edit0_local++;
                    int cw25_bits[25];
                    if (extract_cw25_from_seq_at(sequence, local_startPos, cw25_bits) &&
                        bch25_syndrome_is_zero(cw25_bits))
                    {
                        syndrome0_local++;

                        int q_bits[Q_LENGTH] = {0};
                        bch25_extract_message_q(cw25_bits, q_bits);
                        int code_dec = q_to_decimal_same_as_yours(q_bits);

                        if (code_dec >= 0 && code_dec < CODE_SPACE) {
                            int mapped = dictionary[code_dec];
                            if (mapped >= 1 && mapped <= index_nums) {
                                local_code_decimal = mapped;
                                match_local++;
                                fast_done = true;
                            } else {
                                local_code_decimal = -1;
                                un_match_local++;
                                unmatch_map_miss_local++;
                                fast_done = true;
                            }
                        } else {
                            local_code_decimal = -1;
                            un_match_local++;
                            unmatch_map_miss_local++;
                            fast_done = true;
                        }
                    }
                }

                if (!fast_done)
                {
                    char wm1_char[index_length + 1];
                    for (int i = 0; i < index_length; i++) wm1_char[i] = wm1[i];
                    wm1_char[index_length] = '\0';

                    vector<double> llr_1;
                    FBA_with_PN_pilots(25, wm1_char, index_fb_char, llr_1);

                    double R_local[N], D_local[N];
                    int deco_local[N];

                    for (int i = 0; i < N; i++) {
                        double p1 = llr_1[i];
                        double p0 = 1.0 - p1;
                        if (p1 < 1e-12) p1 = 1e-12;
                        if (p0 < 1e-12) p0 = 1e-12;
                        R_local[i] = log(p0 / p1);
                    }

                    order5(G, R_local, D_local);

                    for (int i = 0; i < N; i++) {
                        deco_local[i] = (D_local[i] == -1.0) ? 0 : 1;
                    }

                    int q[Q_LENGTH] = {0};
                    int r_temp[R_LENGTH] = {0};
                    int q_temp[R_LENGTH] = {0};

                    for (int i = 0; i < R_LENGTH; i++) q_temp[i] = deco_local[i];

                    for (int i = 0; i < Q_LENGTH; i++) {
                        if (q_temp[0] == 1) {
                            q[i] = 1;
                            for (int j2 = 0; j2 < R_LENGTH; j2++) r_temp[j2] = q_temp[j2] ^ g[j2];
                        } else {
                            q[i] = 0;
                            for (int j2 = 0; j2 < R_LENGTH; j2++) r_temp[j2] = q_temp[j2];
                        }

                        if (i == Q_LENGTH - 1) break;

                        for (int j2 = 1; j2 < R_LENGTH; j2++) q_temp[j2 - 1] = r_temp[j2];
                        q_temp[10] = deco_local[i + 11];
                    }

                    int code_dec = 0;
                    for (int i = 0; i < 15; i++) {
                        code_dec += q[14 - i] * (1 << (14 - i));
                    }

                    if (code_dec >= 0 && code_dec < CODE_SPACE) {
                        int mapped = dictionary[code_dec];
                        if (mapped >= 1 && mapped <= index_nums) {
                            local_code_decimal = mapped;
                            match_local++;
                        } else {
                            local_code_decimal = -1;
                            un_match_local++;
                            unmatch_map_miss_local++;
                        }
                    } else {
                        local_code_decimal = -1;
                        un_match_local++;
                        unmatch_map_miss_local++;
                    }
                }
            }
            else if (choose2)
            {
                local_editDistance_wm = editDistance2;
                local_alignLength = alignLength2;
                local_startPos = startPos2;

                string index_fb = sequence.substr(startPos2, alignLength2);
                char index_fb_char[64];
                for (int i = 0; i < alignLength2; i++) index_fb_char[i] = index_fb[i];
                index_fb_char[alignLength2] = '\0';

                bool fast_done = false;
                if (local_editDistance_wm == 0) {
                    wm_edit0_local++;
                    int cw25_bits[25];
                    if (extract_cw25_from_seq_at(sequence, local_startPos, cw25_bits) &&
                        bch25_syndrome_is_zero(cw25_bits))
                    {
                        syndrome0_local++;

                        int q_bits[Q_LENGTH] = {0};
                        bch25_extract_message_q(cw25_bits, q_bits);
                        int code_dec = q_to_decimal_same_as_yours(q_bits);

                        if (code_dec >= 0 && code_dec < CODE_SPACE) {
                            int mapped = dictionary[code_dec];
                            if (mapped >= 1 && mapped <= index_nums) {
                                local_code_decimal = mapped;
                                match_local++;
                                fast_done = true;
                            } else {
                                local_code_decimal = -1;
                                un_match_local++;
                                unmatch_map_miss_local++;
                                fast_done = true;
                            }
                        } else {
                            local_code_decimal = -1;
                            un_match_local++;
                            unmatch_map_miss_local++;
                            fast_done = true;
                        }
                    }
                }

                if (!fast_done)
                {
                    char wm2_char[index_length + 1];
                    for (int i = 0; i < index_length; i++) wm2_char[i] = wm2[i];
                    wm2_char[index_length] = '\0';

                    vector<double> llr_1;
                    FBA_with_PN_pilots(25, wm2_char, index_fb_char, llr_1);

                    double R_local[N], D_local[N];
                    int deco_local[N];

                    for (int i = 0; i < N; i++) {
                        double p1 = llr_1[i];
                        double p0 = 1.0 - p1;
                        if (p1 < 1e-12) p1 = 1e-12;
                        if (p0 < 1e-12) p0 = 1e-12;
                        R_local[i] = log(p0 / p1);
                    }

                    order5(G, R_local, D_local);

                    for (int i = 0; i < N; i++) {
                        deco_local[i] = (D_local[i] == -1.0) ? 0 : 1;
                    }

                    int q[Q_LENGTH] = {0};
                    int r_temp[R_LENGTH] = {0};
                    int q_temp[R_LENGTH] = {0};

                    for (int i = 0; i < R_LENGTH; i++) q_temp[i] = deco_local[i];

                    for (int i = 0; i < Q_LENGTH; i++) {
                        if (q_temp[0] == 1) {
                            q[i] = 1;
                            for (int j2 = 0; j2 < R_LENGTH; j2++) r_temp[j2] = q_temp[j2] ^ g[j2];
                        } else {
                            q[i] = 0;
                            for (int j2 = 0; j2 < R_LENGTH; j2++) r_temp[j2] = q_temp[j2];
                        }

                        if (i == Q_LENGTH - 1) break;

                        for (int j2 = 1; j2 < R_LENGTH; j2++) q_temp[j2 - 1] = r_temp[j2];
                        q_temp[10] = deco_local[i + 11];
                    }

                    int code_dec = 0;
                    for (int i = 0; i < 15; i++) {
                        code_dec += q[14 - i] * (1 << (14 - i));
                    }

                    if (code_dec >= 0 && code_dec < CODE_SPACE) {
                        int mapped = dictionary[code_dec];
                        if (mapped >= 1 && mapped <= index_nums) {
                            local_code_decimal = mapped;
                            match_local++;
                        } else {
                            local_code_decimal = -1;
                            un_match_local++;
                            unmatch_map_miss_local++;
                        }
                    } else {
                        local_code_decimal = -1;
                        un_match_local++;
                        unmatch_map_miss_local++;
                    }
                }
            }
            else
            {
                local_editDistance_wm = -1;
                local_startPos = -1;
                local_alignLength = -1;
                local_code_decimal = -1;
                un_match_local++;
                unmatch_align_local++;
            }

            results[j] = {
                local_alignLength,
                local_code_decimal
            };

            edlibFreeAlignResult(align_result1);
            edlibFreeAlignResult(align_result2);
        }

        #pragma omp atomic
        match += match_local;
        #pragma omp atomic
        un_match += un_match_local;
        #pragma omp atomic
        un_match_align += unmatch_align_local;
        #pragma omp atomic
        un_match_map_miss += unmatch_map_miss_local;
        #pragma omp atomic
        wm_edit0 += wm_edit0_local;
        #pragma omp atomic
        syndrome0 += syndrome0_local;
    }

    ofstream error_file(output_filename);
    if (!error_file) {
        printf("Error: unable to open file %s\n", output_filename.c_str());
        return 1;
    }

    for (size_t j = 0; j < sequences.size(); ++j) {
        const auto& sequence_pair = sequences[j];
        const string& id = sequence_pair.first;
        const ResultLine& r = results[j];

        error_file << id << ","
                   << r.alignLength << ","
                   << r.code_decimal << "\n";
    }
    error_file.close();

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();

    total = match + un_match;
    match_rate = total > 0 ? (float)match / total * 100 : 0.0f;
    unmatch_rate = total > 0 ? (float)un_match / total * 100 : 0.0f;
    un_match_align_rate = total > 0 ? (float)un_match_align / total * 100 : 0.0f;
    un_match_map_miss_rate = total > 0 ? (float)un_match_map_miss / total * 100 : 0.0f;

    float wm_edit0_rate = total > 0 ? (float)wm_edit0 / total * 100 : 0.0f;
    float syndrome0_rate = total > 0 ? (float)syndrome0 / total * 100 : 0.0f;

    printf("Index align threshold: %d\n", threshold_shw);
    printf("Total: %d\n", total);
    printf("Passed: %d (%.2f%%)\n", match, match_rate);
    printf("Failed: %d (%.2f%%)\n", un_match, unmatch_rate);
    printf("Syndrome pass: %d (%.2f%%)\n", syndrome0, syndrome0_rate);
    cout << "Time used: " << seconds << " seconds" << endl;

    ofstream stat_file(stat_filename);
    stat_file << "index_align_threshold: " << threshold_shw << "\n";
    stat_file << "total: " << total << "\n";
    stat_file << "match: " << match << " (" << match_rate << "%)\n";
    stat_file << "failed: " << un_match << " (" << unmatch_rate << "%)\n";
    stat_file << "Syndrome pass: " << syndrome0 << " (" << syndrome0_rate << "%)\n";
    stat_file << "Time used: " << seconds << " seconds\n";
    stat_file.close();

    return 0;
}
