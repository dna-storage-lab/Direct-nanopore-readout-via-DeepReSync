#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/stat.h>

#include "mod2sparse.h"
#include "mod2dense.h"
#include "mod2convert.h"
#include "rcode.h"
#include "dec.h"
#include "check.h"

#define EACHOLIGO_SIZE (423)
#define OLIGOS_PER_CODEWORD (153)
#define PERMUTED_BLOCK_LEN (64719)
#define SRC_INFO_LEN (53919)

#define ROW (30 * 360)        // 10800
#define COLUMN (180 * 360)    // 64800
#define max_iteration 50

using namespace std;


#ifdef LDPC_DECODE_USE_MUTEX
#include <mutex>
static std::mutex g_ldpc_mtx;
#endif


int ldpcdecode(const double* cblk_double,
               char* information_bit,
               const int* randomize,
               const char* configure)
{
    static thread_local bool pchk_loaded = false;
    static thread_local std::string loaded_from;

    if (!pchk_loaded || loaded_from != std::string(configure)) {
        char pchk_file[512];
        std::snprintf(pchk_file, sizeof(pchk_file), "%s/dvb_s2_r5_6.pchk", configure);

        read_pchk(pchk_file);

        pchk_loaded = true;
        loaded_from = std::string(configure);
    }

    static thread_local std::vector<char>   decode_blk(COLUMN);
    static thread_local std::vector<double> LLR_data(COLUMN);
    static thread_local std::vector<char>   parity_blk(ROW);
    static thread_local std::vector<double> probability(COLUMN - ROW);

    char*   decode_blkpt   = decode_blk.data();
    double* LLR_pt         = LLR_data.data();
    char*   parity_blkpt   = parity_blk.data();
    double* probability_pt = probability.data();

    for (int j = 0; j < COLUMN - 81; j++) {
        LLR_pt[j] = cblk_double[j];
    }
    for (int j = COLUMN - 81; j < COLUMN; j++) {
        LLR_pt[j] = 0.0;
    }

    mod2entry* e;
    for (int j = 0; j < COLUMN; j++) {
        for (e = mod2sparse_first_in_col(H, j);
             !mod2sparse_at_end(e);
             e = mod2sparse_next_in_col(e))
        {
            e->pr = LLR_pt[j];
            e->lr = 1;
        }
        decode_blkpt[j] = (LLR_pt[j] >= 1.0);
    }

    for (int it = 0; ; it++) {
        int c = check(H, decode_blkpt, parity_blkpt);
        if (it == max_iteration || it == -max_iteration || (max_iteration > 0 && c == 0)) break;
        iterprp(H, LLR_pt, decode_blkpt, probability_pt);
    }

    int valid = (check(H, decode_blkpt, parity_blkpt) == 0);

    for (int j = 0; j < 53919; j++) {
        information_bit[j] = ((decode_blkpt[j + 10800] + randomize[j]) & 1) + '0';
    }

    return valid;
}


static inline bool file_exists(const std::string& path) {
    struct stat buf;
    return (stat(path.c_str(), &buf) == 0 && S_ISREG(buf.st_mode));
}

static inline size_t parse_size_t_or(const char* s, size_t fallback) {
    if (!s) return fallback;
    try {
        unsigned long long v = std::stoull(s);
        return static_cast<size_t>(v);
    } catch (...) {
        return fallback;
    }
}

static bool read_binary_int_array(const std::string& path, int* arr, size_t n) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    size_t nr = fread(arr, sizeof(int), n, fp);
    fclose(fp);
    return (nr == n);
}


static bool load_llr_text_file(
    const std::string& llr_path,
    double** LLR_combined_2,
    int* countllr,
    int total_oligos
) {
    std::ifstream fin(llr_path);
    if (!fin.is_open()) {
        std::cerr << "Cannot open LLR file: " << llr_path << "\n";
        return false;
    }

    const double neutral_eps = 1e-12;
    std::string line;
    int row = 0;

    while (row < total_oligos && std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        bool any_present = false;

        for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
            double v;
            if (!(iss >> v)) {
                std::cerr << "Format error in LLR file at row " << row
                          << ", column " << j << ". Expected 423 doubles per line.\n";
                return false;
            }
            LLR_combined_2[row][j] = v;
            if (std::fabs(v - 1.0) > neutral_eps) {
                any_present = true;
            }
        }

        countllr[row] = any_present ? 1 : 0;
        row++;
    }

    if (row != total_oligos) {
        std::cerr << "LLR file row count mismatch. Read " << row
                  << " rows, expected " << total_oligos << " rows.\n";
        return false;
    }

    return true;
}



static bool load_llr_binary_file(
    const std::string& llr_path,
    double** LLR_combined_2,
    int* countllr,
    int total_oligos
) {
    FILE* fp = fopen(llr_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Cannot open binary LLR file: " << llr_path << "\n";
        return false;
    }

    const double neutral_eps = 1e-12;

    for (int row = 0; row < total_oligos; ++row) {
        size_t nr = fread(LLR_combined_2[row], sizeof(double), EACHOLIGO_SIZE, fp);
        if (nr != EACHOLIGO_SIZE) {
            std::cerr << "Binary LLR file size mismatch at row " << row
                      << ". Read " << nr << " doubles, expected "
                      << EACHOLIGO_SIZE << ".\n";
            fclose(fp);
            return false;
        }

        bool any_present = false;
        for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
            if (std::fabs(LLR_combined_2[row][j] - 1.0) > neutral_eps) {
                any_present = true;
                break;
            }
        }
        countllr[row] = any_present ? 1 : 0;
    }


    double extra;
    if (fread(&extra, sizeof(double), 1, fp) != 0) {
        std::cerr << "Warning: binary LLR file contains extra data beyond expected size.\n";
    }

    fclose(fp);
    return true;
}



struct DecodeResult {
    int ok = 0;              
    int dropout = 0;         
    std::string info53919;   
};

// ============================================================
// ldpcdecode worker
// ============================================================

static void decode_codewords_worker(
    int kin_begin, int kin_end,                // [begin, end)
    int kk,                                    // kk = pool - 1
    double** LLR_combined_2,
    const int* countllr,
    const int* permutation64719,
    int* randomize,
    const char* configure,
    std::vector<DecodeResult>& results,
    std::atomic<int>& success_cnt
) {
    const double neutral_eps = 1e-12;


    std::vector<double> cblk_double(PERMUTED_BLOCK_LEN, 1.0);
    std::vector<double> cblk_doublet(PERMUTED_BLOCK_LEN, 1.0);
    std::vector<char>   cblk_int(PERMUTED_BLOCK_LEN, 'e');
    std::vector<char>   cblk_intt(PERMUTED_BLOCK_LEN, 'e');
    std::vector<char>   information_bit64719(PERMUTED_BLOCK_LEN, '0');

    for (int kin = kin_begin; kin < kin_end; ++kin) {

        std::fill(cblk_double.begin(),  cblk_double.end(),  1.0);
        std::fill(cblk_doublet.begin(), cblk_doublet.end(), 1.0);
        std::fill(cblk_int.begin(),     cblk_int.end(),     'e');
        std::fill(cblk_intt.begin(),    cblk_intt.end(),    'e');
        std::fill(information_bit64719.begin(), information_bit64719.end(), '0');

        // 153 * 423 -> 64719
        int p = 0;
        for (int k = 0; k < OLIGOS_PER_CODEWORD; ++k) {
            int oligo_pos = k + kin * OLIGOS_PER_CODEWORD;

            for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
                double llr = LLR_combined_2[oligo_pos][j];
                cblk_double[p] = llr;

                if (std::fabs(llr - 1.0) <= neutral_eps) {
                    cblk_int[p] = 'e';
                } else {
                    cblk_int[p] = (llr > 1.0) ? '1' : '0';
                }
                ++p;
            }
        }

        // dropout 
        int sum_present = 0;
        for (int j = 0; j < OLIGOS_PER_CODEWORD; ++j) {
            sum_present += countllr[kin * OLIGOS_PER_CODEWORD + j];
        }
        int dropout = OLIGOS_PER_CODEWORD - sum_present;

        // permutation
        for (int j = 0; j < PERMUTED_BLOCK_LEN; ++j) {
            int to = permutation64719[(j + kin + kk * 196) % PERMUTED_BLOCK_LEN];
            cblk_doublet[to] = cblk_double[j];
            cblk_intt[to]    = cblk_int[j];
        }

        // LDPC decode
        int ok = 0;
#ifdef LDPC_DECODE_USE_MUTEX
        {
            std::lock_guard<std::mutex> lk(g_ldpc_mtx);
            ok = ldpcdecode(
                cblk_doublet.data(),
                information_bit64719.data(),
                randomize,
                configure
            );
        }
#else
        ok = ldpcdecode(
            cblk_doublet.data(),
            information_bit64719.data(),
            randomize,
            configure
        );
#endif

        if (ok) {
            success_cnt.fetch_add(1, std::memory_order_relaxed);
        }


        std::string line;
        line.resize(SRC_INFO_LEN);
        for (int i = 0; i < SRC_INFO_LEN; ++i) {
            line[i] = information_bit64719[i];
        }

        DecodeResult r;
        r.ok = ok;
        r.dropout = dropout;
        r.info53919 = std::move(line);

        results[kin] = std::move(r);
    }
}

// ============================================================
// main
// usage:
//   ./prog <llr_file> <save_path> <pool> <threads> <configure>
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <llr_file> <save_path> <pool> <threads> <configure>\n";
        return 1;
    }

    const char* llr_file_c   = argv[1];
    const char* save_path_c  = argv[2];
    const char* poolindex_c  = argv[3];
    const char* threadnum_c  = argv[4];
    const char* configure    = argv[5];

    size_t num_codewords = 196;

    std::string llr_file(llr_file_c);
    std::string save_path(save_path_c);

    int pool = std::stoi(poolindex_c);
    int num_threads_int = std::stoi(threadnum_c);
    if (num_threads_int <= 0) num_threads_int = 1;
    size_t num_threads = static_cast<size_t>(num_threads_int);

    if (!file_exists(llr_file)) {
        std::cerr << "LLR file does not exist: " << llr_file << "\n";
        return 1;
    }


    std::string recovered_src_path = save_path + "/src_information.txt";
    std::string summary_path       = save_path + "/summary.txt";


    std::string permutation_path = std::string(configure) + "/sequence_permutation";
    std::string random_path      = std::string(configure) + "/sequence_random";


    const int MAX_CODEWORDS_PER_POOL = 196;
    if (num_codewords > (size_t)MAX_CODEWORDS_PER_POOL) {
        std::cerr << "num_codewords cannot exceed " << MAX_CODEWORDS_PER_POOL << "\n";
        return 1;
    }

    int cwN = static_cast<int>(num_codewords);
    int total_oligos = cwN * OLIGOS_PER_CODEWORD;


    int* permutation64719 = new int[PERMUTED_BLOCK_LEN];
    if (!read_binary_int_array(permutation_path, permutation64719, PERMUTED_BLOCK_LEN)) {
        std::cerr << "Cannot read permutation64719 from: " << permutation_path << "\n";
        delete[] permutation64719;
        return 1;
    }


    int randomize[SRC_INFO_LEN];
    if (!read_binary_int_array(random_path, randomize, SRC_INFO_LEN)) {
        std::cerr << "Cannot read SequenceRandom from: " << random_path << "\n";
        delete[] permutation64719;
        return 1;
    }


    double** LLR_combined_2 = new double*[total_oligos];
    for (int i = 0; i < total_oligos; ++i) {
        LLR_combined_2[i] = new double[EACHOLIGO_SIZE];
        for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
            LLR_combined_2[i][j] = 1.0;
        }
    }

    int* countllr = new int[total_oligos];
    for (int i = 0; i < total_oligos; ++i) countllr[i] = 0;

    // summary
    ofstream summary_file(summary_path);
    if (!summary_file.is_open()) {
        std::cerr << "Cannot open summary file: " << summary_path << "\n";
        delete[] permutation64719;
        delete[] countllr;
        for (int i = 0; i < total_oligos; ++i) delete[] LLR_combined_2[i];
        delete[] LLR_combined_2;
        return 1;
    }



    std::cout << "Load LLR binary file\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!load_llr_binary_file(llr_file, LLR_combined_2, countllr, total_oligos)) {
        summary_file.close();
        delete[] permutation64719;
        delete[] countllr;
        for (int i = 0; i < total_oligos; ++i) delete[] LLR_combined_2[i];
        delete[] LLR_combined_2;
        return 1;
    }

    int total_present_oligos = 0;
    for (int i = 0; i < total_oligos; ++i) {
        if (countllr[i]) total_present_oligos++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur1 = t1 - t0;

    int total_dropout = cwN * OLIGOS_PER_CODEWORD - total_present_oligos;

    std::cout << "LLR loading time: " << dur1.count() << " sec\n";
    std::cout << "dropout oligo nums: " << total_dropout << "\n";
    summary_file << "LLR loading time: " << dur1.count() << " sec\n";
    summary_file << "dropout oligo nums: " << total_dropout << "\n";

    

    std::cout << "\nMulti-thread LDPC decoding\n";
    auto t2 = std::chrono::high_resolution_clock::now();

    int kk = pool - 1;
    std::vector<DecodeResult> results(cwN);
    std::atomic<int> successnum_atomic{0};

    std::vector<std::thread> decode_threads;
    int per_cw = (cwN + (int)num_threads - 1) / (int)num_threads;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        int s = (int)tid * per_cw;
        int e = std::min(cwN, s + per_cw);
        if (s >= e) break;

        decode_threads.emplace_back(
            decode_codewords_worker,
            s, e,
            kk,
            LLR_combined_2,
            countllr,
            permutation64719,
            randomize,
            configure,
            std::ref(results),
            std::ref(successnum_atomic)
        );
    }

    for (auto& th : decode_threads) th.join();

    FILE* filesrc = fopen(recovered_src_path.c_str(), "w");
    if (!filesrc) {
        std::cerr << "Cannot open " << recovered_src_path << "\n";
        summary_file.close();
        delete[] permutation64719;
        delete[] countllr;
        for (int i = 0; i < total_oligos; ++i) delete[] LLR_combined_2[i];
        delete[] LLR_combined_2;
        return 1;
    }
    
    for (int kin = 0; kin < cwN; ++kin) {
        const auto& r = results[kin];
        fwrite(r.info53919.data(), 1, r.info53919.size(), filesrc);
        fputc('\n', filesrc);
    }
    
    fclose(filesrc);

    int successnum = successnum_atomic.load();

    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur3 = t3 - t2;

    std::cout << "success decoding: " << successnum << "\n";
    std::cout << "fail decoding: " << (cwN - successnum) << "\n";
    std::cout << "Decoding time: " << dur3.count() << " sec\n";

    summary_file << "success decoding: " << successnum << "\n";
    summary_file << "fail decoding: " << (cwN - successnum) << "\n";
    summary_file << "Decoding time: " << dur3.count() << " sec\n";
    summary_file.close();

    delete[] permutation64719;
    delete[] countllr;

    for (int i = 0; i < total_oligos; ++i) {
        delete[] LLR_combined_2[i];
    }
    delete[] LLR_combined_2;

    return 0;
}