#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include "utils.h"
#include "parameters.h"

using namespace std;

namespace {

const int POSTERIOR_LEN_423 = EACHOLIGO_SIZE;
const char POSTERIOR_MAGIC_423[8] = {'R', 'P', 'S', 'T', '4', '2', '3', '\0'};
const uint32_t POSTERIOR_VERSION = 1;

struct ConsensusConfig {
    std::string mode = "prob_clip";
    double final_prob_eps = 1e-6;
    double log_lr_scale = 0.5;
    double final_log_lr_clip = 8.0;
    double read_log_lr_clip = 4.0;
    int max_reads_per_oligo = 25;
};

static std::string env_string_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    return std::string(value);
}

static double env_double_or(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) return fallback;
    return parsed;
}

static int env_int_or(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0) return fallback;
    return static_cast<int>(parsed);
}

static ConsensusConfig load_consensus_config() {
    ConsensusConfig cfg;
    cfg.mode = env_string_or("CONSENSUS_MODE", cfg.mode);
    if (cfg.mode != "prob_clip" && cfg.mode != "llr_calibrated") {
        cfg.mode = "prob_clip";
    }
    cfg.final_prob_eps = env_double_or("CONSENSUS_FINAL_PROB_EPS", cfg.final_prob_eps);
    if (cfg.final_prob_eps <= 0.0 || cfg.final_prob_eps >= 0.5) cfg.final_prob_eps = 1e-6;
    cfg.log_lr_scale = env_double_or("CONSENSUS_LOG_LR_SCALE", cfg.log_lr_scale);
    cfg.final_log_lr_clip = env_double_or("CONSENSUS_FINAL_LOG_LR_CLIP", cfg.final_log_lr_clip);
    cfg.read_log_lr_clip = env_double_or("CONSENSUS_READ_LOG_LR_CLIP", cfg.read_log_lr_clip);
    cfg.max_reads_per_oligo = env_int_or("CONSENSUS_MAX_READS_PER_OLIGO", cfg.max_reads_per_oligo);
    if (cfg.log_lr_scale < 0.0) cfg.log_lr_scale = 0.5;
    if (cfg.final_log_lr_clip <= 0.0) cfg.final_log_lr_clip = 8.0;
    if (cfg.read_log_lr_clip <= 0.0) cfg.read_log_lr_clip = 4.0;
    return cfg;
}

bool read_u32(std::ifstream& in, uint32_t& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return in.good();
}

bool read_u64(std::ifstream& in, uint64_t& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return in.good();
}

static inline double clipped_lr_from_probability(double p1, const ConsensusConfig& cfg) {
    if (!std::isfinite(p1)) p1 = 0.5;
    if (p1 < cfg.final_prob_eps) p1 = cfg.final_prob_eps;
    if (p1 > 1.0 - cfg.final_prob_eps) p1 = 1.0 - cfg.final_prob_eps;
    return p1 / (1.0 - p1);
}

static inline bool use_llr_calibrated(const ConsensusConfig& cfg) {
    return cfg.mode == "llr_calibrated";
}

static inline double clamp_log_lr(double x, double limit) {
    if (x < -limit) return -limit;
    if (x > limit) return limit;
    return x;
}

static inline double clipped_lr_from_lr(double lr, const ConsensusConfig& cfg) {
    if (std::isnan(lr) || lr < 0.0) return 1.0;
    if (lr == 0.0) return clipped_lr_from_probability(0.0, cfg);
    if (std::isinf(lr)) return clipped_lr_from_probability(1.0, cfg);
    return clipped_lr_from_probability(lr / (1.0 + lr), cfg);
}

static inline double merged_lr_from_log_sum(double log_sum, bool has_info, const ConsensusConfig& cfg) {
    if (!has_info) return 1.0;
    if (use_llr_calibrated(cfg)) {
        return std::exp(clamp_log_lr(cfg.log_lr_scale * log_sum, cfg.final_log_lr_clip));
    }
    double p1;
    if (log_sum >= 0.0) {
        p1 = 1.0 / (1.0 + std::exp(-log_sum));
    } else {
        double e = std::exp(log_sum);
        p1 = e / (1.0 + e);
    }
    return clipped_lr_from_probability(p1, cfg);
}

static inline double calibrated_lr_from_lr(double lr, const ConsensusConfig& cfg) {
    if (std::isnan(lr) || lr <= 0.0) return 1.0;
    if (std::isinf(lr)) {
        return std::exp(cfg.final_log_lr_clip);
    }
    return std::exp(clamp_log_lr(cfg.log_lr_scale * std::log(lr), cfg.final_log_lr_clip));
}

static inline double prob_to_lr(double p1) {
    if (!std::isfinite(p1)) p1 = 0.5;
    if (p1 < 1e-9) p1 = 1e-9;
    if (p1 > 1.0 - 1e-9) p1 = 1.0 - 1e-9;
    return p1 / (1.0 - p1);
}

static std::vector<double> per_read_post423_to_lr423(const std::vector<double>& post423) {
    std::vector<double> lr423(EACHOLIGO_SIZE, 1.0);
    for (int i = 0; i < EACHOLIGO_SIZE && i < (int)post423.size(); ++i) {
        lr423[i] = prob_to_lr(post423[i]);
    }
    return lr423;
}

bool read_posterior_header(std::ifstream& in, uint64_t& record_count, uint32_t& posterior_len) {
    char magic[8];
    uint32_t version = 0;

    in.read(magic, sizeof(magic));
    if (!in.good()) {
        std::cerr << "Invalid read posterior file magic.\n";
        return false;
    }
    if (!read_u32(in, version) || !read_u32(in, posterior_len) || !read_u64(in, record_count)) {
        std::cerr << "Invalid read posterior file header.\n";
        return false;
    }
    bool is423 = std::string(magic, sizeof(magic)) == std::string(POSTERIOR_MAGIC_423, sizeof(POSTERIOR_MAGIC_423));
    if (!is423) {
        std::cerr << "Unsupported read posterior format; expected RPST423.\n";
        return false;
    }
    if (version != POSTERIOR_VERSION || posterior_len != POSTERIOR_LEN_423) {
        std::cerr << "Unsupported read posterior file version or posterior length.\n";
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <read_posteriors.bin> <save_path> <pool> <thread>" << endl;
        return 1;
    }

    const char* posterior_path = argv[1];
    const char* save_path = argv[2];
    const char* poolindex = argv[3];
    const char* threadnum = argv[4];

    (void)poolindex;
    (void)threadnum;

    std::string llr_save_path = std::string(save_path) + "/llrs_for_decoding.bin";
    std::string summary_path = std::string(save_path) + "/summary.txt";
    std::string hard_bits_path = std::string(save_path) + "/bitstream_before_decoding.txt";

    std::ifstream in(posterior_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Cannot open read posterior file: " << posterior_path << "\n";
        return 1;
    }

    uint64_t record_count = 0;
    uint32_t posterior_len = 0;
    if (!read_posterior_header(in, record_count, posterior_len)) {
        return 1;
    }
    ConsensusConfig consensus_cfg = load_consensus_config();

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> log_sums(OLIGO_NUMS, std::vector<double>(EACHOLIGO_SIZE, 0.0));
    std::vector<std::vector<unsigned char>> has_info(OLIGO_NUMS, std::vector<unsigned char>(EACHOLIGO_SIZE, 0));
    std::vector<int> read_counts(OLIGO_NUMS, 0);

    std::vector<double> posterior(posterior_len, 0.0);
    uint64_t used_records = 0;
    uint64_t valid_records = 0;
    uint64_t capped_records = 0;

    for (uint64_t rec = 0; rec < record_count; ++rec) {
        int32_t pos_in_codeword = 0;
        in.read(reinterpret_cast<char*>(&pos_in_codeword), sizeof(pos_in_codeword));
        in.read(reinterpret_cast<char*>(posterior.data()), sizeof(double) * posterior_len);
        if (!in.good()) {
            std::cerr << "Read posterior file ended unexpectedly at record " << rec << ".\n";
            return 1;
        }

        if (pos_in_codeword < 1 || pos_in_codeword > OLIGO_NUMS) continue;

        int row = pos_in_codeword - 1;
        ++valid_records;
        if (use_llr_calibrated(consensus_cfg) &&
            consensus_cfg.max_reads_per_oligo > 0 &&
            read_counts[row] >= consensus_cfg.max_reads_per_oligo) {
            ++capped_records;
            continue;
        }

        ++read_counts[row];
        ++used_records;

        std::vector<double> lr423 = per_read_post423_to_lr423(posterior);
        if ((int)lr423.size() != EACHOLIGO_SIZE) continue;

        for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
                double lr = lr423[j];
                if (!std::isfinite(lr) || lr <= 0.0) lr = 1.0;
                if (std::fabs(lr - 1.0) < 1e-15) continue;

                double log_lr = std::log(lr);
                if (use_llr_calibrated(consensus_cfg)) {
                    log_lr = clamp_log_lr(log_lr, consensus_cfg.read_log_lr_clip);
                }
                log_sums[row][j] += log_lr;
                has_info[row][j] = 1;
        }
    }

    double extra;
    if (in.read(reinterpret_cast<char*>(&extra), sizeof(extra))) {
        std::cerr << "Warning: read posterior file contains extra data beyond expected records.\n";
    }
    in.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();

    int dropoutoligos = 0;
    int total_clusters = 0;
    for (int i = 0; i < OLIGO_NUMS; ++i) {
        if (read_counts[i] == 0) {
            ++dropoutoligos;
        } else {
            ++total_clusters;
        }
    }

    ofstream summary_file(summary_path);
    if (!summary_file.is_open()) {
        printf("\nCan't open %s\n", summary_path.c_str());
        return 1;
    }

    std::cout << "\nMultiple-copy consensus execution time: " << seconds << " s\n";
    // Consensus calibration parameters are intentionally not printed to stdout.
    summary_file << "Multiple-copy consensus execution time: " << seconds << " s\n";
    summary_file << "Posterior length: " << posterior_len << "\n";
    summary_file << "Posterior format: 423\n";
    summary_file << "Total clusters: " << total_clusters << "\n";
    summary_file << "Valid reads: " << valid_records << "\n";
    summary_file << "Total reads: " << used_records << "\n";
    summary_file << "Reads skipped by cap: " << capped_records << "\n";
    summary_file << "Dropout oligos: " << dropoutoligos << "\n";
    summary_file << "Dropout rate: " << 100.0 * dropoutoligos / OLIGO_NUMS << "%\n";
    summary_file << "Consensus mode: " << consensus_cfg.mode << "\n";
    summary_file << "Consensus final probability epsilon: " << consensus_cfg.final_prob_eps << "\n";
    summary_file << "Consensus log LR scale: " << consensus_cfg.log_lr_scale << "\n";
    summary_file << "Consensus final log LR clip: " << consensus_cfg.final_log_lr_clip << "\n";
    summary_file << "Consensus per-read log LR clip: " << consensus_cfg.read_log_lr_clip << "\n";
    summary_file << "Consensus max reads per oligo: " << consensus_cfg.max_reads_per_oligo << "\n";

    FILE* file2 = fopen(llr_save_path.c_str(), "wb");
    if (file2 == NULL) {
        printf("\ncan not open %s\n", llr_save_path.c_str());
        return 1;
    }

    std::ofstream hard_bits_file(hard_bits_path);
    if (!hard_bits_file.is_open()) {
        std::cerr << "Can't open " << hard_bits_path << "\n";
        fclose(file2);
        return 1;
    }

    int codewordnums = OLIGO_NUMS / 153;
    for (int i = 0; i < codewordnums; ++i) {
        std::string bit_line;
        bit_line.reserve(153 * EACHOLIGO_SIZE);

        for (int k = 0; k < 153; ++k) {
            int idx = k + i * 153;
            for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
                double v = merged_lr_from_log_sum(log_sums[idx][j], has_info[idx][j] != 0, consensus_cfg);

                size_t nw = fwrite(&v, sizeof(double), 1, file2);
                if (nw != 1) {
                    printf("\nError: failed to write binary LLR at codeword %d, oligo %d, bit %d\n", i, k, j);
                    fclose(file2);
                    return 1;
                }

                if (fabs(v - 1.0) < 1e-12)
                    bit_line.push_back('e');
                else if (v > 1.0)
                    bit_line.push_back('1');
                else
                    bit_line.push_back('0');
            }
        }

        hard_bits_file << bit_line << '\n';
    }

    fclose(file2);
    hard_bits_file.close();
    summary_file.close();

    return 0;
}
