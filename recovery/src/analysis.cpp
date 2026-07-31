#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

using namespace std;

namespace {

const int BLOCK_LENGTH = 235;
const int UPPER_INFO_LENGTH = 188;
const int EACHOLIGO_SIZE = UPPER_INFO_LENGTH + BLOCK_LENGTH;
const int OLIGOS_PER_CODEWORD = 153;
const int CODEWORD_COUNT = 196;
const int CODEWORD_BIT_LENGTH = OLIGOS_PER_CODEWORD * EACHOLIGO_SIZE;

static string trim_and_keep_valid_chars(const string& s, bool allow_e) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '0' || c == '1' || (allow_e && (c == 'e' || c == 'E'))) {
            out.push_back((c == 'E') ? 'e' : c);
        }
    }
    return out;
}

static vector<string> read_bit_lines(const string& path, bool allow_e) {
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "Cannot open file: " << path << endl;
        exit(1);
    }

    vector<string> lines;
    string line;
    while (getline(fin, line)) {
        string cleaned = trim_and_keep_valid_chars(line, allow_e);
        if (!cleaned.empty()) {
            lines.push_back(cleaned);
        }
    }
    fin.close();
    return lines;
}

static string read_first_bit_line(const string& path, bool allow_e) {
    vector<string> lines = read_bit_lines(path, allow_e);
    if (lines.empty()) {
        cerr << "Error: bit file is empty: " << path << endl;
        exit(1);
    }
    return lines[0];
}

static bool is_bit(char c) {
    return c == '0' || c == '1';
}

static char bit_at_or_e(const string& s, size_t idx) {
    if (idx >= s.size()) return 'e';
    char c = s[idx];
    return is_bit(c) ? c : 'e';
}

static int count_dropout_oligos(const string& pred) {
    int dropout = 0;
    for (int k = 0; k < OLIGOS_PER_CODEWORD; ++k) {
        size_t base = static_cast<size_t>(k) * EACHOLIGO_SIZE;
        bool any_present = false;
        for (int j = 0; j < EACHOLIGO_SIZE; ++j) {
            if (is_bit(bit_at_or_e(pred, base + j))) {
                any_present = true;
                break;
            }
        }
        if (!any_present) ++dropout;
    }
    return dropout;
}

static double substitution_rate(long long sub, long long era, long long total_bits) {
    long long non_erased = total_bits - era;
    if (non_erased <= 0) return 0.0;
    return static_cast<double>(sub) / static_cast<double>(non_erased);
}

static double erasure_rate(long long era, long long total_bits) {
    if (total_bits <= 0) return 0.0;
    return static_cast<double>(era) / static_cast<double>(total_bits);
}

static void count_codeword_errors(
    const string& ref,
    const string& pred,
    long long& sub,
    long long& era,
    long long& upper_sub,
    long long& lower_sub
) {
    sub = 0;
    era = 0;
    upper_sub = 0;
    lower_sub = 0;

    size_t cmp_len = min(ref.size(), pred.size());
    for (size_t j = 0; j < cmp_len; ++j) {
        char r = ref[j];
        char p = pred[j];

        if (!is_bit(p)) {
            ++era;
            continue;
        }
        if (p != r) {
            ++sub;
            int pos_in_oligo = static_cast<int>(j % EACHOLIGO_SIZE);
            if (pos_in_oligo < UPPER_INFO_LENGTH) {
                ++upper_sub;
            } else {
                ++lower_sub;
            }
        }
    }

}

static string sparse45_encode_upper188(const string& upper188) {
    static const char* sparse_map[16] = {
        "00000", "00001", "00010", "00011",
        "00100", "00101", "00110", "11000",
        "01000", "01001", "01010", "10100",
        "01100", "10010", "10001", "10000"
    };

    string upper235;
    upper235.reserve(BLOCK_LENGTH);
    for (int start = 0; start < UPPER_INFO_LENGTH; start += 4) {
        int idx = 0;
        bool ok = true;
        for (int j = 0; j < 4; ++j) {
            char b = bit_at_or_e(upper188, static_cast<size_t>(start + j));
            if (!is_bit(b)) {
                ok = false;
                break;
            }
            idx = (idx << 1) | (b == '1' ? 1 : 0);
        }

        if (ok) {
            upper235 += sparse_map[idx];
        } else {
            upper235 += "eeeee";
        }
    }
    return upper235;
}

static char xor_watermark_bit(char sparse_bit, char watermark_bit) {
    if (!is_bit(sparse_bit) || !is_bit(watermark_bit)) return 'e';
    return (sparse_bit == watermark_bit) ? '0' : '1';
}

static char bits_to_base(char upper, char lower) {
    if (!is_bit(upper) || !is_bit(lower)) return 'N';
    if (upper == '0' && lower == '0') return 'A';
    if (upper == '0' && lower == '1') return 'T';
    if (upper == '1' && lower == '0') return 'G';
    return 'C';
}

static string hard_base_from_oligo_bits(const string& oligo_bits, const string& watermark235) {
    string upper188;
    upper188.reserve(UPPER_INFO_LENGTH);
    for (int i = 0; i < UPPER_INFO_LENGTH; ++i) {
        upper188.push_back(bit_at_or_e(oligo_bits, i));
    }

    string lower235;
    lower235.reserve(BLOCK_LENGTH);
    for (int i = 0; i < BLOCK_LENGTH; ++i) {
        lower235.push_back(bit_at_or_e(oligo_bits, UPPER_INFO_LENGTH + i));
    }

    string upper_sparse235 = sparse45_encode_upper188(upper188);
    string bases;
    bases.reserve(BLOCK_LENGTH);
    for (int i = 0; i < BLOCK_LENGTH; ++i) {
        char upper = xor_watermark_bit(
            bit_at_or_e(upper_sparse235, i),
            bit_at_or_e(watermark235, i)
        );
        char lower = bit_at_or_e(lower235, i);
        bases.push_back(bits_to_base(upper, lower));
    }
    return bases;
}

static void write_fasta_record(ofstream& out, const string& name, const string& seq) {
    out << ">" << name << "\n";
    const size_t wrap = 80;
    for (size_t i = 0; i < seq.size(); i += wrap) {
        out << seq.substr(i, wrap) << "\n";
    }
}

static void write_hard_base_fasta(
    const string& fasta_path,
    const vector<string>& pred_lines,
    const string& watermark235
) {
    ofstream fasta(fasta_path);
    if (!fasta.is_open()) {
        cerr << "Cannot open output file: " << fasta_path << endl;
        exit(1);
    }

    for (size_t cw = 0; cw < pred_lines.size(); ++cw) {
        const string& pred = pred_lines[cw];
        for (int k = 0; k < OLIGOS_PER_CODEWORD; ++k) {
            size_t oligo_offset = (static_cast<size_t>(k) * EACHOLIGO_SIZE);
            string oligo_bits;
            if (oligo_offset < pred.size()) {
                oligo_bits = pred.substr(oligo_offset, EACHOLIGO_SIZE);
            }
            string bases = hard_base_from_oligo_bits(oligo_bits, watermark235);
            size_t oligo_id = cw * OLIGOS_PER_CODEWORD + static_cast<size_t>(k) + 1;
            write_fasta_record(fasta, to_string(oligo_id), bases);
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " <reference_path> <save_path> <pool>" << endl;
        cerr << "Example: " << argv[0]
             << " ./reference ./result 3" << endl;
        return 1;
    }

    const string reference_path = argv[1];
    const string save_path = argv[2];
    const string pool_str = argv[3];
    const int pool = atoi(pool_str.c_str());

    if (pool < 1 || pool > 6) {
        cerr << "Error: pool must be in [1, 6], but got " << pool << endl;
        return 1;
    }

    const string encoded_cw_file =
        reference_path + "/encoded_bits/p" + pool_str + ".txt";
    const string hard_bits_path =
        save_path + "/bitstream_before_decoding.txt";
    const string summary_out =
        save_path + "/bit_error_rates.txt";
    const string per_codeword_out =
        save_path + "/bit_error_rate.txt";
    const string per_codeword_tsv_out =
        save_path + "/bit_error_rate_by_codeword.txt";
    const string hard_base_fasta_out =
        save_path + "/hard_base_from_bits.fasta";
    const string watermark_path =
        reference_path + "/../../configureFiles/watermark_sequence_length_235";

    vector<string> ref_lines = read_bit_lines(encoded_cw_file, false);
    vector<string> pred_lines = read_bit_lines(hard_bits_path, true);
    string watermark235 = read_first_bit_line(watermark_path, false);
    if (watermark235.size() < BLOCK_LENGTH) {
        cerr << "Error: watermark file contains only " << watermark235.size()
             << " bits, expected " << BLOCK_LENGTH << ": " << watermark_path << endl;
        return 1;
    }
    watermark235 = watermark235.substr(0, BLOCK_LENGTH);

    if (ref_lines.empty()) {
        cerr << "Error: reference file is empty: " << encoded_cw_file << endl;
        return 1;
    }
    if (pred_lines.empty()) {
        cerr << "Error: predicted file is empty: " << hard_bits_path << endl;
        return 1;
    }

    size_t n_compare = min(ref_lines.size(), pred_lines.size());

    ofstream fout_summary(summary_out);
    if (!fout_summary.is_open()) {
        cerr << "Cannot open output file: " << summary_out << endl;
        return 1;
    }

    ofstream fout_per_codeword(per_codeword_out);
    if (!fout_per_codeword.is_open()) {
        cerr << "Cannot open output file: " << per_codeword_out << endl;
        return 1;
    }

    ofstream fout_per_codeword_tsv(per_codeword_tsv_out);
    if (!fout_per_codeword_tsv.is_open()) {
        cerr << "Cannot open output file: " << per_codeword_tsv_out << endl;
        return 1;
    }
    fout_per_codeword_tsv
        << "codeword_1based\thard_match\tsubstitution_errors\terasure_errors\t"
        << "dropout_oligos\tupper188_substitution_errors\tlower235_substitution_errors\t"
        << "substitution_rate\terasure_rate\ttotal_error_rate\n";

    long long total_sub = 0;
    long long total_era = 0;

    for (size_t i = 0; i < n_compare; ++i) {
        const string& ref = ref_lines[i];
        const string& pred = pred_lines[i];

        long long sub = 0;
        long long era = 0;
        long long upper_sub = 0;
        long long lower_sub = 0;
        count_codeword_errors(ref, pred, sub, era, upper_sub, lower_sub);

        int dropout = count_dropout_oligos(pred);
        int hard_match = (sub == 0 && era == 0 && ref.size() == pred.size()) ? 1 : 0;

        total_sub += sub;
        total_era += era;

        fout_per_codeword
            << hard_match << " "
            << sub << " "
            << era << " "
            << dropout << " "
            << upper_sub << " "
            << lower_sub << "\n";

        double sub_rate = substitution_rate(sub, era, CODEWORD_BIT_LENGTH);
        double era_rate = erasure_rate(era, CODEWORD_BIT_LENGTH);
        double total_rate = static_cast<double>(sub + era) / static_cast<double>(CODEWORD_BIT_LENGTH);
        fout_per_codeword_tsv
            << (i + 1) << "\t"
            << hard_match << "\t"
            << sub << "\t"
            << era << "\t"
            << dropout << "\t"
            << upper_sub << "\t"
            << lower_sub << "\t"
            << setprecision(12) << sub_rate << "\t"
            << setprecision(12) << era_rate << "\t"
            << setprecision(12) << total_rate << "\n";
    }

    long long total_bits_all = static_cast<long long>(CODEWORD_COUNT) * CODEWORD_BIT_LENGTH;
    double avg_erasure_rate = erasure_rate(total_era, total_bits_all);
    double avg_substitution_rate = substitution_rate(total_sub, total_era, total_bits_all);
    double avg_total_error_rate =
        static_cast<double>(total_sub + total_era) / static_cast<double>(total_bits_all);

    fout_summary << "Pool: " << pool << "\n";
    fout_summary << "Average substitution error rate: " << avg_substitution_rate << "\n";
    fout_summary << "Average erasures rate: " << avg_erasure_rate << "\n";
    fout_summary << "Average total error rate: " << avg_total_error_rate << "\n";
    fout_summary.close();
    fout_per_codeword.close();
    fout_per_codeword_tsv.close();

    write_hard_base_fasta(hard_base_fasta_out, pred_lines, watermark235);

    cout << "Pool: " << pool << endl;
    cout << "Average substitution error rate: " << avg_substitution_rate << endl;
    cout << "Average erasures rate: " << avg_erasure_rate << endl;
    // cout << "Average total error rate: " << avg_total_error_rate << endl;
    // cout << "Per-codeword bit errors: " << per_codeword_out << endl;
    // cout << "Per-codeword BER table: " << per_codeword_tsv_out << endl;
    // cout << "Hard base FASTA: " << hard_base_fasta_out << endl;

    return 0;
}
