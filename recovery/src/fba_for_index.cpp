#include "fba_for_index.h"
#include "parameters.h"

struct RFZOut {
    vector<vector<double>> rho;   // T x q
    vector<vector<double>> f;     // q x q
    vector<vector<double>> zeta;  // T x q
};

// ====================== rfz: compute rho, f, zeta ======================
const RFZOut rfz(const vector<int>& mp1, const vector<int>& mp2, double ps, int T) {
    RFZOut out;
    int q = 4;
    out.rho.assign(T, vector<double>(q, 0.0));

    for (int k = 0; k < T; ++k) {
        vector<double> rhok(4, 0.0);
        if (mp1[k] == 0 && mp2[k] == 0) {
            rhok = { 1,0,0,0 };
        }
        else if (mp1[k] == 0 && mp2[k] == 1) {
            rhok = { 0,1,0,0 };
        }
        else if (mp1[k] == 1 && mp2[k] == 0) {
            rhok = { 0,0,1,0 };
        }
        else if (mp1[k] == 1 && mp2[k] == 1) {
            rhok = { 0,0,0,1 };
        }
        else if (mp1[k] == 0 && mp2[k] == -1) {
            rhok = { 0.5,0.5,0,0 };
        }
        else if (mp1[k] == 1 && mp2[k] == -1) {
            rhok = { 0,0,0.5,0.5 };
        }
        else if (mp1[k] == -1 && mp2[k] == 0) {
            rhok = { 0.5,0,0.5,0 };
        }
        else if (mp1[k] == -1 && mp2[k] == 1) {
            rhok = { 0,0.5,0,0.5 };
        }
        else {
            rhok = { 0.25,0.25,0.25,0.25 };
        }
        out.rho[k] = rhok;
    }

    out.f.assign(q, vector<double>(q, ps / (q - 1)));
    for (int i = 0; i < q; i++) out.f[i][i] = 1.0 - ps;

    out.zeta.assign(T, vector<double>(q, 0.0));
    for (int j = 0; j < T; j++) {
        for (int i = 0; i < q; i++) {
            double sumv = 0.0;
            for (int iprim = 0; iprim < q; iprim++) {
                sumv += out.rho[j][iprim] * out.f[iprim][i];
            }
            out.zeta[j][i] = sumv;
        }
    }
    return out;
}


const vector<double> FB_decode(const vector<int>& y, int T,
    const vector<vector<double>>& mu,
    const vector<vector<double>>& rho,
    const vector<vector<double>>& f,
    const vector<vector<double>>& zeta,
    const vector<int>& mp1, const vector<int>& mp2,
    const vector<double>& log_map_vec, double delta_step, int l_max) {
    double Mval = 1e7;

    int R = (int)y.size();
    int q = 4;

    int ko = 1, no = 2 + l_max; 

    vector<double> q_pow(l_max + 1, 1.0);
    for (int l = 0; l <= l_max; ++l) {
        q_pow[l] = pow((double)q, -l);
    }


    vector<vector<double>> mu_q(l_max + 1, vector<double>(2, 0.0));
    for (int l = 0; l <= l_max; ++l) {
        for (int b = 0; b <= 1; ++b) {
            mu_q[l][b] = mu[l][b] * q_pow[l];
        }
    }


    vector<vector<double>> log_alf(T + 1 + ko, vector<double>(R + 2 * l_max + 3 + no, -Mval));
    vector<vector<double>> log_bet(T + 1 + ko, vector<double>(R + 2 * l_max + 3 + no, -Mval));

    double mu00 = mu[0][0];

    // initialize alfa
    log_alf[0 + ko][0 + no] = 0.0;
    if (mu00 == 0.0) {
        for (int k = 1; k <= T; k++) log_alf[k + ko][0 + no] = -Mval;
    }
    else {
        double log_mu00 = log(mu00); 
        for (int k = 1; k <= T; k++) log_alf[k + ko][0 + no] = log_mu00 * (double)k;
    }

    // initialize beta
    log_bet[T + ko][R + no] = 0.0;
    if (mu00 > 0.0) {
        for (int k = 0; k <= T - 1; k++) log_bet[k + ko][R + no] = 0.0;
    }
    else {
        for (int k = 0; k <= T - 1; k++) log_bet[k + ko][R + no] = -Mval;
    }

    // ================= recursive equations for alfa =================
    for (int k = 1; k <= T; k++) {
        for (int n = 1; n <= R; n++) {
            for (int l = 0; l <= l_max; l++) {
                for (int b = 0; b <= 1; b++) {

                    double z = 1.0;
                    if (b == 1) z = zeta[k - 1][y[n - 1]]; 

                    double coef1 = mu_q[l][b] * (b ? z : 1.0);
                    double calf = -Mval;
                    int nprev = n - l - b;
                    if (coef1 != 0.0 && (0 <= nprev)) {
                        calf = log(coef1) + log_alf[(k - 1) + ko][nprev + no];
                    }
                    double tca = fabs(calf - log_alf[k + ko][n + no]);
                    int tca_loc = min((int)log_map_vec.size(), 1 + (int)floor(tca / delta_step));
                    log_alf[k + ko][n + no] = max(log_alf[k + ko][n + no], calf) + log_map_vec[tca_loc - 1];
                }
            }
        }
    }

    // ================= recursive equations for beta =================
    for (int k = T - 1; k >= 0; k--) {
        for (int n = R - 1; n >= 0; n--) {
            for (int l = 0; l <= l_max; l++) {
                for (int b = 0; b <= 1; b++) {
                    int idxy = min(R, n + l + 1); 
                    int yidx = max(1, idxy);       
                    double zky = zeta[k][y[yidx - 1]]; 
                    double coef1 = mu_q[l][b] * (b ? zky : 1.0);
                    double cbet = -Mval;
                    int nnext = n + l + b;
                    if (coef1 != 0.0) {
                        cbet = log(coef1) + log_bet[(k + 1) + ko][nnext + no];
                    }
                    double tcb = fabs(cbet - log_bet[k + ko][n + no]);
                    int tcb_loc = min((int)log_map_vec.size(), 1 + (int)floor(tcb / delta_step));
                    log_bet[k + ko][n + no] = max(log_bet[k + ko][n + no], cbet) + log_map_vec[tcb_loc - 1];
                }
            }
        }
    }

    // ===== Run forward-backward decoding and find a posteriori probabilities =====
    vector<vector<double>> log_ap_prob(T, vector<double>(q, -Mval));
    vector<vector<double>> joint_prob(T, vector<double>(q, 0.0));
    for (int k = 1; k <= T; k++) {
        for (int a = 0; a < q; a++) {
            for (int l = 0; l <= l_max; l++) {
                for (int b = 0; b <= 1; b++) {
                    int Upper_val = min(R, (l_max + 1) * (k - 1));
                    for (int n = 0; n <= Upper_val; n++) {
                        int idxy = min(R, n + l + 1);
                        int yidx = max(1, idxy);
                        double fbb = 1.0;
                        if (b == 1) fbb = f[a][y[yidx - 1]];
                        double coef1 = mu_q[l][b] * fbb;
                        double log_add_term = -Mval;
                        if (coef1 != 0.0) {
                            log_add_term = log(coef1) + log_alf[(k - 1) + ko][n + no]
                                + log_bet[k + ko][(n + l + b) + no];
                        }
                        double dpa = fabs(log_add_term - log_ap_prob[k - 1][a]);
                        int dpa_loc = min((int)log_map_vec.size(), 1 + (int)floor(dpa / delta_step));
                        log_ap_prob[k - 1][a] = max(log_ap_prob[k - 1][a], log_add_term) + log_map_vec[dpa_loc - 1];
                    }
                }
            }
        }
    }

    double mmap = -1e300;
    for (int k = 0; k < T; k++) for (int a = 0; a < q; a++) mmap = max(mmap, log_ap_prob[k][a]);

    for (int k = 0; k < T; k++) {
        for (int a = 0; a < q; a++) {
            joint_prob[k][a] = rho[k][a] * exp(log_ap_prob[k][a] - mmap);
        }
    }

    vector<vector<double>> bit_prob1(2, vector<double>(T, 0.0));
    vector<vector<double>> bit_prob2(2, vector<double>(T, 0.0));

    for (int k = 0; k < T; k++) {
        bit_prob1[0][k] = joint_prob[k][0] + joint_prob[k][1];
        bit_prob1[1][k] = joint_prob[k][2] + joint_prob[k][3];

        bit_prob2[0][k] = joint_prob[k][0] + joint_prob[k][2];
        bit_prob2[1][k] = joint_prob[k][1] + joint_prob[k][3];
    }

    vector<int> j1, j2;
    for (int i = 0; i < T; i++) if (mp1[i] == -1) j1.push_back(i);
    for (int i = 0; i < T; i++) if (mp2[i] == -1) j2.push_back(i);

    vector<double> bit_llrs;
    bit_llrs.reserve(j1.size() + j2.size());

    for (int idx : j1) {
        double num = bit_prob1[1][idx];
        double den = bit_prob1[0][idx];
        bit_llrs.push_back(log(max(num, 1e-300) / max(den, 1e-300)));
    }
    for (int idx : j2) {
        double num = bit_prob2[1][idx];
        double den = bit_prob2[0][idx];
        bit_llrs.push_back(log(max(num, 1e-300) / max(den, 1e-300)));
    }

    vector<double> p_ub_1(bit_llrs.size(), 0.0);
    for (size_t i = 0; i < bit_llrs.size(); ++i) {
        double e = exp(bit_llrs[i]);
        p_ub_1[i] = e / (1.0 + e);
    }
    return p_ub_1;
}


const void init(int T,vector<vector<double>>& mu, vector<vector<double>>& rho, vector<vector<double>>& f,
    vector<vector<double>>& zeta, vector<int>& mp1, vector<int>& mp2, vector<double>& log_map_vec, 
    double delta_step, int l_max, string marker_str) {

    for (double x = 0.0; x <= 10.0 + 1e-12; x += delta_step) {
        log_map_vec.push_back(log(1.0 + exp(-x)));
    }

    double pi = prob_ins;
    double pd = prob_del;
    double ps = prob_sub;

    if (pi == 0.0) 
        l_max = 0;

    vector<double> pil_vec(l_max + 1, 1.0);
    for (int i = 0; i <= l_max; i++) pil_vec[i] = pow(pi, i);


    for (int i = 0; i <= l_max; i++) {
        mu[i][0] = pd * pil_vec[i];
        mu[i][1] = (1.0 - pd) * pil_vec[i];
    }
    double s = 0.0;
    for (int i = 0; i <= l_max; i++) {
        s += mu[i][0];
        s += mu[i][1];
    }
    if (s > 0) {
        for (int i = 0; i <= l_max; i++) {
            mu[i][0] /= s;
            mu[i][1] /= s;
        }
    }

    int Np = 1;

    vector<int> marker_positions;
    for (int i = 0; i < T; i += Np)
        marker_positions.push_back(i); 
    vector<int> marker(marker_str.size(), 0);

    for (size_t i = 0; i < marker_str.size(); ++i)
        marker[i] = (marker_str[i] - '0');

    // Ensure marker length matches requirement
    if (marker.size() < marker_positions.size()) {
        printf("Marker length is insufficient for the required marker positions.\n");
        return;
    }
    for (size_t i = 0; i < marker_positions.size(); ++i)
        mp1[marker_positions[i]] = marker[i];

    RFZOut rfzout = rfz(mp1, mp2, ps, T);
    rho = rfzout.rho;
    f = rfzout.f;
    zeta = rfzout.zeta;

    int data_length = 0;
    for (int i = 0; i < T; i++) if (mp1[i] == -1) data_length++;
    for (int i = 0; i < T; i++) if (mp2[i] == -1) data_length++;
}


void FBA_with_PN_pilots(int T, string marker_str, string indel_base_sequence, vector<double>& llr_1) {

    int l_max = 2;

    vector<vector<double>> mu(l_max + 1, vector<double>(2, 0.0));
    vector<vector<double>> rho;
    vector<vector<double>> f;
    vector<vector<double>> zeta;
    vector<int> mp1(T, -1);
    vector<int> mp2(T, -1);
    vector<double> log_map_vec;
    double delta_step = 0.01;

    init(T, mu, rho, f, zeta, mp1, mp2, log_map_vec, delta_step, l_max, marker_str);

    vector<int> y;
    for (size_t i = 0; i < indel_base_sequence.length(); i++) {
        if (indel_base_sequence[i] == 'A') {
            y.push_back(0); // 00
        }
        else if (indel_base_sequence[i] == 'T') {
            y.push_back(1); // 01
        }
        else if (indel_base_sequence[i] == 'G') {
            y.push_back(2); // 10
        }
        else if (indel_base_sequence[i] == 'C') {
            y.push_back(3); // 11
        }
    }

    llr_1 = FB_decode(y, T, mu, rho, f, zeta, mp1, mp2, log_map_vec, delta_step, l_max);

}
