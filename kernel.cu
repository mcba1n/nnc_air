
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <stdio.h>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

#include "csv.h"

#define SQRT1_2   0.707106781
#define SQRT1_2PI 0.398942280

// GFA
__host__ double gfa_run(int m, int Ns, int Nd, int max_int_len, double* y, double* P, double* P_K, int* intervals, double sigma, double* f);
__global__ void gfa_slice(int ell, int Ns, int Nd, int max_int_len, double* y, double* Fp, double* Fc,
    double* P, double* P_K, int* intervals, double sigma, double* f);
__global__ void gfa_init(int Ns, int Nd, int max_int_len, double* y, double* F,
    double* P, double* P_K, double sigma, double* f, int s_0);
__global__ void gfa_termination(int m, int Ns, int max_int_len, double* F, int* intervals, double* log_post_vec);

// GCA
__host__ double gca_run(int m, int* s_seq, int Nd, int max_int_len, double* y,
    double* P_K, int* intervals, double sigma, double* f);
__global__ void gca_slice(int ell, int s, int Nd, int max_int_len, double* y,
    double* Cp, double* Cc, double* P_K, int* intervals, double sigma, double* f);
__global__ void gca_init(int Nd, int max_int_len, double* y, double* C,
    double* P_K, double sigma, double* f, int s, int s_0);
__global__ void gca_termination(int m, int max_int_len, double* C, int* intervals, double* log_post_vec);

// Helpers
__host__ __device__ double eln(double x);
__host__ __device__ double eexp(double x);
__host__ __device__ double elnproduct(double x, double y);
__host__ __device__ double elnsum(double x, double y);

__host__ __device__ double normalPDF(double value, double mu, double sigma);
__host__ thrust::host_vector<int> bd_int(int m, int W, int T, double E_K);
std::vector<std::vector<double>> read_data(std::string fname, bool f);


int main()
{
    //printf("result: %.4f\n", elnsum(eln(0), -60));

    std::cout << "air_sim3" << std::endl;
    std::default_random_engine generator;
    //std::srand(std::time(NULL));

    std::ofstream datafile;
    datafile.open("model_biawgn/data.csv");
    //data_genie_int_rad0

    // Markov probabilities
    std::vector<std::vector<double>> P = read_data("model_biawgn/P.csv", 0);
    int Ns = P[0].size();
    thrust::host_vector<double> Ph(Ns * Ns);
    for (int i = 0; i < Ns; i++) for (int j = 0; j < Ns; j++) Ph[i * Ns + j] = P[i][j];

    // Channel mapping
    std::vector<std::vector<double>> f = read_data("model_biawgn/f.csv", 0);
    thrust::host_vector<double> fh(Ns);
    for (int i = 0; i < Ns; i++) fh[i] = f[i][0];

    // Duration probabilities
    std::vector<std::vector<double>> P_K = read_data("model_biawgn/P_K.csv", 0);
    int k_max = P_K.size();
    thrust::host_vector<double> P_Kh(k_max);
    for (int i = 0; i < k_max; i++) P_Kh[i] = P_K[i][0] + std::numeric_limits<double>::epsilon();
    double E_K = 0;
    for (int k = 0; k < k_max; k++) E_K += (k + 1) * P_Kh[k];
    int W = 0;
    for (int i = 0; i < k_max; i++) if (P_Kh[i] > std::numeric_limits<double>::epsilon()) W += 1;

    // Run simulations
    const double sigma_min = std::numeric_limits<double>::epsilon();
    const double sigma_max = 1.0;
    const double delta = 0.05;
    const int N_dps = round((sigma_max - sigma_min) / delta) + 1;

    std::vector<double> sigma_vals(N_dps);
    std::vector<double> air_vals(N_dps);

    for (int sim_idx = 0; sim_idx < N_dps; sim_idx++) {
        // Generator random signal
        double sigma = sigma_min + sim_idx * delta;
        const int m = 100000;
        const int s_0 = 0;
        thrust::host_vector<double> y;
        thrust::host_vector<int> s_seq(m);
        thrust::host_vector<int> t_seq(m);
        int s_prev = s_0;

        for (int i = 0; i < m; i++) {
            int k = std::discrete_distribution<uint64_t>(P_Kh.begin(), P_Kh.end())(generator) + 1;
            int s = std::discrete_distribution<uint64_t>(P[s_prev].begin(), P[s_prev].end())(generator);
            std::normal_distribution<double> norm_dist(fh[s], sigma);
            for (int j = 0; j < k; j++) y.push_back(norm_dist(generator));
            s_seq[i] = s;
            s_prev = s;
            if (i == 0) t_seq[0] = k;
            else t_seq[i] = t_seq[i - 1] + k;
        }

        // Intervals
        int T = y.size();
        thrust::host_vector<int> intervals(m * 2);
        intervals = bd_int(m, W, T, E_K);
        //intervals = bd_int_markers(m, T, t_seq);
        std::vector<int> int_lens(m);
        for (int i = 0; i < m; i++) int_lens[i] = intervals[i * 2 + 1] - intervals[i * 2] + 1;
        int max_int_len = *max_element(int_lens.begin(), int_lens.end());

         
        // Convert host vectors to device vectors
        thrust::device_vector<double> Pd(Ns * Ns);
        Pd = Ph;
        thrust::device_vector<double> fd(Ns);
        fd = fh;
        thrust::device_vector<double> P_Kd(k_max);
        P_Kd = P_Kh;
        thrust::device_vector<int> intervalsd(m * 2);
        intervalsd = intervals;
        thrust::device_vector<double> yd;
        yd = y;
        thrust::device_vector<int> s_seqd(m);
        s_seqd = s_seq;

        // Forward probabilities
        double log_post = gfa_run(m, Ns, k_max, max_int_len, thrust::raw_pointer_cast(yd.data()), thrust::raw_pointer_cast(Pd.data()),
            thrust::raw_pointer_cast(P_Kd.data()), thrust::raw_pointer_cast(intervalsd.data()), sigma, thrust::raw_pointer_cast(fd.data()));
        //printf("H_Y: %.4f\n", -log_post/(log(2)*m));
        
        // Conditional probailities
        double log_post_cond = gca_run(m, thrust::raw_pointer_cast(s_seq.data()), k_max, max_int_len, thrust::raw_pointer_cast(yd.data()),
            thrust::raw_pointer_cast(P_Kd.data()), thrust::raw_pointer_cast(intervalsd.data()), sigma, thrust::raw_pointer_cast(fd.data()));
        //printf("H_YS: %.4f\n", -log_post_cond / (log(2) * m));

        // Information rates
        double H_Y = -log_post / (log(2) * m);
        double H_Y_S = -log_post_cond / (log(2) * m);
        double I_nnc = H_Y - H_Y_S;

        // Record data
        air_vals[sim_idx] = I_nnc;
        sigma_vals[sim_idx] = sigma;
        datafile << std::to_string(sigma) << "," << std::to_string(I_nnc) << "\n";
        printf("sigma: %.4f, I: %.4f\n", sigma, I_nnc);
    }

    return 0;
}


// GFA:

__global__ void gfa_init(int Ns, int Nd, int max_int_len, double* y, double* F,
    double* P, double* P_K, double sigma, double* f, int s_0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / max_int_len;
    int t_shifted = idx % max_int_len;

    double log_gamma = elnproduct(eln(P[s_0 * Ns + s]), eln(P_K[t_shifted]));
    for (int j = 0; j < t_shifted; j++) log_gamma = elnproduct(log_gamma, eln(normalPDF(y[j], f[s], sigma)));

    int F_idx = (s * max_int_len) + t_shifted;
    F[F_idx] = log_gamma;
}

__global__ void gfa_termination(int m, int Ns, int max_int_len, double* F, int* intervals, double* log_post_vec) {
    // does termination after ell=m
    int last_shifted_idx = intervals[(m - 1) * 2 + 1] - intervals[(m - 1) * 2];
    for (int s = 0; s < Ns; s++) {
        int F_idx = (s * max_int_len) + last_shifted_idx;
        log_post_vec[0] = elnsum(log_post_vec[0], F[F_idx]);
    }
}

__host__ double gfa_run(int m, int Ns, int Nd, int max_int_len, double* y, double* P, double* P_K, int* intervals, double sigma, double* f) {
    //printf("Running GFA...\n");
    cudaError_t cudaStatus;
    int N = max_int_len * Ns;
    int THREADS_PER_BLOCK = 800;
    int NUM_BLOCKS = N / THREADS_PER_BLOCK + 1;

    thrust::host_vector<double> Fhp(Ns * max_int_len, eln(0));
    thrust::host_vector<double> Fhc(Ns * max_int_len, eln(0));
    thrust::device_vector<double> Fdp(Ns * max_int_len, eln(0));
    thrust::device_vector<double> Fdc(Ns * max_int_len, eln(0));
    
    // Init
    gfa_init<<<NUM_BLOCKS, THREADS_PER_BLOCK >>>(Ns, Nd, max_int_len, y, thrust::raw_pointer_cast(Fdp.data()), P, P_K, sigma, f, 0);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
    // Slices
    for (int ell = 2; ell <= m; ell++) { 
        gfa_slice << <NUM_BLOCKS, THREADS_PER_BLOCK>> > (ell, Ns, Nd, max_int_len, y, thrust::raw_pointer_cast(Fdp.data()), 
            thrust::raw_pointer_cast(Fdc.data()), P, P_K, intervals, sigma, f);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        
        /*
        if (ell == m) {
            printf("DATA1:\n");
            Fhc = Fdc;
            for (int t = 0; t < max_int_len; t++) {
                for (int s = 0; s < Ns; s++) {
                    int F_idx = (s * max_int_len) + t;
                    printf("(s: %d, t: %d): %.4f\n", s, t, Fhc[F_idx]);
                }
            }
            printf("DATA2:\n");
            Fhp = Fdp;
            for (int t = 0; t < max_int_len; t++) {
                for (int s = 0; s < Ns; s++) {
                    int F_idx = (s * max_int_len) + t;
                    printf("(s: %d, t: %d): %.4f\n", s, t, Fhp[F_idx]);
                }
            }
        }*/

        // Shift slices in F
        thrust::copy(thrust::device, Fdc.begin(), Fdc.end(), Fdp.begin());
        thrust::fill(thrust::device, Fdc.begin(), Fdc.end(), eln(0));

        printf("GFA, ell = %d\n", ell);
    }

    // Termination
    thrust::device_vector<double> log_post_d(1, eln(0));
    thrust::host_vector<double> log_post_h(1, eln(0));
    gfa_termination<<<1,1>>>(m, Ns, max_int_len, thrust::raw_pointer_cast(Fdp.data()), intervals, thrust::raw_pointer_cast(log_post_d.data()));
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    log_post_h = log_post_d;
    return log_post_h[0];
}


__global__ void gfa_slice(int ell, int Ns, int Nd, int max_int_len, double* y, double* Fp, double* Fc, 
    double* P, double* P_K, int* intervals, double sigma, double* f) {
    //NOTE: t should be in I_{ell}, t_prev should be in I_{ell-1}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / max_int_len;
    int t_shifted = idx % max_int_len;
    int t = intervals[(ell - 1) * 2] + t_shifted; 

    int max_int_len_ell_curr = intervals[(ell - 1) * 2 + 1] - intervals[(ell - 1) * 2] + 1;
    int max_int_len_ell_prev = intervals[(ell - 2) * 2 + 1] - intervals[(ell - 2) * 2] + 1;
    if (t_shifted > max_int_len_ell_curr - 1) return; // GPU should not run this kernel if t is outside of its interval

    for (int s_prev = 0; s_prev < Ns; s_prev++) {
        if (P[s_prev * Ns + s] == 0.0) continue;
        for (int k = 0; k < Nd; k++) {
            int t_start = t - k;
            int t_prev = t_start - 1;

            int t_prev_shifted = t_prev - intervals[(ell - 2) * 2]; // the segment starts here, 1 sample after t_prev; this must be in the interval
            if (t_prev_shifted < 0 || t_prev_shifted > max_int_len_ell_prev - 1) continue; // skip t_prev if it is not inside its interval

            double log_gamma = elnproduct(eln(P[s_prev * Ns + s]), eln(P_K[k]));
            for (int tt = t_start; tt <= t; tt++) log_gamma = elnproduct(log_gamma, eln(normalPDF(y[tt], f[s], sigma)));

            int F_prev_idx = (s_prev * max_int_len) + t_prev_shifted;
            int F_idx = (s * max_int_len) + t_shifted;
            Fc[F_idx] = elnsum(Fc[F_idx], elnproduct(log_gamma, Fp[F_prev_idx]));
        }
    }
}


// GCA:
__global__ void gca_init(int Nd, int max_int_len, double* y, double* C,
    double* P_K, double sigma, double* f, int s, int s_0) {
    int t_shifted = blockIdx.x * blockDim.x + threadIdx.x;

    double log_gamma = eln(P_K[t_shifted]);
    for (int j = 0; j < t_shifted; j++) log_gamma = elnproduct(log_gamma, eln(normalPDF(y[j], f[s], sigma)));

    int C_idx = t_shifted;
    C[C_idx] = log_gamma;
}

__global__ void gca_termination(int m, int max_int_len, double* C, int* intervals, double* log_post_vec) {
    int last_shifted_idx = intervals[(m - 1) * 2 + 1] - intervals[(m - 1) * 2];
    int C_idx = last_shifted_idx;
    log_post_vec[0] = elnsum(log_post_vec[0], C[C_idx]);
}

__host__ double gca_run(int m, int* s_seq, int Nd, int max_int_len, double* y,
    double* P_K, int* intervals, double sigma, double* f) {
    //printf("Running GCA...\n");
    cudaError_t cudaStatus;
    int N = max_int_len;
    int THREADS_PER_BLOCK = 800;
    int NUM_BLOCKS = N / THREADS_PER_BLOCK + 1;

    thrust::host_vector<double> Fhp(max_int_len, eln(0));
    thrust::host_vector<double> Fhc(max_int_len, eln(0));
    thrust::device_vector<double> Fdp(max_int_len, eln(0));
    thrust::device_vector<double> Fdc(max_int_len, eln(0));

    // Init
    gca_init << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (Nd, max_int_len, y, thrust::raw_pointer_cast(Fdp.data()), P_K, sigma, f, s_seq[0], 0);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
    // Slices
    for (int ell = 2; ell <= m; ell++) {
        gca_slice << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (ell, s_seq[ell-1], Nd, max_int_len, y, thrust::raw_pointer_cast(Fdp.data()),
            thrust::raw_pointer_cast(Fdc.data()), P_K, intervals, sigma, f);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
        // Shift slices in F
        thrust::copy(thrust::device, Fdc.begin(), Fdc.end(), Fdp.begin());
        thrust::fill(thrust::device, Fdc.begin(), Fdc.end(), eln(0));

        printf("GCA, ell = %d\n", ell);
    }
    
    // Termination
    thrust::device_vector<double> log_post_d(1, eln(0));
    thrust::host_vector<double> log_post_h(1, eln(0));
    gca_termination << <1, 1 >> > (m, max_int_len, thrust::raw_pointer_cast(Fdp.data()), intervals, thrust::raw_pointer_cast(log_post_d.data()));
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    log_post_h = log_post_d;
    return log_post_h[0];
}


__global__ void gca_slice(int ell, int s, int Nd, int max_int_len, double* y,
    double* Cp, double* Cc, double* P_K, int* intervals, double sigma, double* f) {
    int t_shifted = blockIdx.x * blockDim.x + threadIdx.x;
    int t = intervals[(ell - 1) * 2] + t_shifted;

    int max_int_len_ell_curr = intervals[(ell - 1) * 2 + 1] - intervals[(ell - 1) * 2] + 1;
    int max_int_len_ell_prev = intervals[(ell - 2) * 2 + 1] - intervals[(ell - 2) * 2] + 1;
    if (t_shifted > max_int_len_ell_curr - 1) return;

    for (int k = 0; k < Nd; k++) {
        int t_start = t - k;
        int t_prev = t_start - 1;

        int t_prev_shifted = t_prev - intervals[(ell - 2) * 2];
        if (t_prev_shifted < 0 || t_prev_shifted > max_int_len_ell_prev - 1) continue;

        double log_gamma = eln(P_K[k]);
        for (int tt = t_start; tt <= t; tt++) log_gamma = elnproduct(log_gamma, eln(normalPDF(y[tt], f[s], sigma)));

        int C_prev_idx = t_prev_shifted;
        int C_idx = t_shifted;
        Cc[C_idx] = elnsum(Cc[C_idx], elnproduct(log_gamma, Cp[C_prev_idx]));
    }
}



// HELPER FUNCTIONS:

__host__ __device__ double elnsum(double x, double y) {
    double z;
    if (x == -INFINITY || y == -INFINITY) {
        if (x == -INFINITY) z = y;
        else z = x;
    }
    else {
        if (x > y) z = x + log1p(exp(y - x));
        else z = y + log1p(exp(x - y));
    }
    return z;
}

__host__ __device__ double normalPDF(double value, double mu, double sigma) {
    return 1 / sigma * SQRT1_2PI * exp(-pow(value - mu, 2) / (2 * pow(sigma, 2)));
}

__host__ thrust::host_vector<int> bd_int(int m, int W, int T, double E_K) {
    thrust::host_vector<int> intervals(m * 2);
    double eps = 1e-10;
    double rho = W * log(2 / eps) / 2;
    for (int ell = 1; ell <= m; ell++) {
        int t = ceil(sqrt(ell * rho));
        int i_lb = floor((ell - 1) * E_K - t);
        int i_ub = ceil(ell * E_K + t);

        intervals[(ell - 1) * 2] = std::max(1, i_lb) - 1;
        intervals[(ell - 1) * 2 + 1] = std::min(T, i_ub) - 1;
    }
    return intervals;
}


__host__ __device__ double eln(double x) {
    if (x == 0) return -INFINITY;
    else return log(x);
}

__host__ __device__ double eexp(double x) {
    if (x == -INFINITY)  return 0;
    else return exp(x);
}

__host__ __device__ double elnproduct(double x, double y) {
    if (x == -INFINITY || y == -INFINITY) return -INFINITY;
    else return x + y;
}

std::vector<std::vector<double>> read_data(std::string fname, bool f = 0) {
    std::vector<std::vector<double>> data;
    std::ifstream inFile;
    // open the file stream
    inFile.open(fname);
    // check if opening a file failed
    if (inFile.fail()) {
        std::cerr << "Error opening file" << std::endl;
        inFile.close();
        exit(1);
    }
    std::string line;
    while (getline(inFile, line))
    {
        std::vector<double> vect;
        std::stringstream ss(line);
        for (double i; ss >> i;) {
            vect.push_back(i);
            if (ss.peek() == ',')
                ss.ignore();
        }
        data.push_back(vect);

    }
    // close the file stream
    inFile.close();

    // print if flag is set
    if (f) {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                std::cout << "(" << i << ", " << j << "): " << data[i][j] << std::endl;
            }
        }
    }

    return data;
}