#include <vector>
#include <cmath>

using namespace std;

vector<vector<double>> createMatrix(int N) {
    vector<vector<double>> A(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A[i][j] = 100.0;
            }
            else {
                A[i][j] = 1.0 + 0.1 * (i + 1) + 0.2 * (j + 1);
            }
        }
    }
    return A;
}

vector<double> createRightHandSide(const vector<vector<double>>& A) {
    int N = A.size();
    vector<double> f(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f[i] += A[i][j];
        }
    }
    return f;
}

double vectorNorm(const vector<double>& v) {
    double norm = 0.0;
    for (double val : v) {
        norm += val * val;
    }
    return sqrt(norm);
}

double computeError(const vector<double>& x, const vector<double>& x_exact) {
    int N = x.size();
    vector<double> diff(N);
    for (int i = 0; i < N; ++i) {
        diff[i] = x[i] - x_exact[i];
    }
    return vectorNorm(diff) / vectorNorm(x_exact);
}