#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>

using namespace std;

void luDecomposition(vector<vector<double>>& A, vector<int>& pivot) {
    int N = A.size();
    pivot.resize(N);
    for (int i = 0; i < N; ++i) pivot[i] = i;

    for (int k = 0; k < N; ++k) {
        int max_row = k;
        double max_val = abs(A[k][k]);
        for (int i = k + 1; i < N; ++i) {
            if (abs(A[i][k]) > max_val) {
                max_val = abs(A[i][k]);
                max_row = i;
            }
        }

        if (max_row != k) {
            swap(A[k], A[max_row]);
            swap(pivot[k], pivot[max_row]);
        }

        if (abs(A[k][k]) < 1e-12) {
            throw runtime_error("Matrix is singular or nearly singular");
        }

        for (int i = k + 1; i < N; ++i) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

vector<double> solveLU(const vector<vector<double>>& A,
    const vector<double>& f,
    const vector<int>& pivot) {
    int N = A.size();
    vector<double> x(N), b(N);

    for (int i = 0; i < N; ++i) {
        b[i] = f[pivot[i]];
    }

    vector<double> y(N);
    for (int i = 0; i < N; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= A[i][j] * y[j];
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}