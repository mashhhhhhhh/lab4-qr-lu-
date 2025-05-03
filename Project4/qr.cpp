#include <vector>
#include <cmath>
#include <stdexcept>

using namespace std;

void gramSchmidtQR(vector<vector<double>>& A,
    vector<vector<double>>& Q,
    vector<vector<double>>& R) {
    int N = A.size();
    Q = vector<vector<double>>(N, vector<double>(N, 0.0));
    R = vector<vector<double>>(N, vector<double>(N, 0.0));

    for (int j = 0; j < N; ++j) {
        vector<double> v(N);
        for (int k = 0; k < N; ++k) {
            v[k] = A[k][j];
        }

        for (int i = 0; i < j; ++i) {
            R[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                R[i][j] += Q[k][i] * A[k][j];
            }
            for (int k = 0; k < N; ++k) {
                v[k] -= R[i][j] * Q[k][i];
            }
        }

        R[j][j] = 0.0;
        for (int k = 0; k < N; ++k) {
            R[j][j] += v[k] * v[k];
        }
        R[j][j] = sqrt(R[j][j]);

        if (R[j][j] < 1e-12) {
            throw runtime_error("Matrix is singular or nearly singular");
        }

        for (int k = 0; k < N; ++k) {
            Q[k][j] = v[k] / R[j][j];
        }
    }
}

vector<double> solveQR(const vector<vector<double>>& Q,
    const vector<vector<double>>& R,
    const vector<double>& f) {
    int N = Q.size();
    vector<double> x(N), y(N);

    for (int i = 0; i < N; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            y[i] += Q[j][i] * f[j];
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}