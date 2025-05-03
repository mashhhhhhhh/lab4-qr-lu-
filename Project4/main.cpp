#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Îáúÿâëåíèÿ ôóíêöèé èç äðóãèõ ôàéëîâ
vector<vector<double>> createMatrix(int N);
vector<double> createRightHandSide(const vector<vector<double>>& A);
double vectorNorm(const vector<double>& v);
double computeError(const vector<double>& x, const vector<double>& x_exact);
void luDecomposition(vector<vector<double>>& A, vector<int>& pivot);
vector<double> solveLU(const vector<vector<double>>& A, 
                     const vector<double>& f, 
                     const vector<int>& pivot);
void gramSchmidtQR(vector<vector<double>>& A, 
                 vector<vector<double>>& Q, 
                 vector<vector<double>>& R);
vector<double> solveQR(const vector<vector<double>>& Q, 
                     const vector<vector<double>>& R, 
                     const vector<double>& f);

int main() {
    vector<int> sizes = {250, 500, 1000};
    
    //cout << fixed << setprecision(6);
    cout << "N\tLU Time (ms)\tLU Error\tQR Time (ms)\tQR Error\n";
    
    for (int N : sizes) {
        // Ñîçäàåì ìàòðèöó è âåêòîðû
        auto A = createMatrix(N);
        auto x_exact = vector<double>(N, 1.0);
        auto f = createRightHandSide(A);
        
        // LU ðåøåíèå
        auto start_lu = high_resolution_clock::now();
        auto A_lu = A;
        vector<int> pivot;
        luDecomposition(A_lu, pivot);
        auto x_lu = solveLU(A_lu, f, pivot);
        auto end_lu = high_resolution_clock::now();
        auto lu_time = duration<double>(end_lu - start_lu).count();
        long double lu_error = computeError(x_lu, x_exact);
        
        // QR ðåøåíèå
        auto start_qr = high_resolution_clock::now();
        auto A_qr = A;
        vector<vector<double>> Q, R;
        gramSchmidtQR(A_qr, Q, R);
        auto x_qr = solveQR(Q, R, f);
        auto end_qr = high_resolution_clock::now();
        auto qr_time = duration<double>(end_qr - start_qr).count();
        double qr_error = computeError(x_qr, x_exact);
        
        // Âûâîä ðåçóëüòàòîâ
        cout << N << "\t" << lu_time << "\t" << lu_error << "\t" 
             << qr_time << "\t" << qr_error << "\n";
    }
    
    return 0;
}
