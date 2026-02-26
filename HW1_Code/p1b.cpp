#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>

struct State {
    long long x1, x2, x3;
};

static inline void step_once(State &s, std::mt19937_64 &rng) {
    // weights from handout
    double w1 = 0.5 * (double)s.x1 * (double)(s.x1 - 1) * (double)s.x2;
    double w2 = (double)s.x1 * (double)s.x3 * (double)(s.x3 - 1);
    double w3 = 3.0 * (double)s.x2 * (double)s.x3;

    double W = w1 + w2 + w3;
    if (!(W > 0.0)) {
        // no reaction can fire
        return;
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng) * W;

    if (r < w1) {
        // R1: (-2, -1, +4)
        s.x1 -= 2; s.x2 -= 1; s.x3 += 4;
    } else if (r < w1 + w2) {
        // R2: (-1, +3, -2)
        s.x1 -= 1; s.x2 += 3; s.x3 -= 2;
    } else {
        // R3: (+2, -1, -1)
        s.x1 += 2; s.x2 -= 1; s.x3 -= 1;
    }

    // Defensive clamp: should not go negative if weights were correct,
    // but floating errors / edge cases can happen.
    if (s.x1 < 0) s.x1 = 0;
    if (s.x2 < 0) s.x2 = 0;
    if (s.x3 < 0) s.x3 = 0;
}

struct RunningStats {
    // Welford's online mean/variance
    long long n = 0;
    double mean = 0.0;
    double M2 = 0.0;

    void add(double x) {
        n += 1;
        double delta = x - mean;
        mean += delta / (double)n;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    double variance_unbiased() const {
        return (n > 1) ? (M2 / (double)(n - 1)) : 0.0;
    }
};

int main(int argc, char** argv) {
    // Defaults
    long long N = 200000;     
    int steps = 7;
    uint64_t seed = 1;

    // CLI: ./prog [N] [seed]
    if (argc >= 2) N = std::stoll(argv[1]);
    if (argc >= 3) seed = (uint64_t)std::stoull(argv[2]);

    std::mt19937_64 rng(seed);

    RunningStats st1, st2, st3;

    for (long long i = 0; i < N; ++i) {
        State s{9, 8, 7};

        for (int t = 0; t < steps; ++t) {
            step_once(s, rng);
        }

        st1.add((double)s.x1);
        st2.add((double)s.x2);
        st3.add((double)s.x3);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Monte Carlo mean/variance after " << steps
              << " steps, start S0=[9,8,7]\n";
    std::cout << "N=" << N << ", seed=" << seed << "\n\n";

    std::cout << "X1: mean = " << st1.mean
              << ", var = " << st1.variance_unbiased() << "\n";
    std::cout << "X2: mean = " << st2.mean
              << ", var = " << st2.variance_unbiased() << "\n";
    std::cout << "X3: mean = " << st3.mean
              << ", var = " << st3.variance_unbiased() << "\n";

    return 0;
}
