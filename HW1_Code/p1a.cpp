// gillespie_discrete.cpp
// C++17 Monte Carlo simulation using handout's discrete next-reaction probabilities.

#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>

struct Hits {
    bool C1 = false;
    bool C2 = false;
    bool C3 = false;
};

static inline bool inC1(long long x1, long long, long long) { return x1 >= 150; }
static inline bool inC2(long long, long long x2, long long) { return x2 < 10; }
static inline bool inC3(long long, long long, long long x3) { return x3 > 100; }

Hits run_trajectory_discrete(
    long long x1 = 110, long long x2 = 26, long long x3 = 55,
    long long max_steps = 500000,
    bool stop_when_all_hit = false,
    std::mt19937_64* rng_ptr = nullptr
) {
    std::mt19937_64 local_rng;
    if (!rng_ptr) {
        std::random_device rd;
        local_rng.seed(((uint64_t)rd() << 32) ^ (uint64_t)rd());
        rng_ptr = &local_rng;
    }
    std::mt19937_64& rng = *rng_ptr;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    Hits hits;
    hits.C1 = inC1(x1, x2, x3);
    hits.C2 = inC2(x1, x2, x3);
    hits.C3 = inC3(x1, x2, x3);

    for (long long step = 0; step < max_steps; ++step) {
        // update hits at current state
        if (!hits.C1 && inC1(x1, x2, x3)) hits.C1 = true;
        if (!hits.C2 && inC2(x1, x2, x3)) hits.C2 = true;
        if (!hits.C3 && inC3(x1, x2, x3)) hits.C3 = true;

        if (stop_when_all_hit && hits.C1 && hits.C2 && hits.C3) break;

        // weights from the handout:
        // w1 = 1/2 x1(x1-1)x2
        // w2 = x1 x3 (x3-1)
        // w3 = 3 x2 x3
        double w1 = 0.5 * (double)x1 * (double)(x1 - 1) * (double)x2;
        double w2 = (double)x1 * (double)x3 * (double)(x3 - 1);
        double w3 = 3.0 * (double)x2 * (double)x3;

        double W = w1 + w2 + w3;
        if (!(W > 0.0)) { // no reaction can fire
            break;
        }

        double r = uni(rng) * W;

        // choose reaction and update state (stoichiometry)
        if (r < w1) {
            // R1: (-2, -1, +4)
            x1 -= 2; x2 -= 1; x3 += 4;
        } else if (r < w1 + w2) {
            // R2: (-1, +3, -2)
            x1 -= 1; x2 += 3; x3 -= 2;
        } else {
            // R3: (+2, -1, -1)
            x1 += 2; x2 -= 1; x3 -= 1;
        }

        // defensive: should not go negative if weights were correct
        if (x1 < 0 || x2 < 0 || x3 < 0) break;
    }

    // final state check
    hits.C1 = hits.C1 || inC1(x1, x2, x3);
    hits.C2 = hits.C2 || inC2(x1, x2, x3);
    hits.C3 = hits.C3 || inC3(x1, x2, x3);

    return hits;
}

struct Estimate {
    double p = 0.0;
    double lo = 0.0;
    double hi = 0.0;
};

Estimate proportion_ci_95(long long hits, long long N) {
    // normal approx CI: p ± 1.96 sqrt(p(1-p)/N)
    double p = (N > 0) ? (double)hits / (double)N : 0.0;
    double var = std::max(p * (1.0 - p), 1e-16);
    double se = std::sqrt(var / (double)N);
    double lo = std::max(0.0, p - 1.96 * se);
    double hi = std::min(1.0, p + 1.96 * se);
    return {p, lo, hi};
}

int main(int argc, char** argv) {
    // defaults
    long long N = 5000;
    long long max_steps = 500000;
    bool stop_when_all_hit = false;
    uint64_t seed = 1;

    // simple CLI parsing:
    // ./a.out [N] [max_steps] [stop_when_all_hit(0/1)] [seed]
    if (argc >= 2) N = std::stoll(argv[1]);
    if (argc >= 3) max_steps = std::stoll(argv[2]);
    if (argc >= 4) stop_when_all_hit = (std::stoi(argv[3]) != 0);
    if (argc >= 5) seed = (uint64_t)std::stoull(argv[4]);

    std::mt19937_64 rng(seed);

    long long hitC1 = 0, hitC2 = 0, hitC3 = 0;

    for (long long i = 0; i < N; ++i) {
        Hits h = run_trajectory_discrete(110, 26, 55, max_steps, stop_when_all_hit, &rng);
        hitC1 += h.C1 ? 1 : 0;
        hitC2 += h.C2 ? 1 : 0;
        hitC3 += h.C3 ? 1 : 0;
    }

    Estimate e1 = proportion_ci_95(hitC1, N);
    Estimate e2 = proportion_ci_95(hitC2, N);
    Estimate e3 = proportion_ci_95(hitC3, N);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Monte Carlo estimates (ever-hit by step cutoff)\n";
    std::cout << "Start state S0 = [110, 26, 55]\n";
    std::cout << "N = " << N << ", max_steps = " << max_steps
              << ", stop_when_all_hit = " << (stop_when_all_hit ? "true" : "false")
              << ", seed = " << seed << "\n\n";

    std::cout << "P(C1: x1 >= 150) ≈ " << e1.p << "   95% CI [" << e1.lo << ", " << e1.hi << "]"
              << "   (hits " << hitC1 << "/" << N << ")\n";
    std::cout << "P(C2: x2 < 10)   ≈ " << e2.p << "   95% CI [" << e2.lo << ", " << e2.hi << "]"
              << "   (hits " << hitC2 << "/" << N << ")\n";
    std::cout << "P(C3: x3 > 100)  ≈ " << e3.p << "   95% CI [" << e3.lo << ", " << e3.hi << "]"
              << "   (hits " << hitC3 << "/" << N << ")\n";

    // If you get zero hits, a useful bound is the "rule of three": p < 3/N (approx 95%).
    if (hitC1 == 0) std::cout << "Rule-of-three bound for C1 (0 hits): p < " << (3.0 / (double)N) << " (approx 95%)\n";
    if (hitC2 == 0) std::cout << "Rule-of-three bound for C2 (0 hits): p < " << (3.0 / (double)N) << " (approx 95%)\n";
    if (hitC3 == 0) std::cout << "Rule-of-three bound for C3 (0 hits): p < " << (3.0 / (double)N) << " (approx 95%)\n";

    return 0;
}
