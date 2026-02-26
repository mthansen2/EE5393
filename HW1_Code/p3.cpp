// crn_log_then_exp_fixed.cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

static inline long long comb2(long long n) {
    return (n >= 2) ? (n * (n - 1)) / 2 : 0;
}

struct RNG {
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> uni;
    explicit RNG(std::uint64_t seed) : gen(seed), uni(0.0, 1.0) {}
    double u() { return uni(gen); }
};

struct LogRates {
    double k_b_to_ab;     // b -> a + b
    double k_a_2x;        // a + 2x -> c + x' + a
    double k_2c_to_c;     // 2c -> c
    double k_a_decay;     // a -> ∅
    double k_xp_to_x;     // x' -> x
    double k_c_to_y;      // c -> y
};

struct ExpRates {
    double k_x_to_a;      // x -> a
    double k_a_y;         // a + y -> a + 2y'
    double k_a_decay;     // a -> ∅
    double k_yp_to_y;     // y' -> y
};

struct LogState { long long x=0,b=0,a=0,c=0,xp=0,y=0; };
struct ExpState { long long x=0,a=0,y=0,yp=0; };

LogState simulate_log(long long X0, long long B0, const LogRates& R, std::uint64_t seed,
                      long long max_events = 10'000'000)
{
    RNG rng(seed);
    LogState s;
    s.x = X0;
    s.b = B0;

    for (long long ev=0; ev<max_events; ++ev) {
        // Propensities (discrete mass action)
        double r1 = R.k_b_to_ab  * (double)s.b;
        double r2 = R.k_a_2x     * (double)s.a * (double)comb2(s.x);
        double r3 = R.k_2c_to_c  * (double)comb2(s.c);
        double r4 = R.k_a_decay  * (double)s.a;
        double r5 = R.k_xp_to_x  * (double)s.xp;
        double r6 = R.k_c_to_y   * (double)s.c;

        double r0 = r1+r2+r3+r4+r5+r6;
        if (r0 <= 0.0) break;

        double pick = rng.u() * r0;

        if ((pick -= r1) < 0.0) {                 // b -> a + b
            s.a += 1;
        } else if ((pick -= r2) < 0.0) {          // a + 2x -> c + xp + a
            s.x  -= 2;
            s.xp += 1;
            s.c  += 1;
        } else if ((pick -= r3) < 0.0) {          // 2c -> c  (net -1 c)
            s.c -= 1;
        } else if ((pick -= r4) < 0.0) {          // a -> ∅
            s.a -= 1;
        } else if ((pick -= r5) < 0.0) {          // xp -> x
            s.xp -= 1;
            s.x  += 1;
        } else {                                   // c -> y
            s.c -= 1;
            s.y += 1;
        }

        // clamp
        s.x  = std::max(0LL, s.x);
        s.a  = std::max(0LL, s.a);
        s.c  = std::max(0LL, s.c);
        s.xp = std::max(0LL, s.xp);

        // Termination: X stabilized at 1 and intermediates cleared
        if (s.x == 1 && s.xp == 0 && s.c == 0 && s.a == 0) break;
    }
    return s;
}

ExpState simulate_exp(long long L, const ExpRates& R, std::uint64_t seed,
                      long long max_events = 50'000'000)
{
    RNG rng(seed);
    ExpState s;
    s.x = L;
    s.y = 1; // initial output Y=1

    for (long long ev=0; ev<max_events; ++ev) {
        double r1 = R.k_x_to_a   * (double)s.x;            // x -> a
        double r2 = R.k_a_y      * (double)s.a * (double)s.y; // a + y -> a + 2yp
        double r3 = R.k_a_decay  * (double)s.a;            // a -> ∅
        double r4 = R.k_yp_to_y  * (double)s.yp;           // yp -> y

        double r0 = r1+r2+r3+r4;
        if (r0 <= 0.0) break;

        double pick = rng.u() * r0;

        if ((pick -= r1) < 0.0) {           // x -> a
            s.x -= 1;
            s.a += 1;
        } else if ((pick -= r2) < 0.0) {    // a + y -> a + 2yp
            s.y  -= 1;
            s.yp += 2;
        } else if ((pick -= r3) < 0.0) {    // a -> ∅
            s.a -= 1;
        } else {                             // yp -> y
            s.yp -= 1;
            s.y  += 1;
        }

        s.x  = std::max(0LL, s.x);
        s.a  = std::max(0LL, s.a);
        s.y  = std::max(0LL, s.y);
        s.yp = std::max(0LL, s.yp);

        // done when no input and no intermediates
        if (s.x == 0 && s.a == 0 && s.yp == 0) break;
    }
    return s;
}

int main(int argc, char** argv) {
    long long X0 = 16;
    long long B0 = 1;
    int trials = 50;
    std::uint64_t seed = 1;

    if (argc >= 2) X0 = std::stoll(argv[1]);
    if (argc >= 3) B0 = std::stoll(argv[2]);
    if (argc >= 4) trials = std::stoi(argv[3]);
    if (argc >= 5) seed = (std::uint64_t)std::stoull(argv[4]);

    // Strong time-scale separation (key fix)
    LogRates LR{
    /*b->a+b*/   1e-6,
    /*a+2x*/     1e6,
    /*2c->c*/    1e6,
    /*a->∅*/     1e3,
    /*xp->x*/    1.0,
    /*c->y*/     1.0
};

ExpRates ER{
    /*x->a*/     1e-6,
    /*a+y*/      1e6,
    /*a->∅*/     1e3,
    /*yp->y*/    1.0
};


    std::map<long long, int> histL, histY;

    for (int t=0; t<trials; ++t) {
        auto logS = simulate_log(X0, B0, LR, seed + (std::uint64_t)t);
        long long L = logS.y;

        auto expS = simulate_exp(L, ER, seed + 9999 + (std::uint64_t)t);
        long long Y = expS.y;

        histL[L] += 1;
        histY[Y] += 1;
    }

    std::cout << "X0=" << X0 << ", trials=" << trials << "\n";
    std::cout << "Histogram of L = log2(X0) output:\n";
    for (auto &kv : histL) std::cout << "  L=" << kv.first << " : " << kv.second << "\n";

    std::cout << "Histogram of Y = 2^L output:\n";
    for (auto &kv : histY) std::cout << "  Y=" << kv.first << " : " << kv.second << "\n";

    return 0;
}
