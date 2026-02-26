// lambda_standalone.cpp
// Standalone SSA + parser for lambda.r and lambda.in
//
// Build:
//   g++ -O2 -std=c++17 lambda_standalone.cpp -o lambda_standalone
//
// Run:
//   ./lambda_standalone lambda.r lambda.in 20000 1 2000000
//
// Args:
//   argv[1] reaction file (lambda.r)
//   argv[2] init/threshold file (lambda.in)
//   argv[3] trials per MOI (default 20000)
//   argv[4] seed (default 1)
//   argv[5] max steps per trajectory (default 2,000,000)
//
// Outcome:
//   stealth if cI2 > 145 (or comparator in lambda.in for cI2 line)
//   hijack  if Cro2 > 55 (or comparator in lambda.in for Cro2 line)

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using std::cerr;
using std::cout;
using std::ifstream;
using std::string;
using std::unordered_map;
using std::vector;

static inline string ltrim(string s) {
    size_t i = 0;
    while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
    return s.substr(i);
}
static inline string rtrim(string s) {
    size_t i = s.size();
    while (i > 0 && std::isspace((unsigned char)s[i - 1])) i--;
    return s.substr(0, i);
}
static inline string trim(string s) { return rtrim(ltrim(std::move(s))); }

static inline vector<string> split_ws(const string& s) {
    std::istringstream iss(s);
    vector<string> v;
    string t;
    while (iss >> t) v.push_back(t);
    return v;
}

// ---------- model structs (Aleae-ish minimal style) ----------
struct Term {
    int i;      // species index
    int n;      // stoichiometry
};

struct Reaction {
    vector<Term> L;   // reactants
    vector<Term> R;   // products
    double k;         // rate constant
};

enum class Cmp { LT, LE, GE, GT, EQ };

struct Thresh {
    int i;           // species index
    Cmp c;           // comparator
    long long t;     // threshold value
};

struct BioCR {
    vector<string> N;                 // species names
    unordered_map<string,int> id;     // name->index
    vector<Reaction> R;               // reactions
    vector<Thresh> T;                 // thresholds
};

// ---------- helpers ----------
static inline int get_id(BioCR& b, const string& name) {
    auto it = b.id.find(name);
    if (it != b.id.end()) return it->second;
    int idx = (int)b.N.size();
    b.N.push_back(name);
    b.id[name] = idx;
    return idx;
}

// Parse one side of reaction: "A 1 B 2" -> [(A,1),(B,2)]
static inline vector<std::pair<string,int>> parse_side(const string& side) {
    string s = trim(side);
    vector<std::pair<string,int>> out;
    if (s.empty()) return out;

    auto v = split_ws(s);
    if (v.size() % 2 != 0) {
        throw std::runtime_error("Bad stoichiometry side: '" + side + "'");
    }
    for (size_t i = 0; i < v.size(); i += 2) {
        int n = std::stoi(v[i + 1]);
        if (n < 0) throw std::runtime_error("Negative stoichiometry for " + v[i]);
        if (n == 0) continue;
        out.push_back({v[i], n});
    }
    return out;
}

// Split "lhs : rhs : rate"
static inline bool split_reaction_line(const string& line, string& lhs, string& rhs, string& rate) {
    size_t p1 = line.find(':');
    if (p1 == string::npos) return false;
    size_t p2 = line.find(':', p1 + 1);
    if (p2 == string::npos) return false;
    lhs  = trim(line.substr(0, p1));
    rhs  = trim(line.substr(p1 + 1, p2 - (p1 + 1)));
    rate = trim(line.substr(p2 + 1));
    return true;
}

// Combination factor for mass action with stoichiometry n: C(x,n)
static inline long double choose_factor(long long x, int n) {
    if (n == 0) return 1.0L;
    if (x < n) return 0.0L;
    long double num = 1.0L;
    for (int i = 0; i < n; i++) num *= (long double)(x - i);
    long double den = 1.0L;
    for (int i = 2; i <= n; i++) den *= (long double)i;
    return num / den;
}

static inline bool cmp_ok(long long x, Cmp c, long long t) {
    switch (c) {
        case Cmp::LT: return x <  t;
        case Cmp::LE: return x <= t;
        case Cmp::GE: return x >= t;
        case Cmp::GT: return x >  t;
        case Cmp::EQ: return x == t;
    }
    return false;
}

static inline int first_threshold_hit(const BioCR& b, const vector<long long>& S) {
    for (int j = 0; j < (int)b.T.size(); j++) {
        const Thresh& th = b.T[j];
        if (cmp_ok(S[th.i], th.c, th.t)) return j;
    }
    return -1;
}

// ---------- parsing ----------
static void load_reactions(const string& r_path, BioCR& b) {
    ifstream fin(r_path);
    if (!fin) throw std::runtime_error("Could not open " + r_path);

    string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        string lhs, rhs, rate;
        if (!split_reaction_line(line, lhs, rhs, rate))
            throw std::runtime_error("Bad reaction line: '" + line + "'");

        auto L = parse_side(lhs);
        auto R = parse_side(rhs);

        Reaction rx;
        rx.k = std::stod(rate);

        for (auto& p : L) {
            int idx = get_id(b, p.first);
            rx.L.push_back({idx, p.second});
        }
        for (auto& p : R) {
            int idx = get_id(b, p.first);
            rx.R.push_back({idx, p.second});
        }
        b.R.push_back(std::move(rx));
    }
}

static void load_init_and_thresholds(const string& in_path, BioCR& b, vector<long long>& S0) {
    ifstream fin(in_path);
    if (!fin) throw std::runtime_error("Could not open " + in_path);

    // We'll parse into species as we see them; then resize S0 as needed.
    string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        auto v = split_ws(line);

        // Threshold format commonly: "<species> 0 GE 145"
        // Init format commonly:      "<species> 6 N"
        if (v.size() >= 4 && (v[2] == "GE" || v[2] == "GT" || v[2] == "LE" || v[2] == "LT" || v[2] == "EQ")) {
            int idx = get_id(b, v[0]);
            if ((int)S0.size() < (int)b.N.size()) S0.resize(b.N.size(), 0);

            Cmp c = Cmp::GE;
            if (v[2] == "LT") c = Cmp::LT;
            else if (v[2] == "LE") c = Cmp::LE;
            else if (v[2] == "GE") c = Cmp::GE;
            else if (v[2] == "GT") c = Cmp::GT;
            else c = Cmp::EQ;

            long long t = std::stoll(v[3]);
            b.T.push_back({idx, c, t});
        } else if (v.size() >= 2) {
            int idx = get_id(b, v[0]);
            if ((int)S0.size() < (int)b.N.size()) S0.resize(b.N.size(), 0);
            S0[idx] = std::stoll(v[1]);
        } else {
            throw std::runtime_error("Bad .in line: '" + line + "'");
        }
    }
}

// ---------- SSA ----------
enum class Outcome { Stealth, Hijack, Undecided };

static Outcome run_ssa(const BioCR& b,
                       vector<long long> S,
                       int idx_cI2,
                       int idx_Cro2,
                       std::mt19937_64& rng,
                       long long max_steps) {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // check t=0
    int hit0 = first_threshold_hit(b, S);
    if (hit0 >= 0) {
        int which = b.T[hit0].i;
        if (which == idx_cI2) return Outcome::Stealth;
        if (which == idx_Cro2) return Outcome::Hijack;
        return Outcome::Undecided;
    }

    vector<long double> a(b.R.size());

    for (long long step = 0; step < max_steps; step++) {
        long double a0 = 0.0L;

        for (size_t j = 0; j < b.R.size(); j++) {
            const Reaction& rx = b.R[j];
            long double aj = (long double)rx.k;
            for (const auto& t : rx.L) {
                aj *= choose_factor(S[t.i], t.n);
                if (aj == 0.0L) break;
            }
            a[j] = aj;
            a0 += aj;
        }

        if (a0 <= 0.0L) return Outcome::Undecided;

        long double r1 = (long double)U(rng);
        long double pick = r1 * a0;

        long double sum = 0.0L;
        size_t mu = b.R.size() - 1;
        for (size_t j = 0; j < b.R.size(); j++) {
            sum += a[j];
            if (sum >= pick) { mu = j; break; }
        }

        // fire mu
        const Reaction& rx = b.R[mu];
        for (const auto& t : rx.L) S[t.i] -= t.n;
        for (const auto& t : rx.R) S[t.i] += t.n;

        int hit = first_threshold_hit(b, S);
        if (hit >= 0) {
            int which = b.T[hit].i;
            if (which == idx_cI2) return Outcome::Stealth;
            if (which == idx_Cro2) return Outcome::Hijack;
            return Outcome::Undecided;
        }
    }

    return Outcome::Undecided;
}

int main(int argc, char** argv) {
    try {
        string r_path  = (argc >= 2) ? argv[1] : "lambda.r";
        string in_path = (argc >= 3) ? argv[2] : "lambda.in";
        long long trials = (argc >= 4) ? std::stoll(argv[3]) : 20000;
        uint64_t seed    = (argc >= 5) ? (uint64_t)std::stoull(argv[4]) : 1ULL;
        long long max_steps = (argc >= 6) ? std::stoll(argv[5]) : 2000000;

        BioCR b;
        vector<long long> S0;

        // Parse reactions first (defines most species)
        load_reactions(r_path, b);
        // Then parse init/thresholds (may add species too)
        if ((int)S0.size() < (int)b.N.size()) S0.resize(b.N.size(), 0);
        load_init_and_thresholds(in_path, b, S0);

        int idx_MOI  = (b.id.count("MOI")  ? b.id["MOI"]  : -1);
        int idx_cI2  = (b.id.count("cI2")  ? b.id["cI2"]  : -1);
        int idx_Cro2 = (b.id.count("Cro2") ? b.id["Cro2"] : -1);

        cout << "Loaded " << b.N.size() << " species, " << b.R.size() << " reactions.\n";
        cout << "Trials/MOI=" << trials << " seed=" << seed << " max_steps=" << max_steps << "\n\n";

        std::mt19937_64 rng(seed);

        for (int moi = 1; moi <= 10; moi++) {
            long long stealth = 0, hijack = 0, undec = 0;

            for (long long t = 0; t < trials; t++) {
                vector<long long> S = S0;
                if (idx_MOI >= 0) S[idx_MOI] = moi;

                Outcome out = run_ssa(b, std::move(S), idx_cI2, idx_Cro2, rng, max_steps);
                if (out == Outcome::Stealth) stealth++;
                else if (out == Outcome::Hijack) hijack++;
                else undec++;
            }

            double pS = (double)stealth / (double)trials;
            double pH = (double)hijack  / (double)trials;
            double pU = (double)undec   / (double)trials;

            cout << "MOI=" << moi
                 << "  P(stealth)="  << std::fixed << std::setprecision(4) << pS
                 << "  P(hijack)="   << std::fixed << std::setprecision(4) << pH
                 << "  P(undecided)="<< std::fixed << std::setprecision(4) << pU
                 << "  counts=[" << stealth << "," << hijack << "," << undec << "]\n";
        }

        return 0;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}