#include "util.h"

#include <execinfo.h>
#include <unistd.h>
#include <cctype>
#include <cstdio>

constexpr double Util::EPS;

constexpr double Util::INF;

constexpr double Util::SQRT2;

constexpr double Util::SQRT2_INV;

bool Util::str_equals_ci(const std::string& a, const std::string& b) {
  size_t size = a.size();
  if (b.size() != size) return false;
  for (size_t i = 0; i < size; i++) {
    if (tolower(a[i]) != tolower(b[i])) return false;
  }
  return true;
}

void Util::error_handler(const int sig) {
  void* array[128];
  size_t size = backtrace(array, 128);
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

double Util::avg(const std::vector<double>& vec) {
  double sum = 0.0;
  for (const double num : vec) {
    sum += num;
  }
  return sum / vec.size();
}

double Util::stdev(const std::vector<double>& vec) {
  double sum = 0.0;
  double sq_sum = 0.0;
  for (const double num : vec) {
    sum += num;
    sq_sum += num * num;
  }
  const double n = vec.size();
  return sqrt((sq_sum - sum * sum / n) / (n - 1));
}

size_t Util::combine_hash(const size_t a, const size_t b) {
  size_t hash = a;
  hash += (hash << 10);
  hash ^= (hash >> 6);
  hash += b;
  hash += (hash << 10);
  hash ^= (hash >> 6);
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

int Util::ctz(unsigned long long x) { return __builtin_ctzll(x); }

int Util::popcnt(unsigned long long x) { return __builtin_popcountll(x); }
