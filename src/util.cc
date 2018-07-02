#include "util.h"

#include <execinfo.h>
#include <omp.h>
#include <unistd.h>
#include <cctype>
#include <cstdio>
#include <fstream>

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

double Util::dot_omp(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  const size_t n = a.size();
  const int n_threads = omp_get_max_threads();
#ifdef __INTEL_COMPILER
  const size_t n_per_thread = n / n_threads + 1;
#pragma omp parallel
{
  const int thread_id = omp_get_thread_num();
  double sum_thread = 0.0;
  for (size_t i = thread_id * n_per_thread; i < n && i < (thread_id + 1) * n_per_thread; i++) {
    sum_thread += a[i] * b[i]; 
  } 
#pragma omp atomic
  sum += sum_thread;
}
#else
  std::vector<double> sum_thread(n_threads, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    const int thread_id = omp_get_thread_num();
    sum_thread[thread_id] += a[i] * b[i];
  }
  for (int i = 0; i < n_threads; i++) sum += sum_thread[i];
#endif
  return sum;
}

size_t Util::rehash(const size_t a) {
  size_t hash = a;
  hash += (hash << 10);
  hash ^= (hash >> 6);
  hash += 982451653ull;  // 50M-th prime.
  hash += (hash << 10);
  hash ^= (hash >> 6);
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

int Util::ctz(unsigned long long x) { return __builtin_ctzll(x); }

int Util::popcnt(unsigned long long x) { return __builtin_popcountll(x); }

size_t Util::get_mem_info(const std::string& key) {
  std::ifstream meminfo("/proc/meminfo");
  std::string token;
  size_t value;
  while (meminfo >> token) {
    if (token == key + ":") {
      if (meminfo >> value) {
        return value;
      } else {
        return 0;
      }
    }
  }
  return 0;
}

size_t Util::get_mem_total() { return get_mem_info("MemTotal"); }

size_t Util::get_mem_avail() { return get_mem_info("MemAvailable"); }
