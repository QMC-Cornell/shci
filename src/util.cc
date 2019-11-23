#include "util.h"

#include <execinfo.h>
#include <omp.h>
#include <unistd.h>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <stack>
#include <iostream>

constexpr double Util::EPS;

constexpr double Util::INF;

constexpr double Util::SQRT2;

constexpr double Util::SQRT2_INV;

constexpr std::complex<double> Util::I;

// Case insensitive strings comparison.
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

// Dot parallelized on a node.
double Util::dot_omp(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  const size_t n = a.size();
  const int n_threads = omp_get_max_threads();
#ifdef CRAY_KNIGHTS_LANDING
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

std::complex<double> Util::dot_omp(
    const std::vector<double>& a, const std::vector<std::complex<double>>& b) {
  std::complex<double> sum = 0.0;
  const size_t n = a.size();
  const int n_threads = omp_get_max_threads();
  std::vector<std::complex<double>> sum_thread(n_threads, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    const int thread_id = omp_get_thread_num();
    sum_thread[thread_id] += a[i] * b[i];
  }
  for (int i = 0; i < n_threads; i++) sum += sum_thread[i];
  return sum;
}

std::complex<double> Util::dot_omp(
    const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b) {
  std::complex<double> sum = 0.0;
  const size_t n = a.size();
  const int n_threads = omp_get_max_threads();
  std::vector<std::complex<double>> sum_thread(n_threads, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    const int thread_id = omp_get_thread_num();
    sum_thread[thread_id] += a[i] * b[i];
  }
  for (int i = 0; i < n_threads; i++) sum += sum_thread[i];
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

void Util::setup_alias_arrays(
    const std::vector<double>& old_probs,
    std::vector<double>& new_probs,
    std::vector<size_t>& aliases) {
  //======================================================
  // Set up alias arrays (new_probs, aliases) for sampling
  // from a discrete distribution (old_probs).
  // Based on Vose's alias method (O(N) complexity):
  // http://www.keithschwarz.com/darts-dice-coins/
  //
  // Created: Y. Yao, Sept 2019
  //======================================================
  const size_t length = old_probs.size();
  new_probs.resize(length);
  aliases.resize(length);
  std::vector<double> tmp_probs(length);
  for (size_t i = 0; i < length; i++) tmp_probs[i] = old_probs[i] * length;
  std::stack<size_t> smaller, larger;
  for (size_t i = 0; i < length; i++) {
    tmp_probs[i] = old_probs[i] * length;
    if (tmp_probs[i] < 1.)
      smaller.push(i);
    else
      larger.push(i);
  }
  while (!smaller.empty() && !larger.empty()) {
    const size_t l = larger.top(), s = smaller.top();
    new_probs[s] = tmp_probs[s];
    aliases[s] = l;
    tmp_probs[l] = tmp_probs[l] + tmp_probs[s] - 1.;
    if (tmp_probs[l] < 1.) {
      larger.pop();
      smaller.top() = l;
    } else {
      smaller.pop();
    }
  }
  while (!larger.empty()) {
    const size_t l = larger.top();
    larger.pop();
    new_probs[l] = 1.;
  }
  while (!smaller.empty()) {
    const size_t s = smaller.top();
    smaller.pop();
    new_probs[s] = 1.;
  }
}

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

size_t Util::get_mem_total() { return get_mem_info("MemTotal") * 1000; }

size_t Util::get_mem_avail() { return get_mem_info("MemAvailable") * 1000; }
