#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define ENERGY_FORMAT "%.10f Ha"

class Util {
 public:
  template <class... Args>
  static std::string str_printf(const std::string& format, Args... args);

  static bool str_equals_ci(const std::string& a, const std::string& b);

  static double avg(const std::vector<double>& vec);

  static double stdev(const std::vector<double>& vec);

  static double dot_omp(const std::vector<double>& a, const std::vector<double>& b);
 
  static std::complex<double> dot_omp(
      const std::vector<double>& a, const std::vector<std::complex<double>>& b);

  static std::complex<double> dot_omp(
      const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b);

  static size_t rehash(const size_t a);

  static int ctz(unsigned long long x);

  static int popcnt(unsigned long long x);

  static void setup_alias_arrays(const std::vector<double>& old_probs, std::vector<double>& new_probs, std::vector<size_t>& aliases);

  static size_t get_mem_total();

  static size_t get_mem_avail();

  template <class T>
  static void free(T& t);

  template <class T1, class T2>
  static void sort_by_first(std::vector<T1>& v1, std::vector<T2>& v2);

  constexpr static double EPS = 1.0e-12;

  constexpr static double INF = 1.0e100;

  constexpr static double PI = 3.14159265358979323846;

  constexpr static std::complex<double> I = std::complex<double>(0, 1);

  constexpr static double SQRT2 = 1.4142135623730951;

  constexpr static double SQRT2_INV = 0.7071067811865475;

 private:
  static size_t get_mem_info(const std::string& key);
};

template <typename... Args>
std::string Util::str_printf(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...);
  std::unique_ptr<char[]> buf(new char[size + 1]);  // Extra space for '\0'
  snprintf(buf.get(), size + 1, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size);
}

template <class T>
void Util::free(T& t) {
  T dummy;
  std::swap(t, dummy);
}

template <class T1, class T2>
void Util::sort_by_first(std::vector<T1>& vec1, std::vector<T2>& vec2) {
  std::vector<std::pair<T1, T2>> vec;
  const size_t n_vec = vec1.size();
  for (size_t i = 0; i < n_vec; i++) {
    vec.push_back(std::make_pair(vec1[i], vec2[i]));
  }
  std::sort(vec.begin(), vec.end(), [&](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
    return a.first < b.first;
  });
  for (size_t i = 0; i < n_vec; i++) {
    vec1[i] = vec[i].first;
    vec2[i] = vec[i].second;
  }
}
