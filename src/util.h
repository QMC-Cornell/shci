#pragma once

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <limits>
#include <string>

#define ENERGY_FORMAT "%.10f Ha"

class Util {
 public:
  template <class... Args>
  static std::string str_printf(const std::string& format, Args... args);

  static bool str_iequals(const std::string& a, const std::string& b);

  constexpr static double EPS = 1.0e-15;

  constexpr static double INF = std::numeric_limits<double>::max();

  constexpr static double SQRT2 = sqrt(2);

  constexpr static double SQRT2_INV = 1.0 / sqrt(2);
};

template <typename... Args>
std::string Util::str_printf(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...);
  std::unique_ptr<char[]> buf(new char[size + 1]);  // Extra space for '\0'
  snprintf(buf.get(), size + 1, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size);
}
