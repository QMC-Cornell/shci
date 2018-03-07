#pragma once

#include <cctype>
#include <string>

namespace util {
bool str_iequals(const std::string& a, const std::string& b) {
  size_t size = a.size();
  if (b.size() != size) return false;
  for (size_t i = 0; i < size; i++) {
    if (tolower(a[i]) != tolower(b[i])) return false;
  }
  return true;
}
}  // namespace util