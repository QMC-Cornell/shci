#ifndef JL2922_MATH_VECTOR_H
#define JL2922_MATH_VECTOR_H

#include <array>

template <class T, size_t N>
class MathVector {
 public:
  MathVector(const T& default_value = 0.0) {
    for (size_t i = 0; i < N; i++) data[i] = default_value;
  }

  MathVector<T, N>& operator=(const MathVector<T, N>& rhs) {
    for (size_t i = 0; i < N; i++) data[i] = rhs[i];
    return *this;
  }

  MathVector<T, N>& operator+=(const MathVector<T, N>& rhs) {
    for (size_t i = 0; i < N; i++) data[i] += rhs[i];
    return *this;
  }

  T& operator[](const size_t i) { return data[i]; }

  const T& operator[](const size_t i) const { return data[i]; }

  template <class B>
  void serialize(B& buf) const {
    buf << data;
  }

  template <class B>
  void parse(B& buf) {
    buf >> data;
  }

 private:
  std::array<T, N> data;
};

#endif
