#pragma once
#include <fgpl/src/hash_map.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "integrals_hasher.h"

class VectorStorage {
 public:
  double get(const size_t key, const double default_value) const {
    if (key < vectr.size()) return vectr[key];
    return default_value;
  }

  void set(
      const size_t key,
      const double integral,
      const std::function<void(double&, const double&)>& reducer) {
    if (key >= vectr.size()) vectr.resize(key + 1, 0.0);
    if (vectr[key] == 0.0) {
      vectr[key] = integral;
      num_vectr_elems++;
    } else
      reducer(vectr[key], integral);
  }

  void clear() {
    vectr.clear();
    num_vectr_elems = 0;
  }

  template <class B>
  void serialize(B& buf) const {
    buf << num_vectr_elems;
    for (size_t key = 0; key < vectr.size(); key++) {
      double value = vectr[key];
      if (value != 0.0) buf << key << value;
    }
  }

  template <class B>
  void parse(B& buf) {
    vectr.clear();
    auto keep = [](double&, const double&) {};
    size_t n_keys_buf;
    buf >> n_keys_buf;
    unsigned key;
    double value;
    for (size_t i = 0; i < n_keys_buf; i++) {
      buf >> key >> value;
      set(key, value, keep);
    }
  }

  size_t num_elements() const { return num_vectr_elems; }

 private:
  size_t num_vectr_elems = 0;
  std::vector<double> vectr;
};

class IntegralsContainer {
 public:
  double get(const size_t key, const double default_value) const {
    if (hash_integrals) return hash.get(key, default_value);
    return vec.get(key, default_value);
  }

  void set(
      const size_t key,
      const double integral,
      const std::function<void(double&, const double&)>& reducer) {
    if (hash_integrals)
      hash.set(key, integral, reducer);
    else
      vec.set(key, integral, reducer);
  }

  void clear() {
    if (hash_integrals)
      hash.clear();
    else
      vec.clear();
  }

  template <class B>
  void serialize(B& buf) const {
    if (hash_integrals)
      hash.serialize(buf);
    else
      vec.serialize(buf);
  }

  template <class B>
  void parse(B& buf) {
    if (hash_integrals)
      hash.parse(buf);
    else
      vec.parse(buf);
  }

  size_t num_elements() const {
    if (hash_integrals)
      throw std::runtime_error("HashMap doesn't have a public num_elements() implementation\n");
    else
      return vec.num_elements();
  }

  void set_storage(bool bval) { hash_integrals = bval; }

 private:
  bool hash_integrals = false;

  fgpl::HashMap<size_t, double, IntegralsHasher> hash;
  VectorStorage vec;
};
