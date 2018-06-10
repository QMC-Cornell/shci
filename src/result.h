#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <json/single_include/nlohmann/json.hpp>
#include "parallel.h"

class Result {
 public:
  static Result& get_instance() {
    static Result instance;
    return instance;
  }

  static void dump() {
    if (!Parallel::is_master()) return;
    const auto& instance = get_instance();
    std::ofstream result_file("result.json");
    result_file << instance.data.dump(2) << std::endl;
  }

  template <class T>
  static T get(const std::string& key) {
    auto node_ref = std::cref(get_instance().data);
    std::istringstream key_stream(key);
    std::string key_elem;
    try {
      while (std::getline(key_stream, key_elem, '/')) {
        if (!key_elem.empty()) {
          node_ref = std::cref(node_ref.get().at(key_elem));
        }
      }
      return node_ref.get().get<T>();
    } catch (...) {
      throw std::runtime_error(Util::str_printf("Cannot find '%s' in result.json", key.c_str()));
    }
  }

  template <class T>
  static T get(const std::string& key, const T& default_value) {
    try {
      return get<T>(key);
    } catch (...) {
      return default_value;
    }
  }

  template <class T>
  static void put(const std::string& key, const T& value) {
    auto node_ref = std::ref(get_instance().data);
    std::istringstream key_stream(key);
    std::string key_elem;
    while (std::getline(key_stream, key_elem, '/')) {
      if (!key_elem.empty()) {
        node_ref = std::ref(node_ref.get()[key_elem]);
      }
    }
    node_ref.get() = value;
    dump();
  }

 private:
  Result() {
    std::ifstream result_file("result.json");
    if (result_file) {
      result_file >> data;
    }
    data["config"] = Config::get_instance().data;
    result_file.close();
  }

  nlohmann::json data;
};
