#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <json/single_include/nlohmann/json.hpp>
#include <sstream>
#include "util.h"

class Result;

class Config {
 public:
  static Config& get_instance() {
    static Config instance;
    return instance;
  }

  static void print() { std::cout << get_instance().data.dump(2) << std::endl; }

  template <class T>
  static T get(const std::string& key) {
    auto node_ref = std::cref(get_instance().data);
    std::istringstream key_stream(key);
    std::string key_elem;
    try {
      while (std::getline(key_stream, key_elem, '.')) {
        if (!key_elem.empty()) {
          node_ref = std::cref(node_ref.get().at(key_elem));
        }
      }
      return node_ref.get().get<T>();
    } catch (...) {
      throw std::runtime_error(Util::str_printf("Cannot find '%s' in config.json", key.c_str()));
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

 private:
  Config() {
    std::ifstream config_file("config.json");
    if (!config_file) throw std::runtime_error("cannot open config.json");
    config_file >> data;
  }

  nlohmann::json data;

  friend class Result;
};
