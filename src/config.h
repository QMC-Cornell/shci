#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

class Result;

class Config {
 public:
  static Config& get_instance() {
    static Config instance;
    return instance;
  }

  static void print() { std::cout << std::setw(2) << get_instance().data << std::endl; }

  template <class T>
  static T get(const std::string& key) {
    auto node_ref = std::cref(get_instance().data);
    std::istringstream key_stream(key);
    std::string key_elem;
    while (std::getline(key_stream, key_elem, '.')) {
      if (!key_elem.empty()) {
        node_ref = std::cref(node_ref.get().at(key_elem));
      }
    }
    return node_ref.get().get<T>();
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
    if (!config_file.good()) throw std::runtime_error("cannot load config.json");
    config_file >> data;
  }

  nlohmann::json data;

  friend class Result;
};
