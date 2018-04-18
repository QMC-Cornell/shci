#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <json/single_include/nlohmann/json.hpp>

class Result {
 public:
  static Result& get_instance() {
    static Result instance;
    return instance;
  }

  static void dump() {
    std::ofstream result_file("result.json");
    result_file << get_instance().data.dump(2) << std::endl;
  }

  template <class T>
  static void put(const std::string& key, const T& value) {
    auto node_ref = std::ref(get_instance().data);
    std::istringstream key_stream(key);
    std::string key_elem;
    while (std::getline(key_stream, key_elem, '.')) {
      if (!key_elem.empty()) {
        node_ref = std::ref(node_ref.get()[key_elem]);
      }
    }
    node_ref.get() = value;
  }

 private:
  Result() { data["config"] = Config::get_instance().data; }

  nlohmann::json data;
};
