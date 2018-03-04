#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class Config {
 public:
  static Config& get_instance() {
    static Config instance;
    return instance;
  }

  static void print() { boost::property_tree::json_parser::write_json(std::cout, config_tree); }

  template <class T>
  static void get(const std::string& key) {
    return config_tree.get<T>(key);
  }

  template <class T>
  static void get(const std::string& key, const T& default_value) {
    return config_tree.get<T>(key, default_value);
  }

  template <class T>
  static void get_vector(const std::string& key) {
    std::vector<T> res;
    for (auto& item : config_tree.get_child(key)) res.push_back(item);
    return res;
  }

 private:
  Config() { boost::property_tree::read_json("config.json", config_tree); }

  boost::property_tree::ptree config_tree;
};
