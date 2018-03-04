#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

class Config {
 public:
  static Config& get_instance() {
    static Config instance;
    return instance;
  }

  static void print() {
    boost::property_tree::json_parser::write_json(std::cout, get_instance().config_tree);
  }

  template <class T>
  static T get(const std::string& key) {
    return get_instance().config_tree.get<T>(key);
  }

  template <class T>
  static T get(const std::string& key, const T& default_value) {
    return get_instance().config_tree.get<T>(key, default_value);
  }

  template <class T>
  static std::vector<T> get_vector(const std::string& key) {
    std::vector<T> res;
    for (auto& item : get_instance().config_tree.get_child(key)) res.push_back(item);
    return res;
  }

 private:
  Config() { boost::property_tree::read_json("config.json", config_tree); }

  boost::property_tree::ptree config_tree;
};
