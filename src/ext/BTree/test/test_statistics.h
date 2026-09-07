#ifndef FC_TEST_STATISTICS_H
#define FC_TEST_STATISTICS_H

#include "fc_catch2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct stats {
  float average = 0.0f;
  float stdev = 0.0f;
  float percentile_95 = 0.0f;
  float percentile_99 = 0.0f;
  float percentile_999 = 0.0f;
};

struct perf_result {
  size_t values_cnt{0};
  stats insert;
  stats find;
  stats erase;
  void print_stats(std::ostream &os) const;
};

stats get_statistics(std::vector<float> &v);

std::vector<std::string> generate_random_strings(int max_n, int max_length, bool allow_duplicates);

template <typename TreeType>
[[maybe_unused]] perf_result tree_perf_test(TreeType &tree, std::vector<typename TreeType::value_type> v, size_t trials = 1) {
  const size_t max_n = v.size();
  const size_t max_trials = trials;

  std::mt19937 gen(std::random_device{}());
  std::vector<float> durations_insert;
  std::vector<float> durations_find;
  std::vector<float> durations_erase;

  for (size_t t = 0; t < max_trials; ++t) {
    float duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      tree.insert(num);
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_insert.push_back(duration);

    duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      if (!tree.contains(num)) {
        FAIL("Lookup verification fail!");
      }
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_find.push_back(duration);

    duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      if (!tree.erase(num)) {
        FAIL("Erase verification fail!");
      }
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_erase.push_back(duration);
  }
  perf_result result;
  result.values_cnt = max_n;
  result.insert = get_statistics(durations_insert);
  result.find = get_statistics(durations_find);
  result.erase = get_statistics(durations_erase);
  return result;
}

#endif  // FC_TEST_STATISTICS_H
