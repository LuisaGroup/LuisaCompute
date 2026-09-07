#include "test_statistics.h"
#include <cmath>
#include <numeric>
#include <sstream>

stats get_statistics(std::vector<float> &v) {
  auto n = std::ssize(v);
  if (n == 0) {
    return {};
  }
  stats s;
  s.average = std::accumulate(v.begin(), v.end(), 0.0f) / n;
  float variance = 0.0f;
  for (auto value : v) {
    variance += std::pow(value - s.average, 2.0f);
  }
  variance /= n;
  s.stdev = std::sqrt(variance);
  std::ranges::sort(v);
  s.percentile_95 = *(v.begin() + (19 * n / 20));
  s.percentile_99 = *(v.begin() + (99 * n / 100));
  s.percentile_999 = *(v.begin() + (999 * n / 1000));
  return s;
}

void perf_result::print_stats(std::ostream &os) const {
  auto print = [this, &os](const std::string &stat_name, const stats &stat) {
    os << "\tTime to " << stat_name << " " << values_cnt << " elements:\n"
       << "\tAverage : " << stat.average << "ms,\n"
       << "\tStdev   : " << stat.stdev << "ms,\n"
       << "\t95%     : " << stat.percentile_95 << "ms,\n"
       << "\t99%     : " << stat.percentile_99 << "ms,\n"
       << "\t99.9%   : " << stat.percentile_999 << "ms,\n";
  };
  print("insert", insert);
  print("find", find);
  print("erase", erase);
}

std::vector<std::string> generate_random_strings(int max_n, int max_length, bool allow_duplicates) {
  std::vector<std::string> res;

  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> length_dist(1, max_length);
  std::uniform_int_distribution<int> ch_dist(32, 126);

  for (int i = 0; i < max_n; ++i) {
    int len = length_dist(gen);
    std::string s;
    for (int l = 0; l < len; ++l) {
      s += static_cast<char>(ch_dist(gen));
    }
    res.push_back(std::move(s));
  }

  if (!allow_duplicates) {
    std::ranges::sort(res);
    auto ret = std::ranges::unique(res);
    res.erase(ret.begin(), ret.end());
  }

  return res;
}
