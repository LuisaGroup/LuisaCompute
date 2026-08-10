#if defined(FC_USE_SIMD)
#undef FC_USE_SIMD
#define FC_USE_SIMD 0
#endif  // FC_USE_SIMD

#include "fc_catch2.h"
#include "fc/btree.h"
#include <iostream>
#include <set>
#include <vector>
#include <sstream>
#include "test_statistics.h"


TEST_CASE("perftest-string") {
  namespace fc = frozenca;
  constexpr int max_n = 1'000'000;
  constexpr int max_length = 50;
  
  std::unordered_map<std::string, perf_result> result;
  auto str_vec = generate_random_strings(max_n, max_length, false);

  BENCHMARK("Balanced tree test - warmap") {
    fc::BTreeSet<std::string, 64> btree;
    tree_perf_test(btree, str_vec);
  };

  BENCHMARK("frozenca::BTreeSet string (fanout 64)") {
    fc::BTreeSet<std::string, 64> btree;
    result.emplace("BTreeSet(64)", tree_perf_test(btree, str_vec));
  };
  BENCHMARK("frozenca::BTreeSet string (fanout 128)") {
    fc::BTreeSet<std::string, 128> btree;
    result.emplace("BTreeSet(128)", tree_perf_test(btree, str_vec));
  };
  BENCHMARK("std::set string") {
    std::set<std::string> rbtree;
    result.emplace("std::set", tree_perf_test(rbtree, str_vec));
  };
  
  for (const auto &[key, value] : result) {
    INFO("----------------");
    INFO(key);
    std::stringstream ss;
    value.print_stats(ss);
    INFO(ss.str());
  }
}
