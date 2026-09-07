#if defined(__x86_64__) || defined(_M_X64)
#if defined(FC_USE_SIMD)
#undef FC_USE_SIMD
#define FC_USE_SIMD 1
#endif  // FC_USE_SIMD
#endif

#include "fc_catch2.h"
#include "fc/disk_btree.h"
#include <set>
#include <vector>
#include <unordered_map>
#include <sstream>
#include "test_statistics.h"

TEST_CASE("perftest") {
  namespace fc = frozenca;
  std::unordered_map<std::string, perf_result> result;

  auto generate_values = []() {
    std::vector<fc::BTreeSet<std::int64_t>::value_type> v(1'000'000);
    std::iota(v.begin(), v.end(), 0);
    return v;
  };

  BENCHMARK("Balanced tree test - warmap") {
    fc::BTreeSet<std::int64_t> btree;
    tree_perf_test(btree, generate_values());
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 64)") {
    fc::BTreeSet<std::int64_t> btree;
    result.emplace("BTreeSet(64)", tree_perf_test(btree, generate_values()));
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 96)") {
    fc::BTreeSet<std::int64_t, 96> btree;
    result.emplace("BTreeSet(96)", tree_perf_test(btree, generate_values()));
  };
  BENCHMARK("frozenca::DiskBTreeSet test (fanout 128)") {
    fc::DiskBTreeSet<std::int64_t, 128> btree("database.bin", 1UL << 25UL, true);
    result.emplace("DiskBTreeSet(128)", tree_perf_test(btree, generate_values()));
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 128)") {
    fc::BTreeSet<std::int64_t, 128> btree;
    result.emplace("BTreeSet(128)", tree_perf_test(btree, generate_values()));
  };
  BENCHMARK("std::set test") {
    std::set<std::int64_t> rbtree;
    result.emplace("std::set", tree_perf_test(rbtree, generate_values()));
  };

  for (const auto &[key, value] : result) {
    INFO("----------------");
    INFO(key);
    std::stringstream ss;
    value.print_stats(ss);
    INFO(ss.str());
  }
}
