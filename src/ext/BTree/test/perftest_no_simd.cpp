#if defined(FC_USE_SIMD)
#undef FC_USE_SIMD
#define FC_USE_SIMD 0
#endif  // FC_USE_SIMD

#include "fc_catch2.h"
#include "fc/disk_btree.h"
#include <set>
#include <vector>
#include <unordered_map>
#include <sstream>
#include "test_statistics.h"

TEST_CASE("perftest-no-simd") {
  namespace fc = frozenca;
  std::unordered_map<std::string, perf_result> result;

  auto generate_values = []() {
    std::vector<fc::BTreeSet<std::uint64_t>::value_type> v(1'000'000);
    std::iota(v.begin(), v.end(), 0);
    return v;
  };

  BENCHMARK("frozenca::BTreeSet test (don't use SIMD)") {
    fc::BTreeSet<std::uint64_t> btree;
    result.emplace("BTreeSet(no-simd)", tree_perf_test(btree, generate_values()));
  };

  for (const auto &[key, value] : result) {
    INFO("----------------");
    INFO(key);
    std::stringstream ss;
    value.print_stats(ss);
    INFO(ss.str());
  }
}
