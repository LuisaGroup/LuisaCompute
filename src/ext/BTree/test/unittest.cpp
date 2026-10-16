#define _CONTROL_IN_TEST

#ifndef CATCH_CONFIG_MAIN
#define CATCH_CONFIG_MAIN
#endif // CATCH_CONFIG_MAIN

#include "fc_catch2.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "fc/btree.h"

namespace fc = frozenca;

TEST_CASE("BTree insert-lookup-erase") {
  fc::BTreeSet<int> btree;
  constexpr int n = 100;

  std::mt19937 gen(std::random_device{}());

  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), 0);
  SECTION("Random insert") {
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      btree.insert(num);
    }
  }
  SECTION("Random lookup") {
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      btree.insert(num);
    }
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      REQUIRE(btree.contains(num));
    }
  }
  SECTION("Random erase") {
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      btree.insert(num);
    }
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      REQUIRE(btree.erase(num));
    }
  }
}

TEST_CASE("BTree std::initializer_list-test") {
  fc::BTreeSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
  REQUIRE(btree.size() == 7);
}

TEST_CASE("Multiset test") {
  fc::BTreeMultiSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
  REQUIRE(btree.size() == 9);
  REQUIRE_NOTHROW(btree.erase(3));
  REQUIRE(btree.size() == 6);
}

TEST_CASE("Order statistic test") {
  fc::BTreeSet<int> btree;
  constexpr int n = 100;

  for (int i = 0; i < n; ++i) {
    REQUIRE_NOTHROW(btree.insert(i));
  }

  for (int i = 0; i < n; ++i) {
    REQUIRE(btree.kth(i) == i);
  }

  for (int i = 0; i < n; ++i) {
    REQUIRE(btree.order(btree.find(i)) == i);
  }
}

TEST_CASE("Enumerate") {
  fc::BTreeSet<int> btree;
  constexpr int n = 100;

  for (int i = 0; i < n; ++i) {
    REQUIRE_NOTHROW(btree.insert(i));
  }
  auto rg = btree.enumerate(20, 30);
  REQUIRE(std::ranges::distance(rg.begin(), rg.end()) == 11);

  SECTION("erase_if test") {
    REQUIRE_NOTHROW(btree.erase_if([](auto n) { return n >= 20 && n <= 90; }));
    REQUIRE(btree.size() == 29);
  }
}

TEST_CASE("BTreeMap") {
  fc::BTreeMap<std::string, int> btree;

  REQUIRE_NOTHROW(btree["asd"] = 3);
  REQUIRE_NOTHROW(btree["a"] = 6);
  REQUIRE_NOTHROW(btree["bbb"] = 9);
  REQUIRE_NOTHROW(btree["asdf"] = 8);
  REQUIRE_NOTHROW(btree["asdf"] = 333);
  REQUIRE(btree["asdf"] == 333);

  REQUIRE_NOTHROW(btree.emplace("asdfgh", 200));
  REQUIRE(btree["asdfgh"] == 200);
}

TEST_CASE("Join/Split") {
  fc::BTreeSet<int> btree1;
  for (int i = 0; i < 100; ++i) {
    REQUIRE_NOTHROW(btree1.insert(i));
  }

  fc::BTreeSet<int> btree2;
  for (int i = 101; i < 300; ++i) {
    REQUIRE_NOTHROW(btree2.insert(i));
  }
  fc::BTreeSet<int> btree3;

  REQUIRE_NOTHROW(
      btree3 = fc::join(std::move(btree1), 100, std::move(btree2)));

  for (int i = 0; i < 300; ++i) {
    REQUIRE(btree3.contains(i));
  }

  auto [btree4, btree5] = fc::split(std::move(btree3), 200);
  for (int i = 0; i < 200; ++i) {
    REQUIRE(btree4.contains(i));
  }
  REQUIRE_FALSE(btree5.contains(200));

  for (int i = 201; i < 300; ++i) {
    REQUIRE(btree5.contains(i));
  }
}

TEST_CASE("Multiset split") {
  fc::BTreeMultiSet<int> btree6;
  REQUIRE_NOTHROW(btree6.insert(0));
  REQUIRE_NOTHROW(btree6.insert(2));
  for (int i = 0; i < 100; ++i) {
    REQUIRE_NOTHROW(btree6.insert(1));
  }
  auto [btree7, btree8] = fc::split(std::move(btree6), 1);
  REQUIRE(btree7.size() == 1);
  REQUIRE(btree8.size() == 1);
}

TEST_CASE("Two arguments join") {
  fc::BTreeSet<int> tree1;
  for (int i = 0; i < 100; ++i) {
    REQUIRE_NOTHROW(tree1.insert(i));
  }
  fc::BTreeSet<int> tree2;
  for (int i = 100; i < 200; ++i) {
    REQUIRE_NOTHROW(tree2.insert(i));
  }
  auto tree3 = fc::join(std::move(tree1), std::move(tree2));
  for (int i = 0; i < 200; ++i) {
    REQUIRE(tree3.contains(i));
  }
}

TEST_CASE("Three arguments split") {
  fc::BTreeSet<int> tree1;
  for (int i = 0; i < 100; ++i) {
    tree1.insert(i);
  }
  auto [tree2, tree3] = fc::split(std::move(tree1), 10, 80);
  REQUIRE(tree2.size() == 10);
  REQUIRE(tree3.size() == 19);
}

TEST_CASE("Multiset erase") {
  fc::BTreeMultiSet<int> tree1;
  REQUIRE_NOTHROW(tree1.insert(0));
  for (int i = 0; i < 100; ++i) {
    REQUIRE_NOTHROW(tree1.insert(1));
  }
  REQUIRE_NOTHROW(tree1.insert(2));

  REQUIRE_NOTHROW(tree1.erase(1));

  REQUIRE(tree1.size() == 2);
}

TEST_CASE("Range insert-1") {
  fc::BTreeSet<int> btree;
  REQUIRE_NOTHROW(btree.insert(1));
  REQUIRE_NOTHROW(btree.insert(10));

  std::vector<int> v{2, 5, 4, 3, 7, 6, 6, 6, 2, 8, 8, 9};
  REQUIRE_NOTHROW(btree.insert_range(std::move(v)));

  for (int i = 1; i < 10; ++i) {
    REQUIRE(btree.contains(i));
  }
}

TEST_CASE("Range insert-2") {
  fc::BTreeSet<int> btree;
  REQUIRE_NOTHROW(btree.insert(1));
  REQUIRE_NOTHROW(btree.insert(10));

  std::vector<int> v{2, 5, 4, 3, 7, 6, 6, 6, 2, 8, 8, 9, 10};
  REQUIRE_NOTHROW(btree.insert_range(std::move(v)));

  for (int i = 1; i < 10; ++i) {
    REQUIRE(btree.contains(i));
  }
}

TEST_CASE("count()") {
  fc::BTreeMultiSet<int> btree2;
  REQUIRE_NOTHROW(btree2.insert(1));
  REQUIRE_NOTHROW(btree2.insert(1));
  REQUIRE(btree2.count(1) == 2);
  REQUIRE(btree2.count(0) == 0);
  REQUIRE(btree2.count(2) == 0);
}
