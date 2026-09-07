# B-Tree

This library implements a general-purpose header-only STL-like B-Tree in C++, including supports for using it for memory-mapped disk files and fixed-size allocators.

A B-Tree is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time. Unlike other self-balancing binary search trees, the [B-tree](https://en.wikipedia.org/wiki/B-tree) is well suited for storage systems that read and write relatively large blocks of data, such as databases and file systems

Just like ordered associative containers in the C++ standard library, key-value pairs can be supported and duplicates can be allowed.

There are four specialized B-Tree classes: ```frozenca::BTreeSet```, ```frozenca::BTreeMultiSet```, ```frozenca::BTreeMap``` and ```frozenca::BTreeMultiMap```, which corresponds to ```std::set```, ```std::multiset```, ```std::map``` and ```std::multimap``` respectively.

## How to use

This library is header-only, so no additional setup process is required beyond including the headers.

Or

For cmake projects:

Install one of package BTree..rpm or BTree..deb or include this project into yours and then

 ```cmake
 find_package(BTree)
 #...
 target_link_libraries(${your_target} PRIVATE BTree::BTree)
 ```


## Target OS/Compiler version

This library aggressively uses C++20 features, and verified to work in gcc 11.2 and MSVC 19.32.

POSIX and Windows operating systems are supported in order to use the memory-mapped disk file interface.

There are currently no plans to support C++17 and earlier.

## Example usages

Usage is very similar to the C++ standard library ordered associative containers (i.e. ```std::set``` and its friends)

```cpp
#include "fc/btree.h"
#include <iostream>
#include <string>

int main() {
  namespace fc = frozenca;
  fc::BTreeSet<int> btree;
  
  btree.insert(3);
  btree.insert(4);
  btree.insert(2);
  btree.insert(1);
  btree.insert(5);
  
  // 1 2 3 4 5
  for (auto num : btree) {
    std::cout << num << ' ';
  }
  std::cout << '\n';
  
  fc::BTreeMap<std::string, int> strtree;

  strtree["asd"] = 3;
  strtree["a"] = 6;
  strtree["bbb"] = 9;
  strtree["asdf"] = 8;
  
  for (const auto &[k, v] : strtree) {
    std::cout << k << ' ' << v << '\n';
  }

  strtree["asdf"] = 333;
  
  // 333
  std::cout << strtree["asdf"] << '\n';

  strtree.emplace("asdfgh", 200);
  for (const auto &[k, v] : strtree) {
    std::cout << k << ' ' << v << '\n';
  }
}
```

You can refer more example usages in ```test/unittest.cpp```.

Users can specify a fanout parameter for B-tree: the default is 64.

```cpp
  // btree with fanout 128
  fc::BTreeSet<int, 128> btree;
```

The smallest possible value for fanout is 2, where a B-Tree boils down to an [2-3-4 tree](https://en.wikipedia.org/wiki/2%E2%80%933%E2%80%934_tree) 

## Supported operations

Other than regular operations supported by ```std::set``` and its friends (```lower_bound()```, ```upper_bound()```, ```equal_range()``` and etc), the following operations are supported.

```tree.count(const key_type& key)``` : Returns the number of elements in the tree for their key is equivalent to ```key```. Time complexity: ```O(log n)```

```tree.kth(std::ptrdiff_t k)``` : Returns the k-th element in the tree as 0-based index. Time complexity: ```O(log n)```

```tree.order(const_iterator_type iter)``` : Returns the rank of the element in the iterator in the tree as 0-based index. Time complexity: ```O(log n)```

```tree.enumerate(const key_type& a, const key_type& b)``` : Range query. Returns the range of values for their key in ```[a, b]```. Time complexity: ```O(log n)```

```tree.insert_range(ForwardIter first, ForwardIter last)``` : Inserts the elements in ```[first, last)```. The range version also exists. Time complexity: ```O(k log k + log n)``` if all of elements in the range can be fit between two elements in the tree, otherwise ```O(k log n)```

```tree.erase_range(const key_type& a, const key_type&)``` : Erases the elements for their key in ```[a, b]```. Time complexity: ```O(log n) + O(k)``` (NOT ```O(k log n)```)

```frozenca::join(Tree&& tree1, Tree&& tree2)``` : Joins two trees to a single tree. The largest key in ```tree1``` should be less than or equal to the smallest key in ```tree2```. Time complexity: ```O(log n)```

```frozenca::join(Tree&& tree1, value_type val, Tree&& tree2)``` : Joins two trees to a single tree. The largest key in ```tree1``` should be less than or equal to the key of ```val``` and the smallest key in ```tree2``` should be greater than or equal to the key of ```val```. Time complexity: ```O(1 + diff_height)```

```frozenca::split(Tree&& tree, key_type key)``` : Splits a tree to two trees, so that the first tree contains keys less than ```key```, and the second tree contains keys greater than ```key```. Time complexity: ```O(log n)```

```frozenca::split(Tree&& tree, key_type key1, key_type key2)``` : Splits a tree to two trees, so that the first tree contains keys less than ```key1```, and the second tree contains keys greater than ```key2```. ```key2``` must be greater than or equal to ```key1```. Time complexity: ```O(log n) + O(k)```

## Iterators
STL compatible iterators are fully supported. (both ```const``` and non-```const```) However, unlike ```std::set``` and its friends, all insert and erase operations can invalidate iterators. This is because ```std::set``` and its friends are node-based containers where a single node can only have a single key, but a node in B-Trees can have multiple keys.

## Concurrency

Currently, thread safety is not guaranteed. Lock-free support is the first TODO, but contributions are welcome if you're interested.

## Linear search vs Binary search

The core operation for B-Tree is a search in the sorted key array of each node. For small arrays with primitive key types that have relatively cheap comparisons, linear search is often better than binary search. This threshold may vary by compiler by a big margin.

If you use Clang, I recommend that you set this variable to 1. For gcc users, it seems better not to change the variable (may be changed by future gcc optimizations)
https://github.com/frozenca/BTree/blob/7083e8034b5905552cc6a3b8277452c56c05d587/fc_btree.h#L22

## SIMD Operation

When keys are signed integers or floating point types, if your machine supports AVX-512, you can activate SIMD intrinsics to speed up B-Tree operations, by setting this variable to 1:
https://github.com/frozenca/BTree/blob/3498a53e75e916015561008cf91fecc3f7df69d1/fc_btree.h#L4
(Inspired from: [Static B-Trees](https://en.algorithmica.org/hpc/data-structures/s-tree/))

## Disk B-Tree

You can use a specialized variant that utilizes memory-mapped disk files and an associated fixed-size allocator. You have to include ```fc_disk_btree.h```, ```fc_disk_fixed_alloc.h``` and ```fc_mmfile.h``` to use it.

For this variant, supported types have stricter type constraints: it should satisfy ```std::trivially_copyable_v```, and its alignment should at least be the alignment of the pointer type in the machine (for both key type and value type for key-value pairs).

The following code initializes a ```frozenca::DiskBTreeSet```, which generates a memory-mapped disk file ```database.bin``` and uses it, with an initial byte size of 32 megabytes. If the third argument is ```true```, it will destroy the existing file and create a new one (default is ```false```). You can't extend the pool size of the memory-mapped disk file once you initialized (doing so invalidates all pointers in the associated allocator).

```cpp
fc::DiskBTreeSet<std::int64_t, 128> btree("database.bin", 1UL << 25UL, true);
```

## Serialization and deserialization

Serialization/deserialization of B-Trees via byte streams using ```operator<<``` and ```operator>>``` is also supported when key types (and value types, if present) meet the above requirements for disk B-Tree. You can refer how to do serialization/deserialization in ```test/rwtest.cpp```.

## Performance

Using a performance test code similar with ```test/perftest.cpp```, that inserts/retrieves/erases 1 million ```std::int64_t``` in random order, I see the following results in my machine (gcc 11.2, -O3, 200 times repeated per each target), compared to ```std::set``` and Google's B-Tree implementation(https://code.google.com/archive/p/cpp-btree/):

```
Balanced tree test
Warming up complete...
frozenca::BTreeSet test (fanout 64 - default, SIMD)
Time to insert 1000000 elements: Average : 175.547ms, Stdev   : 8.65575ms, 95%     : 189.553ms,
Time to lookup 1000000 elements: Average : 197.75ms, Stdev   : 7.75456ms, 95%     : 208.783ms,
Time to erase 1000000 elements: Average : 211.274ms, Stdev   : 10.3499ms, 95%     : 225.221ms,

frozenca::BTreeSet test (fanout 96, SIMD)
Time to insert 1000000 elements: Average : 176.432ms, Stdev   : 9.12931ms, 95%     : 192.688ms,
Time to lookup 1000000 elements: Average : 194.997ms, Stdev   : 11.3563ms, 95%     : 205.048ms,
Time to erase 1000000 elements: Average : 212.86ms, Stdev   : 11.3598ms, 95%     : 228.145ms,

frozenca::DiskBTreeSet test (fanout 128, SIMD)
Time to insert 1000000 elements: Average : 187.797ms, Stdev   : 8.69872ms, 95%     : 202.318ms,
Time to lookup 1000000 elements: Average : 200.799ms, Stdev   : 7.10905ms, 95%     : 211.436ms,
Time to erase 1000000 elements: Average : 216.105ms, Stdev   : 6.83771ms, 95%     : 228.9ms,

frozenca::BTreeSet test (fanout 128, SIMD)
Time to insert 1000000 elements: Average : 189.536ms, Stdev   : 15.3073ms, 95%     : 221.393ms,
Time to lookup 1000000 elements: Average : 204.741ms, Stdev   : 17.8811ms, 95%     : 232.494ms,
Time to erase 1000000 elements: Average : 219.17ms, Stdev   : 20.6449ms, 95%     : 244.232ms,

frozenca::BTreeSet test (fanout 64, uint64, don't use SIMD)
Time to insert 1000000 elements: Average : 204.187ms, Stdev   : 57.3915ms, 95%     : 222.939ms,
Time to lookup 1000000 elements: Average : 221.049ms, Stdev   : 25.3429ms, 95%     : 245.708ms,
Time to erase 1000000 elements: Average : 249.832ms, Stdev   : 52.1106ms, 95%     : 288.095ms,

std::set test
Time to insert 1000000 elements: Average : 907.104ms, Stdev   : 43.7566ms, 95%     : 966.12ms,
Time to lookup 1000000 elements: Average : 961.859ms, Stdev   : 30.1132ms, 95%     : 1019.59ms,
Time to erase 1000000 elements: Average : 990.027ms, Stdev   : 37.1807ms, 95%     : 1049.58ms,

Google btree::btree_set test (fanout 64)
Time to insert 1000000 elements: Average : 425.071ms, Stdev   : 13.117ms, 95%     : 434.819ms,
Time to lookup 1000000 elements: Average : 377.009ms, Stdev   : 15.2407ms, 95%     : 385.736ms,
Time to erase 1000000 elements: Average : 421.514ms, Stdev   : 17.3882ms, 95%     : 432.955ms,

Google btree::btree_set test (fanout 256 - default value)
Time to insert 1000000 elements: Average : 251.597ms, Stdev   : 14.3492ms, 95%     : 289.579ms,
Time to lookup 1000000 elements: Average : 235.204ms, Stdev   : 11.8999ms, 95%     : 255.495ms,
Time to erase 1000000 elements: Average : 250.782ms, Stdev   : 12.1752ms, 95%     : 270.575ms,
```

For 1 million ```std::string```s with length 1~50, I see the following results in my machine:
```
frozenca::BTreeSet test (fanout 64 - default, std::string)
Time to insert 1000000 elements: Average : 1519.62ms, Stdev   : 81.3793ms, 95%     : 1685.13ms,
Time to lookup 1000000 elements: Average : 1188.33ms, Stdev   : 83.8154ms, 95%     : 1392.47ms,
Time to erase 1000000 elements: Average : 1570.44ms, Stdev   : 93.771ms, 95%     : 1747.73ms,

frozenca::BTreeSet test (fanout 128, std::string)
Time to insert 1000000 elements: Average : 1774.12ms, Stdev   : 41.601ms, 95%     : 1812.62ms,
Time to lookup 1000000 elements: Average : 1089.02ms, Stdev   : 22.8206ms, 95%     : 1127.83ms,
Time to erase 1000000 elements: Average : 1670.09ms, Stdev   : 24.2791ms, 95%     : 1711.33ms,

std::set test (std::string)
Time to insert 1000000 elements: Average : 1662.92ms, Stdev   : 178.644ms, 95%     : 1861.37ms,
Time to lookup 1000000 elements: Average : 1666.16ms, Stdev   : 127.095ms, 95%     : 1845.49ms,
Time to erase 1000000 elements: Average : 1639.79ms, Stdev   : 82.7256ms, 95%     : 1770.9ms,
```


## Sanity check and unit test

If you want to contribute and test the code, pay attention and use macro _CONTROL_IN_TEST, which will do full sanity checks on the entire tree:

https://github.com/frozenca/BTree/blob/adf3c3309f45a65010d767df674c232c12f5c00a/fc_btree.h#L350
https://github.com/frozenca/BTree/blob/adf3c3309f45a65010d767df674c232c12f5c00a/fc_btree.h#L531-#L532

and by running ```test/unittest.cpp``` you can verify basic operations.


## License

This library is licensed under either of Apache License Version 2.0 with LLVM Exceptions (LICENSE-Apache2-LLVM or https://llvm.org/foundation/relicensing/LICENSE.txt) or Boost Software License Version 1.0 (LICENSE-Boost or https://www.boost.org/LICENSE_1_0.txt).
