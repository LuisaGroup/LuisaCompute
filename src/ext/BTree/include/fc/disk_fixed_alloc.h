#ifndef __FC_DISK_FIXED_ALLOC_H__
#define __FC_DISK_FIXED_ALLOC_H__

#include "fc/mmfile.h"
#include "fc/details.h"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iostream>
#if defined(__clang__) && __clang_major__ < 15
#include <experimental/memory_resource>
namespace stdpmr = std::experimental::pmr;
#elif defined(__clang__) || (__GNUC__)
#include <memory_resource>
namespace stdpmr = std::pmr;
#endif
#include <stdexcept>
#include <type_traits>

namespace frozenca {

template <typename T>
class MemoryResourceFixed : public stdpmr::memory_resource {
  T *pool_ = nullptr;
  std::size_t pool_size_ = 0;
  T *free_ = nullptr;

public:
  MemoryResourceFixed(unsigned char *pool_ptr, std::size_t pool_byte_size) {
    if (!pool_ptr) {
      throw std::invalid_argument("pool ptr is null");
    }
    if ((std::bit_cast<std::size_t>(pool_ptr) % alignof(T)) ||
        (std::bit_cast<std::size_t>(pool_ptr) % sizeof(T *))) {
      throw std::invalid_argument("pool ptr is not aligned with T/T*");
    }
    if (pool_byte_size < sizeof(T)) {
      throw std::invalid_argument("pool byte size is too small");
    }
    if ((pool_byte_size % alignof(T)) || (pool_byte_size % sizeof(T *))) {
      throw std::invalid_argument("pool byte size is not aligned with T/T*");
    }

    pool_ = reinterpret_cast<T *>(pool_ptr);
    // size in chunks
    pool_size_ = pool_byte_size / sizeof(T);

    auto curr_chunk = pool_;
    for (size_t i = 0; i < pool_size_; i++, curr_chunk++) {
      *(reinterpret_cast<uint64_t *>(curr_chunk)) =
          std::bit_cast<uint64_t>(curr_chunk + 1);
    }
    free_ = pool_;
  }

  explicit MemoryResourceFixed(MemoryMappedFile &file)
      : MemoryResourceFixed(static_cast<unsigned char *>(file.data()),
                            file.size()) {}

private:
  void *do_allocate([[maybe_unused]] std::size_t num_bytes,
                    [[maybe_unused]] std::size_t alignment) override {
    if (free_ == pool_ + pool_size_) {
      throw std::invalid_argument("fixed allocator out of memory");
    } else {
      auto x = free_;
      free_ = std::bit_cast<T *>(*(reinterpret_cast<uint64_t *>(x)));
      return reinterpret_cast<void *>(x);
    }
  }

  void do_deallocate(void *x, [[maybe_unused]] std::size_t num_bytes,
                     [[maybe_unused]] std::size_t alignment) override {
    auto x_chunk = reinterpret_cast<T *>(x);
    *(reinterpret_cast<uint64_t *>(x)) = std::bit_cast<uint64_t>(free_);
    free_ = x_chunk;
  }

  [[nodiscard]] bool
  do_is_equal(const stdpmr::memory_resource &other) const noexcept override {
    if (this == &other) {
      return true;
    }
    auto op = dynamic_cast<const MemoryResourceFixed *>(&other);
    return op && op->pool_ == pool_ && op->pool_size_ == pool_size_ &&
           op->free_ == free_;
  }
};

template <typename T> class AllocatorFixed {
  stdpmr::memory_resource *mem_res_;

public:
  template <typename Other> struct rebind {
    using other = AllocatorFixed<Other>;
  };

  using value_type = T;

  explicit AllocatorFixed(
      stdpmr::memory_resource *mem_res = stdpmr::get_default_resource())
      : mem_res_{mem_res} {}

  template <typename Other>
  AllocatorFixed(const AllocatorFixed<Other> &other)
      : AllocatorFixed(other.get_memory_resource()) {}

  T *allocate(size_t n) {
    return reinterpret_cast<T *>(
        mem_res_->allocate(sizeof(T) * n, std::alignment_of_v<T>));
  }

  void deallocate(T *ptr, size_t n) {
    mem_res_->deallocate(reinterpret_cast<void *>(ptr), sizeof(T) * n,
                         std::alignment_of_v<T>);
  }

  [[nodiscard]] stdpmr::memory_resource *
  get_memory_resource() const noexcept {
    return mem_res_;
  }
};

template <typename> struct isDiskAlloc : std::false_type {};

template <DiskAllocable T>
struct isDiskAlloc<AllocatorFixed<T>> : std::true_type {};

} // namespace frozenca

#endif //__FC_DISK_FIXED_ALLOC_H__
