#ifndef __FC_DISK_BTREE_H__
#define __FC_DISK_BTREE_H__

#include "fc/btree.h"
#include "fc/disk_fixed_alloc.h"

namespace frozenca {

template <DiskAllocable K, DiskAllocable V, attr_t Fanout, typename Comp,
          bool AllowDup>
class DiskBTreeBase
    : public BTreeBase<K, V, Fanout, Comp, AllowDup, AllocatorFixed> {
public:
  using Base = BTreeBase<K, V, Fanout, Comp, AllowDup, AllocatorFixed>;
  using Node = typename Base::node_type;

private:
  MemoryMappedFile mm_file_;
  MemoryResourceFixed<Node> mem_res_;

public:
  explicit DiskBTreeBase(const MemoryMappedFile &mm_file)
      : mm_file_{mm_file}, mem_res_{reinterpret_cast<unsigned char *>(
                                        mm_file_.data()),
                                    mm_file_.size()} {
    Base(AllocatorFixed<Node>(&mem_res_));
  }

  DiskBTreeBase(const std::filesystem::path &path, std::size_t pool_size,
                bool trunc = false)
      : mm_file_{path, pool_size, trunc}, mem_res_{
                                              reinterpret_cast<unsigned char *>(
                                                  mm_file_.data()),
                                              mm_file_.size()} {
    Base(AllocatorFixed<Node>(&mem_res_));
  }
};

template <DiskAllocable K, attr_t t = 64, typename Comp = std::ranges::less>
using DiskBTreeSet = DiskBTreeBase<K, K, t, Comp, false>;

template <DiskAllocable K, attr_t t = 64, typename Comp = std::ranges::less>
using DiskBTreeMultiSet = DiskBTreeBase<K, K, t, Comp, true>;

template <DiskAllocable K, DiskAllocable V, attr_t t = 64,
          typename Comp = std::ranges::less>
using DiskBTreeMap = DiskBTreeBase<K, BTreePair<K, V>, t, Comp, false>;

template <DiskAllocable K, DiskAllocable V, attr_t t = 64,
          typename Comp = std::ranges::less>
using DiskBTreeMultiMap = DiskBTreeBase<K, BTreePair<K, V>, t, Comp, true>;

} // namespace frozenca

#endif //__FC_DISK_BTREE_H__
