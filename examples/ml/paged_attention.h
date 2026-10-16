// Multi-Head Latent Attention (MLA) Example -- reusable paged KV-cache driver.
//
// `PagedAttention` replaces the previously flattened host sparse-buffer
// management in attention_runner.cpp with a single class built from two
// focused modules:
//
//   * Allocation (`PagedAttention::PageAllocator`) -- a vLLM-BlockPool-style
//     allocator over the sparse-buffer heap pool. Physical residency is a
//     pool of `SparseBufferHeap`s (one per sparse tile). Because one physical
//     tile is not necessarily one attention page (the device chooses the
//     tile size), the allocator *custom-allocates* attention pages inside
//     tiles: pages_per_tile = tile_elems / page_elems, so each page owns a
//     (tile, page-in-tile) pair and its element offset in the pool. Freed
//     pages return to a free queue for reuse.
//
//   * Indexing (`_index_buffer` + the allocator page table) -- the block
//     table analog. The device-visible index buffer stores one *element
//     offset* per logical page (offsets, not page ids), so the kernels'
//     logical -> physical indirection works unchanged across pool
//     re-allocation, where physical pages move.
//
// The class is split into two phases:
//
//   * prepare()  -- geometry discovery, sparse resource registration (pool
//     segment creation, heap allocation, tile mapping), kernel compilation,
//     initial index-buffer upload.
//   * compute()  -- enqueues the reshape + paged-attention dispatches on a
//     caller-owned CommandList.
//
// Re-allocation: when the current sparse buffer cannot hold enough attention
// pages (grow / reserve), a larger buffer is created, all live pages are
// re-allocated (and copied) into it, and the retired buffer + its heaps are
// destroyed through `CommandList::add_callback` submitted on the same
// stream, so they are only released after the GPU finished every command
// that still references them.

#pragma once

#include <cstdint>

#include <luisa/core/stl.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/sparse_buffer.h>

#include "attention_kernels.h"

namespace mla {

class PagedAttention {
public:
    // -- Allocation module ---------------------------------------------------
    // Custom page allocator over the sparse-buffer heap pool.
    //
    // Physical pages are sparse tiles; attention pages are carved out of them
    // (pages_per_tile per tile), because one tile generally does not equal
    // one attention page. The allocator owns the logical -> physical page
    // table and a free queue of released physical pages (the vLLM BlockPool
    // block/free-queue analog). It performs no device calls: residency
    // mapping is driven by PagedAttention from the allocation results.
    class PageAllocator {
    public:
        // Physical page handle: the tile backing the page, the page slot
        // inside that tile, and the element offset in the pool.
        struct Page {
            uint32_t tile{};
            uint32_t page_in_tile{};
            uint32_t elem_offset{};
        };
        // Physical pages are never released before the segment dies, so the
        // allocator keeps a raw pointer with an externally guaranteed lifetime.
        static constexpr uint32_t null_page = ~uint32_t{0};

        PageAllocator() noexcept = default;
        // Provision `capacity` physical pages (must be a multiple of
        // pages_per_tile) laid out as capacity/pages_per_tile tiles of
        // tile_stride_elems elements each.
        PageAllocator(uint32_t capacity, uint32_t pages_per_tile,
                      uint32_t page_elems, uint32_t tile_stride_elems) noexcept;

        // Allocate one physical page from the free queue; null_page on exhaustion.
        [[nodiscard]] uint32_t allocate() noexcept;
        // Return a physical page to the free queue (idempotent).
        void free(uint32_t page) noexcept;
        // Logical -> physical binding (no residency side effects).
        void assign(uint32_t logical, uint32_t page) noexcept;
        [[nodiscard]] uint32_t page_of(uint32_t logical) const noexcept { return _table[logical]; }
        // Element offset of a logical page in the pool (what the device index
        // buffer stores); null_page when unmapped.
        [[nodiscard]] uint32_t offset_of(uint32_t logical) const noexcept;
        // Physical page geometry.
        [[nodiscard]] const Page &physical(uint32_t page) const noexcept { return _pages[page]; }
        [[nodiscard]] bool is_mapped(uint32_t page) const noexcept { return _mapped[page]; }
        void set_mapped(uint32_t page, bool mapped) noexcept { _mapped[page] = mapped; }

        [[nodiscard]] uint32_t capacity() const noexcept { return static_cast<uint32_t>(_pages.size()); }
        [[nodiscard]] uint32_t free_count() const noexcept { return _free_count; }

        // (Re)size the logical -> physical table (keeps existing assignments).
        void resize_table(uint32_t logical_pages) noexcept;
        // Drop the old physical-page set and adopt a new pool segment
        // (capacity pages, all free, table cleared).
        void reset(uint32_t capacity, uint32_t pages_per_tile,
                   uint32_t page_elems, uint32_t tile_stride_elems) noexcept;

    private:
        void build_pages(uint32_t capacity, uint32_t pages_per_tile,
                         uint32_t page_elems, uint32_t tile_stride_elems) noexcept;

        luisa::vector<Page> _pages;     // physical page table
        luisa::vector<bool> _mapped;    // residency flag per physical page
        luisa::vector<uint32_t> _table; // logical -> physical page
        luisa::vector<uint32_t> _free;  // free queue (vLLM BlockPool free_blocks)
        uint32_t _free_count{};         // entries in _free
    };

    // Discovered pool geometry (valid after a successful prepare()).
    struct Geometry {
        uint32_t tile_bytes{};      // device sparse tile granularity
        uint32_t tile_elems{};      // tile_bytes / sizeof(float)
        uint32_t pages_per_tile{};  // custom allocation: attention pages per tile
        uint32_t page_elems{};      // elements per attention page (used region)
        uint32_t page_stride{};     // element stride between pages within a tile
        uint32_t tokens_per_page{}; // vLLM block_size
        uint32_t pages_per_seq{};   // block-table row stride
        uint32_t total_pages{};     // logical pages initially bound
        uint32_t sub_tile{};        // shared-memory sub-tile (tokens)
    };

    // Declared here, defined in paged_attention.cpp: Segment is incomplete in
    // the header, so the implicit unique_ptr destructor of the ctor/dtor must
    // not be instantiated at include sites.
    PagedAttention(luisa::compute::Device &device,
                   luisa::compute::Stream &stream) noexcept;
    ~PagedAttention();
    PagedAttention(const PagedAttention &) = delete;
    PagedAttention &operator=(const PagedAttention &) = delete;
    PagedAttention(PagedAttention &&) = delete;
    PagedAttention &operator=(PagedAttention &&) = delete;

    // -- Prepare phase -------------------------------------------------------
    // Probe sparse support, discover geometry, provision the initial pool
    // segment (sparse buffers + heaps + tile mapping), bind
    // num_sequences * pages_per_seq logical pages to shuffled physical pages,
    // upload the index buffer, and compile both kernels.
    // Returns false when the backend lacks sparse-buffer support or the
    // geometry is degenerate; the caller should then fall back to dense MHA.
    bool prepare(uint32_t num_sequences);

    // -- Compute phase -------------------------------------------------------
    // Append the reshape (dense K/V -> paged pool scatter) and the paged
    // online-softmax attention dispatches to `cmd`.
    void compute(luisa::compute::CommandList &cmd,
                 const luisa::compute::Buffer<float> &q,
                 const luisa::compute::Buffer<float> &k,
                 const luisa::compute::Buffer<float> &v,
                 const luisa::compute::Buffer<float> &o) const;

    // -- Re-allocation -------------------------------------------------------
    // Ensure room for `capacity` physical pages. If the current buffer is too
    // small, allocate a larger one, re-allocate the whole heap pool into it
    // (copying live pages), and retire the old buffer + heaps through a
    // stream callback (deferred destruction). Returns false on OOM.
    bool reserve(uint32_t capacity);
    // Unconditional re-allocation into a segment of exactly `capacity` pages.
    bool grow(uint32_t capacity);

    // -- Block-pool churn (eviction / reuse demo) -----------------------------
    // Evict one logical page: sparse-unmap its physical page and return it to
    // the free queue. Returns false when already evicted.
    bool evict_page(uint32_t logical);
    // Re-allocate a logical page from the free queue and sparse-map it.
    // The content is undefined afterwards (fresh block). Returns false when
    // the free queue is empty.
    bool allocate_page(uint32_t logical);

    // Re-upload the full logical -> offset index buffer (after churn).
    void upload_index();

    // -- Accessors -------------------------------------------------------------
    [[nodiscard]] bool valid() const noexcept { return _prepared; }
    [[nodiscard]] const Geometry &geometry() const noexcept { return _geom; }
    [[nodiscard]] uint32_t capacity() const noexcept;
    [[nodiscard]] uint32_t total_pages() const noexcept { return _geom.total_pages; }
    [[nodiscard]] const PageAllocator &allocator() const noexcept { return _alloc; }
    // Element offset per logical page (index-buffer contents).
    [[nodiscard]] luisa::vector<uint32_t> index_snapshot() const noexcept;

private:
    // One pool segment: the pair of sparse KV buffers plus the heap pool that
    // backs their tiles. Move-only (sparse resources are non-copyable).
    struct Segment;

    [[nodiscard]] luisa::unique_ptr<Segment> allocate_segment(uint32_t tiles) noexcept;
    bool map_segment(Segment &seg, uint32_t tiles);
    void unmap_segment(Segment &seg, uint32_t tiles);
    // Deferred destruction of a retired segment via CommandList::add_callback.
    void retire(luisa::unique_ptr<Segment> seg);
    bool check_geometry(const luisa::compute::SparseBuffer<float> &probe) noexcept;

    luisa::compute::Device &_device;
    luisa::compute::Stream &_stream;

    Geometry _geom{};
    PageAllocator _alloc;
    luisa::unique_ptr<Segment> _segment;
    luisa::compute::Buffer<uint32_t> _index_buffer;

    ReshapeKVToPagedShader _reshape_shader;
    PagedAttentionShader _paged_shader;

    bool _prepared{false};
};

}// namespace mla
