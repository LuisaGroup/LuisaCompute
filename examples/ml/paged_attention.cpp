// Multi-Head Latent Attention (MLA) Example -- reusable paged KV-cache driver.

#include "paged_attention.h"

#include <algorithm>
#include <numeric>
#include <random>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/sparse_buffer.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/sparse_command_list.h>

#include "attention_config.h"

using namespace luisa;
using namespace luisa::compute;

namespace mla {

// ---------------------------------------------------------------------------
// Allocation module: PageAllocator
// ---------------------------------------------------------------------------
// Custom allocation over the heap pool: residency granularity is the sparse
// tile (one SparseBufferHeap each), but one tile is generally *not* one
// attention page, so pages are carved out of tiles (pages_per_tile per tile).

PagedAttention::PageAllocator::PageAllocator(uint32_t capacity,
                                             uint32_t pages_per_tile,
                                             uint32_t page_elems,
                                             uint32_t tile_stride_elems) noexcept {
    reset(capacity, pages_per_tile, page_elems, tile_stride_elems);
}

void PagedAttention::PageAllocator::build_pages(uint32_t capacity,
                                                uint32_t pages_per_tile,
                                                uint32_t page_elems,
                                                uint32_t tile_stride_elems) noexcept {
    _pages.resize(capacity);
    _mapped.assign(capacity, false);
    for (uint32_t p = 0u; p < capacity; ++p) {
        const uint32_t tile = p / pages_per_tile;
        const uint32_t slot = p % pages_per_tile;
        _pages[p] = Page{
            .tile = tile,
            .page_in_tile = slot,
            .elem_offset = tile * tile_stride_elems + slot * page_elems};
    }
}

void PagedAttention::PageAllocator::reset(uint32_t capacity,
                                          uint32_t pages_per_tile,
                                          uint32_t page_elems,
                                          uint32_t tile_stride_elems) noexcept {
    build_pages(capacity, pages_per_tile, page_elems, tile_stride_elems);
    _table.clear();
    _free.resize(capacity);
    // Free queue: last page first so allocation walks ascending physical ids.
    std::iota(_free.rbegin(), _free.rend(), 0u);
    _free_count = capacity;
}

uint32_t PagedAttention::PageAllocator::allocate() noexcept {
    if (_free_count == 0u) { return null_page; }
    return _free[--_free_count];
}

void PagedAttention::PageAllocator::free(uint32_t page) noexcept {
    if (page == null_page) { return; }
    // Idempotent: never double-free a page already in the queue.
    for (uint32_t i = 0u; i < _free_count; ++i) {
        if (_free[i] == page) { return; }
    }
    _free[_free_count++] = page;
}

void PagedAttention::PageAllocator::assign(uint32_t logical, uint32_t page) noexcept {
    LUISA_ASSERT(logical < _table.size(), "logical page out of range");
    _table[logical] = page;
}

uint32_t PagedAttention::PageAllocator::offset_of(uint32_t logical) const noexcept {
    const uint32_t page = _table[logical];
    return page == null_page ? null_page : _pages[page].elem_offset;
}

void PagedAttention::PageAllocator::resize_table(uint32_t logical_pages) noexcept {
    _table.resize(logical_pages, null_page);
}

namespace {

// Bind `total_pages` logical pages to physical pages popped from the
// allocator free queue, then shuffle the pairing so residency is scattered
// (proving the indirection path). Returns the updated allocator.
[[nodiscard]] PagedAttention::PageAllocator
bind_scattered(PagedAttention::PageAllocator alloc, uint32_t total_pages,
               uint32_t seed) noexcept {
    luisa::vector<uint32_t> pages(total_pages, PagedAttention::PageAllocator::null_page);
    for (auto &p : pages) {
        p = alloc.allocate();
        LUISA_ASSERT(p != PagedAttention::PageAllocator::null_page,
                     "pool exhausted while binding logical pages");
    }
    std::shuffle(pages.begin(), pages.end(), std::mt19937{seed});
    alloc.resize_table(total_pages);
    for (uint32_t l = 0u; l < total_pages; ++l) { alloc.assign(l, pages[l]); }
    return alloc;
}

}// namespace

// ---------------------------------------------------------------------------
// Pool segment (sparse buffers + heap pool)
// ---------------------------------------------------------------------------

struct PagedAttention::Segment {
    SparseBuffer<float> k;
    SparseBuffer<float> v;
    luisa::vector<SparseBufferHeap> heaps_k;
    luisa::vector<SparseBufferHeap> heaps_v;
    uint32_t tiles{};
};

PagedAttention::PagedAttention(Device &device, Stream &stream) noexcept
    : _device{device}, _stream{stream} {}

PagedAttention::~PagedAttention() {
    // Vulkan requires no active mappings when a sparse resource or its heaps
    // are destroyed; tear the live segment down deterministically.
    if (_segment) {
        unmap_segment(*_segment, _segment->tiles);
        _stream << synchronize();
        _segment = nullptr;
    }
}

luisa::unique_ptr<PagedAttention::Segment>
PagedAttention::allocate_segment(uint32_t tiles) noexcept {
    auto seg = luisa::make_unique<Segment>();
    const auto tile_elems = _geom.tile_elems;
    const auto pool_elems = static_cast<size_t>(tiles) * tile_elems;
    seg->tiles = tiles;
    seg->k = _device.create_sparse_buffer<float>(pool_elems);
    seg->v = _device.create_sparse_buffer<float>(pool_elems);
    if (!seg->k || !seg->v) { return nullptr; }
    // Heap pool: one SparseBufferHeap per tile per buffer (the backend
    // sparse-residency registry forbids one heap backing multiple live
    // ranges, so a heap allocation IS a residency block here).
    seg->heaps_k.resize(tiles);
    seg->heaps_v.resize(tiles);
    for (uint32_t t = 0u; t < tiles; ++t) {
        seg->heaps_k[t] = _device.allocate_sparse_buffer_heap(_geom.tile_bytes);
        seg->heaps_v[t] = _device.allocate_sparse_buffer_heap(_geom.tile_bytes);
        if (!seg->heaps_k[t] || !seg->heaps_v[t]) { return nullptr; }
    }
    return seg;
}

bool PagedAttention::map_segment(Segment &seg, uint32_t tiles) {
    SparseCommandList map;
    for (uint32_t t = 0u; t < tiles; ++t) {
        map << seg.k.map_tile(t, 1u, seg.heaps_k[t])
            << seg.v.map_tile(t, 1u, seg.heaps_v[t]);
    }
    // The map commit must complete before any kernel touches the tiles.
    _stream << map.commit() << synchronize();
    return true;
}

void PagedAttention::unmap_segment(Segment &seg, uint32_t tiles) {
    SparseCommandList unmap;
    unmap << seg.k.unmap_tile(0u, tiles)
          << seg.v.unmap_tile(0u, tiles);
    _stream << unmap.commit() << synchronize();
}

void PagedAttention::retire(luisa::unique_ptr<Segment> seg) {
    // Deferred destruction: the segment (sparse buffers + heaps) is captured
    // by a stream callback so it is only released once the GPU has finished
    // every command still referencing the old pool.
    CommandList cb = CommandList::create();
    cb.add_callback([seg = std::move(seg)]() mutable noexcept {
        LUISA_INFO("  [paged] retired pool segment released ({} tiles)", seg->tiles);
        seg = nullptr;
    });
    _stream << cb.commit() << synchronize();
}

bool PagedAttention::check_geometry(const SparseBuffer<float> &probe) noexcept {
    _geom.tile_bytes = static_cast<uint32_t>(probe.tile_size_bytes());
    _geom.tile_elems = static_cast<uint32_t>(probe.tile_size());
    constexpr auto elems_per_token_page = num_heads * head_dim;// [h][d] slice per token
    const auto token_capacity = _geom.tile_elems / elems_per_token_page;
    if (token_capacity == 0u) {
        LUISA_WARNING("Sparse tile ({} B) cannot hold one token of all heads ({} floats).",
                      _geom.tile_bytes, elems_per_token_page);
        return false;
    }
    // tokens_per_page = vLLM block_size: the largest divisor of seq_len that
    // fits one tile, so every page is fully packed with whole tokens.
    _geom.tokens_per_page = paged_largest_divisor(seq_len, token_capacity);
    _geom.pages_per_seq = seq_len / _geom.tokens_per_page;
    _geom.page_elems = _geom.tokens_per_page * elems_per_token_page;
    // Custom allocation: one physical tile may hold several attention pages
    // when the page does not fill the tile (divisor rounding).
    _geom.pages_per_tile = _geom.tile_elems / _geom.page_elems;
    _geom.page_stride = _geom.page_elems;
    _geom.sub_tile = paged_sub_tile(_geom.tokens_per_page);
    if (_geom.pages_per_tile == 0u ||
        (_geom.sub_tile * head_dim) % paged_attention_block_size != 0u) {
        LUISA_WARNING("Paged geometry (tokens/page={}, pages/tile={}, sub-tile={}) is "
                      "incompatible with the {}-thread attention block.",
                      _geom.tokens_per_page, _geom.pages_per_tile,
                      _geom.sub_tile, paged_attention_block_size);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prepare phase
// ---------------------------------------------------------------------------

bool PagedAttention::prepare(uint32_t num_sequences) {
    LUISA_ASSERT(!_prepared, "prepare() called twice");

    // -- Geometry discovery --------------------------------------------------
    // The sparse tile size is chosen by the device and only known after
    // creating a sparse buffer; probe it before sizing the real pool.
    auto probe = _device.create_sparse_buffer<float>(1u);
    if (!probe) [[unlikely]] {
        LUISA_WARNING("Paged attention requires a sparse-buffer-capable backend "
                      "(vk/cuda/dx/hip).");
        return false;
    }
    if (!check_geometry(probe)) [[unlikely]] { return false; }
    probe = {};

    LUISA_INFO("Paged KV: tile={} B ({} floats), tokens/page={} (vLLM block_size), "
               "pages/tile={}, pages/seq={}, layout [page][h][t][d]",
               _geom.tile_bytes, _geom.tile_elems, _geom.tokens_per_page,
               _geom.pages_per_tile, _geom.pages_per_seq);
    if (_geom.pages_per_tile > 1u) {
        LUISA_INFO("  Custom allocation: {} attention pages carved per physical tile "
                   "({} unused floats per tile)",
                   _geom.pages_per_tile,
                   _geom.tile_elems - _geom.pages_per_tile * _geom.page_elems);
    } else if (const auto padding = _geom.tile_elems - _geom.page_elems) {
        LUISA_INFO("  Internal fragmentation: {} unused floats per tile", padding);
    }

    // -- Initial pool segment --------------------------------------------------
    // The logical page count follows the discovered geometry: every sequence
    // is cut into pages_per_seq logical pages.
    const uint32_t total_pages = num_sequences * _geom.pages_per_seq;
    const uint32_t tiles = (total_pages + _geom.pages_per_tile - 1u) / _geom.pages_per_tile;
    const uint32_t capacity = tiles * _geom.pages_per_tile;
    _segment = allocate_segment(tiles);
    if (!_segment) [[unlikely]] {
        LUISA_WARNING("Failed to allocate sparse paged-KV resources.");
        return false;
    }
    map_segment(*_segment, tiles);

    // -- Indexing: allocator + device-visible page index -----------------------
    _alloc = PageAllocator{capacity, _geom.pages_per_tile,
                           _geom.page_elems, _geom.tile_elems};
    _geom.total_pages = total_pages;

    // Bind logical pages to shuffled physical pages so residency is maximally
    // scattered, proving the indirection path correct. Pages are popped
    // through the allocator free queue (the vLLM BlockPool allocation path);
    // bind_scattered also sizes the logical page table.
    _alloc = bind_scattered(std::move(_alloc), total_pages, 42u);
    // map_segment() made every tile resident, so every physical page of the
    // segment is residency-mapped (eviction operates at tile granularity).
    for (uint32_t p = 0u; p < capacity; ++p) { _alloc.set_mapped(p, true); }
    LUISA_INFO("  Logical page 0 -> physical page {} (elem offset {})",
               _alloc.page_of(0u), _alloc.offset_of(0u));

    _index_buffer = _device.create_buffer<uint32_t>(total_pages);
    if (!_index_buffer) [[unlikely]] { return false; }
    upload_index();

    // -- Compile ---------------------------------------------------------------
    LUISA_INFO("Compiling paged attention kernels ...");
    Clock compile_clock;
    ShaderOption opt{.enable_debug_info = false};
    opt.name = "paged_reshape_kv";
    _reshape_shader = _device.compile(create_reshape_kv_to_paged_kernel(), opt);
    opt.name = "paged_attention";
    _paged_shader = _device.compile(
        create_paged_attention_kernel(_geom.tokens_per_page), opt);
    LUISA_INFO("  Paged attention kernels compiled in {:.2f} ms", compile_clock.toc());

    _prepared = true;
    return true;
}

void PagedAttention::upload_index() {
    auto offsets = index_snapshot();
    _stream << _index_buffer.copy_from(luisa::span{offsets}) << synchronize();
}

luisa::vector<uint32_t> PagedAttention::index_snapshot() const noexcept {
    luisa::vector<uint32_t> offsets(_geom.total_pages);
    for (uint32_t l = 0u; l < _geom.total_pages; ++l) {
        offsets[l] = _alloc.offset_of(l);
    }
    return offsets;
}

uint32_t PagedAttention::capacity() const noexcept {
    return _alloc.capacity();
}

// ---------------------------------------------------------------------------
// Compute phase
// ---------------------------------------------------------------------------

void PagedAttention::compute(CommandList &cmd,
                             const Buffer<float> &q,
                             const Buffer<float> &k,
                             const Buffer<float> &v,
                             const Buffer<float> &o) const {
    LUISA_ASSERT(_prepared, "compute() before a successful prepare()");
    cmd << _reshape_shader(k, v,
                           _segment->k.view(), _segment->v.view(),
                           _index_buffer,
                           _geom.tokens_per_page, _geom.pages_per_seq)
               .dispatch(qkv_size)
        << _paged_shader(q,
                         _segment->k.view(), _segment->v.view(),
                         o, _index_buffer, _geom.pages_per_seq)
               .dispatch(batch * num_heads * seq_len);
}

// ---------------------------------------------------------------------------
// Re-allocation
// ---------------------------------------------------------------------------

bool PagedAttention::reserve(uint32_t capacity) {
    if (capacity <= this->capacity()) { return true; }
    return grow(capacity);
}

bool PagedAttention::grow(uint32_t capacity) {
    LUISA_ASSERT(_prepared, "grow() before a successful prepare()");
    const uint32_t tiles = (capacity + _geom.pages_per_tile - 1u) / _geom.pages_per_tile;
    if (tiles <= _segment->tiles) { return true; }
    const uint32_t new_capacity = tiles * _geom.pages_per_tile;
    LUISA_INFO("  [paged] re-allocating pool: {} -> {} pages ({} -> {} tiles)",
               _alloc.capacity(), new_capacity, _segment->tiles, tiles);

    auto new_segment = allocate_segment(tiles);
    if (!new_segment) [[unlikely]] {
        LUISA_WARNING("Failed to allocate a larger sparse paged-KV segment.");
        return false;
    }
    map_segment(*new_segment, tiles);

    // Fresh allocator over the new segment; replay the live bindings through
    // the free queue in a scattered order so the new pool is not trivially
    // sequential.
    PageAllocator new_alloc{new_capacity, _geom.pages_per_tile,
                            _geom.page_elems, _geom.tile_elems};
    new_alloc.resize_table(_geom.total_pages);
    // Collect the live logical pages, pop one new physical page per live
    // binding, then scatter the pairing with a permutation.
    luisa::vector<uint32_t> live;
    live.reserve(_geom.total_pages);
    for (uint32_t l = 0u; l < _geom.total_pages; ++l) {
        const uint32_t old_page = _alloc.page_of(l);
        if (old_page != PageAllocator::null_page && _alloc.is_mapped(old_page)) {
            live.emplace_back(l);
        } else {
            new_alloc.assign(l, PageAllocator::null_page);
        }
    }
    luisa::vector<uint32_t> new_pages(live.size(), PageAllocator::null_page);
    for (auto &np : new_pages) {
        np = new_alloc.allocate();
        if (np == PageAllocator::null_page) [[unlikely]] { return false; }
    }
    std::shuffle(new_pages.begin(), new_pages.end(), std::mt19937{7});

    // Copy every live page into the new pool and record the new bindings.
    CommandList copies = CommandList::create();
    uint32_t copied = 0u;
    for (size_t n = 0u; n < live.size(); ++n) {
        const uint32_t l = live[n];
        const uint32_t new_page = new_pages[n];
        const auto &src = _alloc.physical(_alloc.page_of(l));
        const auto &dst = new_alloc.physical(new_page);
        copies << new_segment->k.view(dst.elem_offset, _geom.page_elems)
                      .copy_from(_segment->k.view(src.elem_offset, _geom.page_elems))
               << new_segment->v.view(dst.elem_offset, _geom.page_elems)
                      .copy_from(_segment->v.view(src.elem_offset, _geom.page_elems));
        new_alloc.assign(l, new_page);
        ++copied;
    }
    _stream << copies.commit() << synchronize();
    // The new segment is fully tile-mapped, so every page is resident.
    for (uint32_t p = 0u; p < new_capacity; ++p) { new_alloc.set_mapped(p, true); }

    // Retire the old segment: sparse-unmap first (Vulkan requires no active
    // mappings at destruction), then hand the resources to a stream callback
    // so they are released only after the GPU drained every reference.
    unmap_segment(*_segment, _segment->tiles);
    retire(std::move(_segment));

    _segment = std::move(new_segment);
    _alloc = std::move(new_alloc);
    // Offsets changed (new pool), so refresh the device-visible page index.
    upload_index();
    LUISA_INFO("  [paged] re-allocation complete: {} live pages copied", copied);
    return true;
}

// ---------------------------------------------------------------------------
// Block-pool churn (eviction / reuse demo)
// ---------------------------------------------------------------------------

bool PagedAttention::evict_page(uint32_t logical) {
    const uint32_t page = _alloc.page_of(logical);
    if (page == PageAllocator::null_page || !_alloc.is_mapped(page)) { return false; }
    const auto &phys = _alloc.physical(page);
    // Residency granularity is the tile: only evict when the page owns its
    // tile (no other live page shares it).
    if (_geom.pages_per_tile > 1u) {
        for (uint32_t l = 0u; l < _geom.total_pages; ++l) {
            const uint32_t other = _alloc.page_of(l);
            if (other != PageAllocator::null_page && other != page &&
                _alloc.is_mapped(other) && _alloc.physical(other).tile == phys.tile) {
                return false;
            }
        }
    }
    {
        SparseCommandList evict;
        evict << _segment->k.unmap_tile(phys.tile, 1u)
              << _segment->v.unmap_tile(phys.tile, 1u);
        _stream << evict.commit() << synchronize();
    }
    _alloc.set_mapped(page, false);
    _alloc.assign(logical, PageAllocator::null_page);
    _alloc.free(page);// back to the free queue (ref_cnt -> 0)
    upload_index();
    return true;
}

bool PagedAttention::allocate_page(uint32_t logical) {
    const uint32_t page = _alloc.allocate();
    if (page == PageAllocator::null_page) { return false; }
    const auto &phys = _alloc.physical(page);
    if (!_alloc.is_mapped(page)) {
        SparseCommandList map;
        map << _segment->k.map_tile(phys.tile, 1u, _segment->heaps_k[phys.tile])
            << _segment->v.map_tile(phys.tile, 1u, _segment->heaps_v[phys.tile]);
        _stream << map.commit() << synchronize();
        _alloc.set_mapped(page, true);
    }
    _alloc.assign(logical, page);
    upload_index();
    return true;
}

}// namespace mla
