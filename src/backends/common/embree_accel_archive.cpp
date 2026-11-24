#include "embree_accel_archive.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#define LUISA_EMBREE_OOT_X64 1
#else
#define LUISA_EMBREE_OOT_X64 0
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
#define LUISA_EMBREE_OOT_ARM64 1
#else
#define LUISA_EMBREE_OOT_ARM64 0
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#if LUISA_EMBREE_OOT_X64
#include <intrin.h>
#endif
#include <malloc.h>
#include <windows.h>
#else
#if LUISA_EMBREE_OOT_X64
#include <cpuid.h>
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#endif
#endif

#if (LUISA_EMBREE_OOT_X64 || LUISA_EMBREE_OOT_ARM64) && !defined(LUISA_EMBREE_DISABLE_OOT_ARCHIVE) && \
    !(defined(LUISA_COMPUTE_FALLBACK_EMBREE_VERSION) && LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3) && \
    !(defined(LUISA_COMPUTE_SIMD_EMBREE_VERSION) && LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3) && \
    __has_include(<embree4/rtcore.h>)
#define LUISA_EMBREE_OOT_ENABLED 1
#include <embree4/rtcore.h>
#else
#define LUISA_EMBREE_OOT_ENABLED 0
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/vstl/functional.h>

namespace luisa::compute {

EmbreeImportedBVHHandle::EmbreeImportedBVHHandle(EmbreeImportedBVHHandle &&other) noexcept
    : slab{other.slab},
      slab_bytes{other.slab_bytes},
      bvh_ptr{other.bvh_ptr},
      bvh_root_ofs{other.bvh_root_ofs},
      saved_dummy_root{other.saved_dummy_root},
      bvh_bounds_ofs{other.bvh_bounds_ofs},
      has_saved_bvh_bounds{other.has_saved_bvh_bounds},
      rtc_scene{other.rtc_scene},
      scene_bounds_ofs{other.scene_bounds_ofs},
      has_saved_scene_bounds{other.has_saved_scene_bounds},
      vertices_slot{other.vertices_slot},
      saved_dummy_verts{other.saved_dummy_verts},
      has_saved_verts{other.has_saved_verts} {
    std::memcpy(saved_bvh_bounds, other.saved_bvh_bounds, sizeof(saved_bvh_bounds));
    std::memcpy(saved_scene_bounds, other.saved_scene_bounds, sizeof(saved_scene_bounds));
    other.slab = nullptr;
    other.slab_bytes = 0u;
    other.bvh_ptr = nullptr;
    other.bvh_root_ofs = 0u;
    other.saved_dummy_root = 0u;
    other.bvh_bounds_ofs = 0u;
    other.has_saved_bvh_bounds = false;
    other.rtc_scene = nullptr;
    other.scene_bounds_ofs = 0u;
    other.has_saved_scene_bounds = false;
    other.vertices_slot = nullptr;
    other.saved_dummy_verts = nullptr;
    other.has_saved_verts = false;
}

EmbreeImportedBVHHandle &EmbreeImportedBVHHandle::operator=(EmbreeImportedBVHHandle &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    restore_and_release();
    slab = other.slab;
    slab_bytes = other.slab_bytes;
    bvh_ptr = other.bvh_ptr;
    bvh_root_ofs = other.bvh_root_ofs;
    saved_dummy_root = other.saved_dummy_root;
    bvh_bounds_ofs = other.bvh_bounds_ofs;
    std::memcpy(saved_bvh_bounds, other.saved_bvh_bounds, sizeof(saved_bvh_bounds));
    has_saved_bvh_bounds = other.has_saved_bvh_bounds;
    rtc_scene = other.rtc_scene;
    scene_bounds_ofs = other.scene_bounds_ofs;
    std::memcpy(saved_scene_bounds, other.saved_scene_bounds, sizeof(saved_scene_bounds));
    has_saved_scene_bounds = other.has_saved_scene_bounds;
    vertices_slot = other.vertices_slot;
    saved_dummy_verts = other.saved_dummy_verts;
    has_saved_verts = other.has_saved_verts;

    other.slab = nullptr;
    other.slab_bytes = 0u;
    other.bvh_ptr = nullptr;
    other.bvh_root_ofs = 0u;
    other.saved_dummy_root = 0u;
    other.bvh_bounds_ofs = 0u;
    other.has_saved_bvh_bounds = false;
    other.rtc_scene = nullptr;
    other.scene_bounds_ofs = 0u;
    other.has_saved_scene_bounds = false;
    other.vertices_slot = nullptr;
    other.saved_dummy_verts = nullptr;
    other.has_saved_verts = false;
    return *this;
}

void EmbreeImportedBVHHandle::restore_and_release() noexcept {
    // Restore Embree's original dummy BVH pointers BEFORE releasing the slab or calling rtcReleaseScene.
    if (has_saved_verts && vertices_slot != nullptr) {
        *vertices_slot = saved_dummy_verts;
    }
    if (bvh_ptr != nullptr && bvh_root_ofs != 0u) {
        std::memcpy(static_cast<std::byte *>(bvh_ptr) + bvh_root_ofs, &saved_dummy_root, sizeof(uint64_t));
    }
    if (has_saved_bvh_bounds && bvh_ptr != nullptr) {
        std::memcpy(static_cast<std::byte *>(bvh_ptr) + bvh_bounds_ofs, saved_bvh_bounds, 64u);
    }
    if (has_saved_scene_bounds && rtc_scene != nullptr) {
        std::memcpy(static_cast<std::byte *>(rtc_scene) + scene_bounds_ofs, saved_scene_bounds, 64u);
    }
    if (slab != nullptr) {
#if defined(_WIN32)
        _aligned_free(slab);
#else
        std::free(slab);
#endif
        slab = nullptr;
    }
    slab_bytes = 0u;
    bvh_ptr = nullptr;
    bvh_root_ofs = 0u;
    saved_dummy_root = 0u;
    bvh_bounds_ofs = 0u;
    has_saved_bvh_bounds = false;
    rtc_scene = nullptr;
    scene_bounds_ofs = 0u;
    has_saved_scene_bounds = false;
    vertices_slot = nullptr;
    saved_dummy_verts = nullptr;
    has_saved_verts = false;
}

namespace {

constexpr uint32_t kEmbreeCompatMagic = 0x454D4252u;// "EMBR"
constexpr uint32_t kEmbreeCompatVersion = 1u;
constexpr uint32_t kEmbreePayloadMagic = 0x45425648u;// "EBVH"
constexpr uint32_t kEmbreePayloadFormatVersion = 1u;
constexpr uint64_t kMaxEmbreePayloadBytes = 4ull << 30u;

enum : uint32_t {
    kPrimKindUnknown = 0u,
    kPrimKindTriangle4 = 1u,
    kPrimKindTriangle4v = 2u,
    kPrimKindTriangle4i = 3u,
};

enum : uint32_t {
    kIvariantFast = 0u,
    kIvariantRobust = 1u,
};

enum : uint32_t {
    kAccelTypeUnknown = 0u,
    kAccelTypeAccelN = 1u,
    kAccelTypeAccelInstance = 2u,
    kAccelTypeBVH4 = 3u,
    kAccelTypeBVH8 = 4u,
};

enum : uint32_t {
    kIsaSSE2 = 1u << 0u,
    kIsaSSE42 = 1u << 1u,
    kIsaAVX = 1u << 2u,
    kIsaAVX2 = 1u << 3u,
    kIsaAVX512F = 1u << 4u,
    kIsaNEON = 1u << 5u,
};

// Embree 4 NodeRefPtr<N> tag bits (kernels/bvh/bvh_node_ref.h)
constexpr uint64_t kAlignMask = 0xFull;
constexpr uint64_t kBarrierMask = (1ull << 63u);
constexpr uint64_t kTyAABBNode = 0ull;
constexpr uint64_t kTyLeaf = 8ull;
constexpr uint64_t kEmptyNode = kTyLeaf;
constexpr uint64_t kInvalidNode = (~kAlignMask) | (kTyLeaf + 0ull);
constexpr uint64_t kPopRay = (~kAlignMask) | (kTyLeaf + 1ull);
constexpr uint64_t kMaxLeafBlocks = 7ull;

// Archive payload NodeRef encoding
constexpr uint64_t kRelocLeafFlag = (1ull << 63u);
constexpr uint64_t kRelocBlockCountShift = 56u;
constexpr uint64_t kRelocBlockCountMask = 0x7full;
constexpr uint64_t kRelocIndexMask = ((1ull << 56u) - 1ull);
constexpr uint64_t kRelocEmpty = std::numeric_limits<uint64_t>::max();

struct EmbreeStaticTriangleBVHPayloadHeader {
    uint32_t magic;
    uint32_t format_version;
    uint32_t embree_version_number;
    uint32_t build_config_flags;
    uint32_t required_isa_mask;
    uint32_t branching_factor;
    uint32_t prim_kind;
    uint32_t intersect_variant;
    uint32_t scene_flags;
    uint32_t quality_flags;
    uint32_t node_struct_bytes;
    uint32_t prim_struct_bytes;
    uint64_t vertex_stride_bytes;
    uint64_t vertex_count;
    uint64_t triangle_count;
    uint64_t num_primitives_in_bvh;
    uint64_t node_count;
    uint64_t leaf_block_count;
    uint64_t root_encoded_ref;
    float bounds_lower[4];
    float bounds_upper[4];
};
static_assert(sizeof(EmbreeStaticTriangleBVHPayloadHeader) == 136u);

struct AABBNode4POD {
    uint64_t children[4];
    float lower_x[4];
    float upper_x[4];
    float lower_y[4];
    float upper_y[4];
    float lower_z[4];
    float upper_z[4];
};
static_assert(sizeof(AABBNode4POD) == 128u);

struct AABBNode8POD {
    uint64_t children[8];
    float lower_x[8];
    float upper_x[8];
    float lower_y[8];
    float upper_y[8];
    float lower_z[8];
    float upper_z[8];
};
static_assert(sizeof(AABBNode8POD) == 256u);

struct Triangle4POD {
    float v0[3][4];
    float e1[3][4];
    float e2[3][4];
    uint32_t geomIDs[4];
    uint32_t primIDs[4];
};
static_assert(sizeof(Triangle4POD) == 176u);

struct Triangle4vPOD {
    float v0[3][4];
    float v1[3][4];
    float v2[3][4];
    uint32_t geomIDs[4];
    uint32_t primIDs[4];
};
static_assert(sizeof(Triangle4vPOD) == 176u);

struct Triangle4iPOD {
    uint32_t v0_[4];
    uint32_t v1_[4];
    uint32_t v2_[4];
    uint32_t geomIDs[4];
    uint32_t primIDs[4];
};
static_assert(sizeof(Triangle4iPOD) == 80u);

struct LBBox3faPOD {
    float lo0[4];
    float hi0[4];
    float lo1[4];
    float hi1[4];
};
static_assert(sizeof(LBBox3faPOD) == 64u);

[[nodiscard]] bool options_match(const AccelOption &a, const AccelOption &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           !static_cast<bool>(a.motion) &&
           !static_cast<bool>(b.motion);
}

[[nodiscard]] bool checked_mul_add(
    uint64_t a,
    uint64_t b,
    uint64_t addend,
    uint64_t &out) noexcept {
    if (a != 0ull && b > (std::numeric_limits<uint64_t>::max() - addend) / a) {
        return false;
    }
    out = addend + a * b;
    return true;
}

template<typename Fn>
void parallel_for_range(size_t total, size_t parallel_threshold, Fn &&fn) noexcept {
    if (total == 0u) {
        return;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t num_workers = (total >= parallel_threshold && hw > 1u) ?
                                   std::min<size_t>(8u, hw) :
                                   1u;
    if (num_workers <= 1u) {
        fn(0u, total);
        return;
    }
#if defined(_WIN32)
    struct TaskData {
        Fn *fn;
        size_t begin;
        size_t end;
    };
    auto callback = [](PTP_CALLBACK_INSTANCE, PVOID ctx, PTP_WORK) noexcept {
        const auto *d = static_cast<const TaskData *>(ctx);
        (*d->fn)(d->begin, d->end);
    };
    std::array<TaskData, 8> tasks{};
    std::array<PTP_WORK, 7> works{};
    size_t submitted = 0u;
    const size_t chunk = (total + num_workers - 1u) / num_workers;
    for (size_t w = 1u; w < num_workers; ++w) {
        const size_t b = w * chunk;
        if (b >= total) {
            break;
        }
        const size_t e = std::min(total, b + chunk);
        tasks[w] = TaskData{&fn, b, e};
        PTP_WORK wk = CreateThreadpoolWork(callback, &tasks[w], nullptr);
        if (wk != nullptr) {
            works[submitted++] = wk;
            SubmitThreadpoolWork(wk);
        } else {
            fn(b, e);
        }
    }
    fn(0u, std::min(total, chunk));
    for (size_t i = 0u; i < submitted; ++i) {
        WaitForThreadpoolWorkCallbacks(works[i], FALSE);
        CloseThreadpoolWork(works[i]);
    }
#else
    std::vector<std::thread> threads;
    threads.reserve(num_workers - 1u);
    const size_t chunk = (total + num_workers - 1u) / num_workers;
    for (size_t w = 1u; w < num_workers; ++w) {
        const size_t b = w * chunk;
        if (b >= total) {
            break;
        }
        const size_t e = std::min(total, b + chunk);
        threads.emplace_back([&fn, b, e]() noexcept { fn(b, e); });
    }
    fn(0u, std::min(total, chunk));
    for (auto &t : threads) {
        t.join();
    }
#endif
}

[[nodiscard]] bool is_readable_memory(const void *ptr, size_t bytes) noexcept {
    if (ptr == nullptr || bytes == 0u) {
        return false;
    }
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr < 0x10000ull || addr > (std::numeric_limits<uintptr_t>::max() - bytes)) {
        return false;
    }
#if defined(_WIN32)
    uintptr_t cur = addr;
    const uintptr_t end = addr + bytes;
    while (cur < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(reinterpret_cast<const void *>(cur), &mbi, sizeof(mbi)) == 0u) {
            return false;
        }
        if (mbi.State != MEM_COMMIT) {
            return false;
        }
        const DWORD prot = mbi.Protect;
        if ((prot & (PAGE_GUARD | PAGE_NOACCESS)) != 0u) {
            return false;
        }
        constexpr DWORD kReadableMask =
            PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
            PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if ((prot & kReadableMask) == 0u) {
            return false;
        }
        const auto reg_end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        if (reg_end <= cur) {
            return false;
        }
        cur = reg_end;
    }
    return true;
#elif defined(__linux__)
    // The ABI probe follows candidate pointers from opaque Embree objects.
    // Check the mapped region before dereferencing any candidate.
    auto *maps = std::fopen("/proc/self/maps", "r");
    if (maps == nullptr) {
        return false;
    }
    auto close_maps = vstd::scope_exit([maps] { std::fclose(maps); });
    const auto end = addr + bytes;
    char line[512];
    while (std::fgets(line, sizeof(line), maps) != nullptr) {
        unsigned long long begin = 0u, limit = 0u;
        char access = '\0';
        if (std::sscanf(line, "%llx-%llx %c", &begin, &limit, &access) == 3 &&
            addr >= begin && addr < limit) {
            return access == 'r' && end <= limit;
        }
    }
    return false;
#elif defined(__APPLE__) && TARGET_OS_OSX
    mach_vm_address_t begin = addr;
    mach_vm_size_t size = 0u;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto task = mach_task_self();
    const auto status = mach_vm_region(task, &begin, &size, VM_REGION_BASIC_INFO_64,
                                       reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) {
        mach_port_deallocate(task, object);
    }
    return status == KERN_SUCCESS && begin <= addr &&
           (info.protection & VM_PROT_READ) != 0 && addr + bytes - begin <= size;
#else
    return false;
#endif
}

[[nodiscard]] bool try_read_cstring(const char *ptr, size_t max_len, std::string_view &out) noexcept {
    const auto whole_range_readable = is_readable_memory(ptr, max_len);
    if (!whole_range_readable && !is_readable_memory(ptr, 1u)) {
        return false;
    }
    for (size_t i = 0u; i < max_len; ++i) {
        if (!whole_range_readable && !is_readable_memory(ptr + i, 1u)) {
            return false;
        }
        if (ptr[i] == '\0') {
            out = std::string_view{ptr, i};
            return i > 0u;
        }
    }
    return false;
}

[[nodiscard]] uint32_t detect_cpu_isa_mask() noexcept {
    uint32_t mask = 0u;
#if LUISA_EMBREE_OOT_X64
#if defined(_WIN32)
    int info0[4]{};
    __cpuid(info0, 0);
    const int max_leaf = info0[0];
    if (max_leaf >= 1) {
        int info1[4]{};
        __cpuid(info1, 1);
        if ((info1[3] & (1 << 26)) != 0) {
            mask |= kIsaSSE2;
        }
        if ((info1[2] & (1 << 20)) != 0) {
            mask |= kIsaSSE42;
        }
        const bool osxsave = (info1[2] & (1 << 27)) != 0;
        const bool cpu_avx = (info1[2] & (1 << 28)) != 0;
        if (osxsave && cpu_avx) {
            const uint64_t xcr0 = _xgetbv(0);
            const bool os_avx = (xcr0 & 0x6ull) == 0x6ull;
            if (os_avx) {
                mask |= kIsaAVX;
                if (max_leaf >= 7) {
                    int info7[4]{};
                    __cpuidex(info7, 7, 0);
                    if ((info7[1] & (1 << 5)) != 0) {
                        mask |= kIsaAVX2;
                    }
                    const bool os_avx512 = (xcr0 & 0xE6ull) == 0xE6ull;
                    if (os_avx512 && (info7[1] & (1 << 16)) != 0) {
                        mask |= kIsaAVX512F;
                    }
                }
            }
        }
    }
#else
    unsigned eax = 0u, ebx = 0u, ecx = 0u, edx = 0u;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) && eax >= 1u) {
        const unsigned max_leaf = eax;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            if (edx & (1u << 26u)) mask |= kIsaSSE2;
            if (ecx & (1u << 20u)) mask |= kIsaSSE42;
            if ((ecx & (1u << 27u)) && (ecx & (1u << 28u))) {
                uint32_t xlo = 0u, xhi = 0u;
                asm volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
                if ((xlo & 0x6u) == 0x6u) {
                    mask |= kIsaAVX;
                    if (max_leaf >= 7u && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                        if (ebx & (1u << 5u)) mask |= kIsaAVX2;
                        if (((xlo & 0xE6u) == 0xE6u) && (ebx & (1u << 16u))) mask |= kIsaAVX512F;
                    }
                }
            }
        }
    }
#endif
#elif LUISA_EMBREE_OOT_ARM64
    // Advanced SIMD is baseline on the AArch64 targets supported by Embree.
    mask = kIsaNEON;
#endif
    return mask;
}

[[nodiscard]] size_t prim_struct_bytes_for_kind(uint32_t prim_kind) noexcept {
    switch (prim_kind) {
        case kPrimKindTriangle4: return sizeof(Triangle4POD);
        case kPrimKindTriangle4v: return sizeof(Triangle4vPOD);
        case kPrimKindTriangle4i: return sizeof(Triangle4iPOD);
        default: return 0u;
    }
}

[[nodiscard]] bool is_supported_triangle_prim_and_ivariant(
    uint32_t prim_kind,
    uint32_t ivariant) noexcept {
    switch (prim_kind) {
        case kPrimKindTriangle4: return ivariant == kIvariantFast;
        case kPrimKindTriangle4v: return ivariant == kIvariantRobust;
        case kPrimKindTriangle4i: return ivariant == kIvariantFast || ivariant == kIvariantRobust;
        default: return false;
    }
}

[[nodiscard]] bool parse_intersector1_name(
    std::string_view name,
    uint32_t &out_bf,
    uint32_t &out_prim_kind,
    uint32_t &out_ivariant,
    uint32_t &out_required_isa) noexcept {
    if (name.find("Triangle4vMB") != std::string_view::npos ||
        name.find("Triangle4iMB") != std::string_view::npos ||
        name.find("Triangle4MB") != std::string_view::npos ||
        name.find("Quantized") != std::string_view::npos ||
        name.find("Woop") != std::string_view::npos) {
        return false;
    }
    if (name.find("BVH8") != std::string_view::npos) {
        out_bf = 8u;
    } else if (name.find("BVH4") != std::string_view::npos) {
        out_bf = 4u;
    } else {
        return false;
    }

    if (name.find("Triangle4i") != std::string_view::npos) {
        out_prim_kind = kPrimKindTriangle4i;
    } else if (name.find("Triangle4v") != std::string_view::npos) {
        out_prim_kind = kPrimKindTriangle4v;
    } else if (name.find("Triangle4") != std::string_view::npos) {
        out_prim_kind = kPrimKindTriangle4;
    } else {
        return false;
    }

    if (name.find("Pluecker") != std::string_view::npos) {
        out_ivariant = kIvariantRobust;
    } else if (name.find("Moeller") != std::string_view::npos) {
        out_ivariant = kIvariantFast;
    } else {
        return false;
    }

    if (!is_supported_triangle_prim_and_ivariant(out_prim_kind, out_ivariant)) {
        return false;
    }

#if LUISA_EMBREE_OOT_ARM64
    // Embree reuses the x86 dispatch names for its NEON and 2xNEON builds.
    // The 2x variant uses two NEON vectors, not a distinct CPU extension.
    if (!(name.starts_with("sse2::BVH4") && out_bf == 4u) && !name.starts_with("avx2::BVH")) {
        return false;
    }
    out_required_isa = kIsaNEON;
#else
    out_required_isa = kIsaSSE2;
    if (out_bf == 8u || name.find("AVX::") != std::string_view::npos) {
        out_required_isa |= kIsaAVX;
    }
    if (name.find("AVX2") != std::string_view::npos) {
        out_required_isa |= kIsaAVX | kIsaAVX2;
    }
    if (name.find("AVX512") != std::string_view::npos) {
        out_required_isa |= kIsaAVX | kIsaAVX2 | kIsaAVX512F;
    }
#endif
    return true;
}

[[nodiscard]] bool bounds_close(const float *f, const float lo[3], const float hi[3], float tol = 1e-4f) noexcept {
    for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(f[k]) || !std::isfinite(f[4 + k]) ||
            std::fabs(f[k] - lo[k]) > tol ||
            std::fabs(f[4 + k] - hi[k]) > tol) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool lbbox_matches(const std::byte *addr, const float lo[3], const float hi[3], float tol = 1e-4f) noexcept {
    LBBox3faPOD lb{};
    std::memcpy(&lb, addr, sizeof(lb));
    return bounds_close(lb.lo0, lo, hi, tol) && bounds_close(lb.lo1, lo, hi, tol);
}

#if LUISA_EMBREE_OOT_ENABLED

[[nodiscard]] uint32_t expected_scene_flags_for_option(const AccelOption &opt) noexcept {
    uint32_t flags = static_cast<uint32_t>(RTC_SCENE_FLAG_ROBUST) |
                     static_cast<uint32_t>(RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
    if (opt.allow_compaction) {
        flags |= static_cast<uint32_t>(RTC_SCENE_FLAG_COMPACT);
    }
    if (opt.allow_update) {
        flags |= static_cast<uint32_t>(RTC_SCENE_FLAG_DYNAMIC);
    }
    return flags;
}

[[nodiscard]] uint32_t expected_quality_flags_for_option(const AccelOption &opt) noexcept {
    if (opt.allow_update) {
        return static_cast<uint32_t>(RTC_BUILD_QUALITY_LOW);
    }
    return opt.hint == AccelOption::UsageHint::FAST_TRACE ?
               static_cast<uint32_t>(RTC_BUILD_QUALITY_HIGH) :
               static_cast<uint32_t>(RTC_BUILD_QUALITY_MEDIUM);
}

[[nodiscard]] uint32_t query_embree_build_config_flags(RTCDevice dev) noexcept {
    uint32_t flags = 0u;
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED) != 0) flags |= (1u << 0u);
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_FILTER_FUNCTION_SUPPORTED) != 0) flags |= (1u << 1u);
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED) != 0 ||
        rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED) != 0 ||
        rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED) != 0) {
        flags |= (1u << 2u);
    }
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED) != 0) flags |= (1u << 3u);
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED) != 0) flags |= (1u << 4u);
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED) != 0) flags |= (1u << 5u);
    // Raw leaf blocks belong to the architecture that produced them.
    if (LUISA_EMBREE_OOT_ARM64) flags |= (1u << 6u);
    if (sizeof(void *) == 8u) flags |= (1u << 7u);
#ifdef RTC_MAX_INSTANCE_LEVEL_COUNT
    flags |= (static_cast<uint32_t>(RTC_MAX_INSTANCE_LEVEL_COUNT) & 0xffu) << 8u;
#else
    flags |= (1u << 8u);
#endif
    return flags;
}

struct InspectSceneResult {
    void *bvh_ptr{nullptr};
    uint32_t branching_factor{0u};
    uint32_t prim_kind{0u};
    uint32_t intersect_variant{0u};
    uint32_t required_isa_mask{0u};
    uint64_t bvh_root{kEmptyNode};
    LBBox3faPOD bvh_bounds{};
    float **vertices_slot{nullptr};
};

[[nodiscard]] bool inspect_committed_scene_bvh(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    InspectSceneResult &out) noexcept {
    out = {};
    const size_t min_scene_bytes = std::max<size_t>(
        abi.accel_bounds_ofs + sizeof(LBBox3faPOD),
        abi.scene_intersectors_ptr_ofs + abi.intersector1_name_ofs + sizeof(const char *));
    if (!abi.valid || rtc_scene == nullptr || !is_readable_memory(rtc_scene, min_scene_bytes)) {
        return false;
    }
    const auto *sbytes = static_cast<const std::byte *>(rtc_scene);
    void *bvh_ptr = nullptr;
    std::memcpy(&bvh_ptr, sbytes + abi.scene_intersectors_ptr_ofs, sizeof(void *));
    const size_t min_bvh_bytes = std::max<size_t>(
        abi.bvh_root_ofs + sizeof(uint64_t),
        std::max<size_t>(
            abi.accel_bounds_ofs + sizeof(LBBox3faPOD),
            std::max<size_t>(abi.accel_type_ofs + sizeof(uint32_t), abi.bvh_device_ofs + 2u * sizeof(void *))));
    if (bvh_ptr == nullptr || !is_readable_memory(bvh_ptr, min_bvh_bytes)) {
        return false;
    }

    const char *name_ptr = nullptr;
    std::memcpy(&name_ptr, sbytes + abi.scene_intersectors_ptr_ofs + abi.intersector1_name_ofs, sizeof(const char *));
    std::string_view name_sv;
    if (!try_read_cstring(name_ptr, 160u, name_sv)) {
        return false;
    }
    uint32_t bf_name = 0u, pk = 0u, iv = 0u, req_isa = 0u;
    if (!parse_intersector1_name(name_sv, bf_name, pk, iv, req_isa)) {
        return false;
    }

    const auto *bbytes = static_cast<const std::byte *>(bvh_ptr);
    uint32_t bvh_type = 0u;
    std::memcpy(&bvh_type, bbytes + abi.accel_type_ofs, sizeof(uint32_t));
    const uint32_t bf_type = (bvh_type == kAccelTypeBVH4) ? 4u : (bvh_type == kAccelTypeBVH8 ? 8u : 0u);
    if (bf_type == 0u || bf_type != bf_name) {
        return false;
    }

    void *bvh_scene_ptr = nullptr;
    std::memcpy(&bvh_scene_ptr, bbytes + abi.bvh_device_ofs + sizeof(void *), sizeof(void *));
    if (bvh_scene_ptr != rtc_scene) {
        return false;
    }

    uint64_t root_ref = 0ull;
    std::memcpy(&root_ref, bbytes + abi.bvh_root_ofs, sizeof(uint64_t));
    std::memcpy(&out.bvh_bounds, bbytes + abi.accel_bounds_ofs, sizeof(LBBox3faPOD));

    if (pk == kPrimKindTriangle4i) {
        if (!abi.supports_compact_vertices ||
            !is_readable_memory(sbytes + abi.scene_vertices_vec_data_ofs, sizeof(float **))) {
            return false;
        }
        float **arr = nullptr;
        std::memcpy(&arr, sbytes + abi.scene_vertices_vec_data_ofs, sizeof(float **));
        if (arr == nullptr || !is_readable_memory(arr, sizeof(float *))) {
            return false;
        }
        out.vertices_slot = arr;
    }

    out.bvh_ptr = bvh_ptr;
    out.branching_factor = bf_type;
    out.prim_kind = pk;
    out.intersect_variant = iv;
    out.required_isa_mask = req_isa;
    out.bvh_root = root_ref;
    return true;
}

template<typename PrimPOD>
[[nodiscard]] bool validate_leaf_block_non_compact(
    const PrimPOD &prim,
    uint64_t triangle_count) noexcept {
    if (prim.geomIDs[0] == ~0u) {
        return false;
    }
    for (size_t i = 0u; i < 4u; ++i) {
        if (prim.geomIDs[i] != ~0u) {
            if (prim.geomIDs[i] != 0u || static_cast<uint64_t>(prim.primIDs[i]) >= triangle_count) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool validate_leaf_block_4i(
    const Triangle4iPOD &prim,
    uint64_t vertex_count,
    uint64_t triangle_count,
    uint64_t vertex_stride_bytes) noexcept {
    if (prim.primIDs[0] == ~0u) {
        return false;
    }
    const uint64_t float_stride = vertex_stride_bytes / 4ull;
    if (float_stride < 3ull || vertex_count == 0ull) {
        return false;
    }
    const uint64_t max_float_ofs = (vertex_count - 1ull) * float_stride;
    for (size_t i = 0u; i < 4u; ++i) {
        if (prim.geomIDs[i] != 0u) {
            return false;
        }
        const uint32_t pid = prim.primIDs[i];
        if (pid != ~0u && static_cast<uint64_t>(pid) >= triangle_count) {
            return false;
        }
        if (static_cast<uint64_t>(prim.v0_[i]) > max_float_ofs ||
            static_cast<uint64_t>(prim.v1_[i]) > max_float_ofs ||
            static_cast<uint64_t>(prim.v2_[i]) > max_float_ofs ||
            (static_cast<uint64_t>(prim.v0_[i]) % float_stride) != 0ull ||
            (static_cast<uint64_t>(prim.v1_[i]) % float_stride) != 0ull ||
            (static_cast<uint64_t>(prim.v2_[i]) % float_stride) != 0ull) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] uint64_t encode_leaf_ref(uint64_t block_idx, uint64_t num_blocks) noexcept {
    return kRelocLeafFlag |
           ((num_blocks & kRelocBlockCountMask) << kRelocBlockCountShift) |
           (block_idx & kRelocIndexMask);
}

[[nodiscard]] bool decode_reloc_ref(
    uint64_t reloc_ref,
    size_t node_bytes,
    size_t prim_bytes,
    uint64_t node_count,
    uint64_t leaf_block_count,
    const std::byte *nodes_base,
    const std::byte *leaves_base,
    uint64_t &out_native_ref) noexcept {
    if (reloc_ref == kRelocEmpty) {
        out_native_ref = kEmptyNode;
        return true;
    }
    if ((reloc_ref & kRelocLeafFlag) != 0ull) {
        const uint64_t num_blocks = (reloc_ref >> kRelocBlockCountShift) & kRelocBlockCountMask;
        const uint64_t leaf_idx = reloc_ref & kRelocIndexMask;
        if (num_blocks == 0ull || num_blocks > kMaxLeafBlocks ||
            leaf_idx >= leaf_block_count ||
            num_blocks > leaf_block_count - leaf_idx) {
            return false;
        }
        const auto addr = reinterpret_cast<uintptr_t>(leaves_base + leaf_idx * prim_bytes);
        if ((addr & kAlignMask) != 0u) {
            return false;
        }
        out_native_ref = static_cast<uint64_t>(addr) | kTyLeaf | num_blocks;
        return true;
    }
    if (reloc_ref >= node_count) {
        return false;
    }
    const auto addr = reinterpret_cast<uintptr_t>(nodes_base + reloc_ref * node_bytes);
    if ((addr & kAlignMask) != 0u) {
        return false;
    }
    out_native_ref = static_cast<uint64_t>(addr);
    return true;
}

struct ExportPlan {
    struct NodeJob {
        const void *src;
        uint64_t reloc_children[8];
    };
    struct LeafJob {
        const void *src;
        uint64_t dst_block;
        uint64_t num_blocks;
    };
    InspectSceneResult inspected{};
    uint32_t node_bytes{0u};
    uint32_t prim_bytes{0u};
    uint32_t scene_flags{0u};
    uint32_t quality_flags{0u};
    uint64_t vertex_stride{0u};
    uint64_t vertex_count{0u};
    uint64_t triangle_count{0u};
    uint64_t root_reloc_ref{kRelocEmpty};
    uint64_t leaf_block_count{0u};
    std::vector<NodeJob> node_jobs;
    std::vector<LeafJob> leaf_jobs;

    [[nodiscard]] uint64_t unpacked_as_bytes() const noexcept {
        return static_cast<uint64_t>(node_jobs.size()) * node_bytes +
               leaf_block_count * prim_bytes;
    }

    [[nodiscard]] size_t payload_bytes() const noexcept {
        return sizeof(EmbreeStaticTriangleBVHPayloadHeader) +
               static_cast<size_t>(unpacked_as_bytes());
    }
};

[[nodiscard]] EmbreeArchiveStatus build_export_plan_oot(
    const EmbreeOutOfTreeABI &abi,
    const EmbreeMeshExportState &state,
    ExportPlan &out_plan) noexcept {
    out_plan = {};
    if (!abi.valid || !state.is_built || state.committed_rtc_scene == nullptr) {
        return EmbreeArchiveStatus::INVALID_ARGUMENT;
    }
    if (state.option.allow_update || static_cast<bool>(state.option.motion)) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }
    if (!inspect_committed_scene_bvh(abi, state.committed_rtc_scene, out_plan.inspected)) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }

    // Reject motion-blur bounds where bounds0 != bounds1
    for (int k = 0; k < 3; ++k) {
        if (out_plan.inspected.bvh_bounds.lo0[k] != out_plan.inspected.bvh_bounds.lo1[k] ||
            out_plan.inspected.bvh_bounds.hi0[k] != out_plan.inspected.bvh_bounds.hi1[k]) {
            return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
        }
    }

    const uint32_t bf = out_plan.inspected.branching_factor;
    const size_t node_bytes = (bf == 8u) ? sizeof(AABBNode8POD) : sizeof(AABBNode4POD);
    const size_t prim_bytes = prim_struct_bytes_for_kind(out_plan.inspected.prim_kind);
    if ((bf != 4u && bf != 8u) || prim_bytes == 0u) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }

    out_plan.node_bytes = static_cast<uint32_t>(node_bytes);
    out_plan.prim_bytes = static_cast<uint32_t>(prim_bytes);
    out_plan.scene_flags = expected_scene_flags_for_option(state.option);
    out_plan.quality_flags = expected_quality_flags_for_option(state.option);
    out_plan.vertex_stride = state.vertex_stride;
    out_plan.vertex_count = state.vertex_count;
    out_plan.triangle_count = state.triangle_count;

    struct StackItem {
        const void *src;
        uint64_t node_idx;
    };
    std::vector<StackItem> stack;

    auto map_child_ref = [&](uint64_t ref, uint64_t &out_reloc) noexcept -> EmbreeArchiveStatus {
        if ((ref & kBarrierMask) != 0ull || ref == kInvalidNode || ref == kPopRay) {
            return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
        }
        // emptyNode == kTyLeaf (8), so check kEmptyNode BEFORE (ref & kTyLeaf).
        if (ref == kEmptyNode) {
            out_reloc = kRelocEmpty;
            return EmbreeArchiveStatus::OK;
        }
        if ((ref & kTyLeaf) != 0ull) {
            const uint64_t num_blocks = ref & kMaxLeafBlocks;
            const void *leaf_ptr = reinterpret_cast<const void *>(ref & ~kAlignMask);
            if (leaf_ptr == nullptr || num_blocks == 0ull || num_blocks > kMaxLeafBlocks) {
                return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
            }
            if (out_plan.leaf_block_count + num_blocks > kRelocIndexMask) {
                return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
            }
            const uint64_t start_block = out_plan.leaf_block_count;
            out_plan.leaf_block_count += num_blocks;
            out_plan.leaf_jobs.push_back(ExportPlan::LeafJob{leaf_ptr, start_block, num_blocks});
            out_reloc = encode_leaf_ref(start_block, num_blocks);
            return EmbreeArchiveStatus::OK;
        }
        if ((ref & kAlignMask) != kTyAABBNode) {
            return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
        }
        const void *child_ptr = reinterpret_cast<const void *>(ref & ~kAlignMask);
        if (child_ptr == nullptr) {
            return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
        }
        const uint64_t child_idx = static_cast<uint64_t>(out_plan.node_jobs.size());
        if (child_idx + 1ull >= kRelocLeafFlag || child_idx >= kRelocIndexMask) {
            return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
        }
        out_plan.node_jobs.push_back(ExportPlan::NodeJob{child_ptr, {kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty}});
        stack.push_back(StackItem{child_ptr, child_idx});
        out_reloc = child_idx;
        return EmbreeArchiveStatus::OK;
    };

    const auto root_st = map_child_ref(out_plan.inspected.bvh_root, out_plan.root_reloc_ref);
    if (root_st != EmbreeArchiveStatus::OK) {
        return root_st;
    }

    while (!stack.empty()) {
        const StackItem item = stack.back();
        stack.pop_back();
        uint64_t children[8]{};
        std::memcpy(children, item.src, bf * sizeof(uint64_t));
        uint64_t relocs[8]{kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty, kRelocEmpty};
        for (uint32_t c = 0u; c < bf; ++c) {
            const auto st = map_child_ref(children[c], relocs[c]);
            if (st != EmbreeArchiveStatus::OK) {
                return st;
            }
        }
        for (uint32_t c = 0u; c < bf; ++c) {
            out_plan.node_jobs[static_cast<size_t>(item.node_idx)].reloc_children[c] = relocs[c];
        }
    }

    uint64_t total_payload = 0ull;
    uint64_t node_bytes_total = 0ull;
    if (!checked_mul_add(out_plan.node_jobs.size(), node_bytes, 0ull, node_bytes_total) ||
        !checked_mul_add(out_plan.leaf_block_count, prim_bytes, sizeof(EmbreeStaticTriangleBVHPayloadHeader) + node_bytes_total, total_payload) ||
        total_payload > kMaxEmbreePayloadBytes) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }
    return EmbreeArchiveStatus::OK;
}

void write_export_payload_oot(
    const EmbreeOutOfTreeABI &abi,
    const ExportPlan &plan,
    std::byte *out_buf) noexcept {
    EmbreeStaticTriangleBVHPayloadHeader hdr{};
    hdr.magic = kEmbreePayloadMagic;
    hdr.format_version = kEmbreePayloadFormatVersion;
    hdr.embree_version_number = abi.embree_version_number;
    hdr.build_config_flags = abi.build_config_flags;
    hdr.required_isa_mask = plan.inspected.required_isa_mask;
    hdr.branching_factor = plan.inspected.branching_factor;
    hdr.prim_kind = plan.inspected.prim_kind;
    hdr.intersect_variant = plan.inspected.intersect_variant;
    hdr.scene_flags = plan.scene_flags;
    hdr.quality_flags = plan.quality_flags;
    hdr.node_struct_bytes = plan.node_bytes;
    hdr.prim_struct_bytes = plan.prim_bytes;
    hdr.vertex_stride_bytes = plan.vertex_stride;
    hdr.vertex_count = plan.vertex_count;
    hdr.triangle_count = plan.triangle_count;
    hdr.num_primitives_in_bvh = plan.triangle_count;
    hdr.node_count = static_cast<uint64_t>(plan.node_jobs.size());
    hdr.leaf_block_count = plan.leaf_block_count;
    hdr.root_encoded_ref = plan.root_reloc_ref;
    std::memcpy(hdr.bounds_lower, plan.inspected.bvh_bounds.lo0, sizeof(hdr.bounds_lower));
    std::memcpy(hdr.bounds_upper, plan.inspected.bvh_bounds.hi0, sizeof(hdr.bounds_upper));
    std::memcpy(out_buf, &hdr, sizeof(hdr));

    const size_t nbytes = plan.node_bytes;
    const size_t pbytes = plan.prim_bytes;
    const size_t bf = plan.inspected.branching_factor;
    std::byte *const dst_nodes = out_buf + sizeof(hdr);
    std::byte *const dst_leaves = dst_nodes + plan.node_jobs.size() * nbytes;

    parallel_for_range(plan.node_jobs.size(), 1024u, [&](size_t b, size_t e) noexcept {
        for (size_t i = b; i < e; ++i) {
            std::byte *dst = dst_nodes + i * nbytes;
            std::memcpy(dst, plan.node_jobs[i].src, nbytes);
            std::memcpy(dst, plan.node_jobs[i].reloc_children, bf * sizeof(uint64_t));
        }
    });

    parallel_for_range(plan.leaf_jobs.size(), 2048u, [&](size_t b, size_t e) noexcept {
        for (size_t i = b; i < e; ++i) {
            const auto &lj = plan.leaf_jobs[i];
            std::memcpy(
                dst_leaves + lj.dst_block * pbytes,
                lj.src,
                static_cast<size_t>(lj.num_blocks) * pbytes);
        }
    });
}

struct ProbeGeometry {
    struct alignas(16) Vertex {
        float x, y, z, w;
    };
    struct Tri {
        uint32_t v0, v1, v2;
    };
    std::vector<Vertex> verts;
    std::vector<Tri> tris;
    float lo[3]{};
    float hi[3]{};
};

[[nodiscard]] ProbeGeometry make_probe_geometry(float base_x, float base_y, float base_z) noexcept {
    ProbeGeometry g{};
    // 16 disjoint quads = 32 triangles spanning [base + 0 .. base + 18] cleanly in all 3 axes.
    g.verts.reserve(64u);
    g.tris.reserve(32u);
    g.lo[0] = base_x;
    g.lo[1] = base_y;
    g.lo[2] = base_z;
    g.hi[0] = base_x + 18.0f;
    g.hi[1] = base_y + 18.0f;
    g.hi[2] = base_z + 20.0f;
    for (uint32_t q = 0u; q < 16u; ++q) {
        const float qx = base_x + static_cast<float>((q % 4u) * 5u);
        const float qy = base_y + static_cast<float>((q / 4u) * 5u);
        const float qz = base_z + static_cast<float>((q * 20u) / 15u);
        const float x0 = qx;
        const float x1 = std::min(base_x + 18.0f, qx + 3.0f);
        const float y0 = qy;
        const float y1 = std::min(base_y + 18.0f, qy + 3.0f);
        const uint32_t base_v = static_cast<uint32_t>(g.verts.size());
        g.verts.push_back({x0, y0, qz, 0.f});
        g.verts.push_back({x1, y0, qz, 0.f});
        g.verts.push_back({x1, y1, qz, 0.f});
        g.verts.push_back({x0, y1, qz, 0.f});
        g.tris.push_back({base_v + 0u, base_v + 1u, base_v + 2u});
        g.tris.push_back({base_v + 0u, base_v + 2u, base_v + 3u});
    }
    return g;
}

[[nodiscard]] RTCScene build_probe_scene(
    RTCDevice dev,
    const ProbeGeometry &pg,
    bool compact,
    RTCGeometry &out_geom) noexcept {
    RTCScene sc = rtcNewScene(dev);
    if (sc == nullptr) {
        return nullptr;
    }
    uint32_t flags = static_cast<uint32_t>(RTC_SCENE_FLAG_ROBUST) |
                     static_cast<uint32_t>(RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
    if (compact) {
        flags |= static_cast<uint32_t>(RTC_SCENE_FLAG_COMPACT);
    }
    rtcSetSceneFlags(sc, static_cast<RTCSceneFlags>(flags));
    rtcSetSceneBuildQuality(sc, RTC_BUILD_QUALITY_MEDIUM);

    out_geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (out_geom == nullptr) {
        rtcReleaseScene(sc);
        return nullptr;
    }
    rtcSetGeometryBuildQuality(out_geom, RTC_BUILD_QUALITY_MEDIUM);
    rtcSetGeometryEnableFilterFunctionFromArguments(out_geom, true);
    rtcSetGeometryMask(out_geom, ~0u);
    rtcSetSharedGeometryBuffer(
        out_geom, RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT3,
        pg.verts.data(), 0u, sizeof(ProbeGeometry::Vertex), pg.verts.size());
    rtcSetSharedGeometryBuffer(
        out_geom, RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT3,
        pg.tris.data(), 0u, sizeof(ProbeGeometry::Tri), pg.tris.size());
    rtcAttachGeometry(sc, out_geom);
    rtcReleaseGeometry(out_geom);
    rtcCommitGeometry(out_geom);
    rtcCommitScene(sc);
    return sc;
}

[[nodiscard]] bool verify_aabb_node_children_union(
    uint64_t root_ref,
    uint32_t bf,
    const float expected_lo[3],
    const float expected_hi[3]) noexcept {
    if ((root_ref & kBarrierMask) != 0ull || (root_ref & kAlignMask) != kTyAABBNode || root_ref == kEmptyNode) {
        return false;
    }
    const auto *node_ptr = reinterpret_cast<const std::byte *>(root_ref);
    const size_t node_bytes = (bf == 8u) ? sizeof(AABBNode8POD) : sizeof(AABBNode4POD);
    if (!is_readable_memory(node_ptr, node_bytes)) {
        return false;
    }
    uint64_t children[8]{};
    float lx[8]{}, ux[8]{}, ly[8]{}, uy[8]{}, lz[8]{}, uz[8]{};
    if (bf == 8u) {
        AABBNode8POD n{};
        std::memcpy(&n, node_ptr, sizeof(n));
        std::memcpy(children, n.children, sizeof(n.children));
        std::memcpy(lx, n.lower_x, sizeof(n.lower_x));
        std::memcpy(ux, n.upper_x, sizeof(n.upper_x));
        std::memcpy(ly, n.lower_y, sizeof(n.lower_y));
        std::memcpy(uy, n.upper_y, sizeof(n.upper_y));
        std::memcpy(lz, n.lower_z, sizeof(n.lower_z));
        std::memcpy(uz, n.upper_z, sizeof(n.upper_z));
    } else {
        AABBNode4POD n{};
        std::memcpy(&n, node_ptr, sizeof(n));
        std::memcpy(children, n.children, 32u);
        std::memcpy(lx, n.lower_x, 16u);
        std::memcpy(ux, n.upper_x, 16u);
        std::memcpy(ly, n.lower_y, 16u);
        std::memcpy(uy, n.upper_y, 16u);
        std::memcpy(lz, n.lower_z, 16u);
        std::memcpy(uz, n.upper_z, 16u);
    }

    float lo[3]{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    float hi[3]{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    size_t valid_cnt = 0u;
    for (uint32_t c = 0u; c < bf; ++c) {
        if (children[c] == kEmptyNode) {
            continue;
        }
        ++valid_cnt;
        lo[0] = std::min(lo[0], lx[c]);
        lo[1] = std::min(lo[1], ly[c]);
        lo[2] = std::min(lo[2], lz[c]);
        hi[0] = std::max(hi[0], ux[c]);
        hi[1] = std::max(hi[1], uy[c]);
        hi[2] = std::max(hi[2], uz[c]);
    }
    if (valid_cnt == 0u) {
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) ||
            std::fabs(lo[k] - expected_lo[k]) > 1e-2f ||
            std::fabs(hi[k] - expected_hi[k]) > 1e-2f) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool verify_compact_vertices_slot_live(
    RTCScene sc,
    size_t vertices_vec_data_ofs,
    const ProbeGeometry &pgA) noexcept {
    const auto *sbytes = static_cast<const std::byte *>(static_cast<const void *>(sc));
    float **arr = nullptr;
    std::memcpy(&arr, sbytes + vertices_vec_data_ofs, sizeof(float **));
    if (arr == nullptr || !is_readable_memory(arr, sizeof(float *)) ||
        arr[0] != reinterpret_cast<const float *>(pgA.verts.data())) {
        return false;
    }

    // Modify quad 0 vertices strictly inside its existing [11..14, 13..16, 17] leaf AABB
    // so Embree's BVH slab test visits the Triangle4i leaf intersector and reads *arr[0].
    std::vector<ProbeGeometry::Vertex> modified = pgA.verts;
    modified[0].x = pgA.lo[0] + 1.5f;
    modified[3].x = pgA.lo[0] + 1.5f;

    auto trace_at_xy = [&](float ox, float oy) noexcept -> RTCRayHit {
        RTCRayHit rh{};
        rh.ray.org_x = ox;
        rh.ray.org_y = oy;
        rh.ray.org_z = pgA.lo[2] + 5.0f;
        rh.ray.dir_x = 0.0f;
        rh.ray.dir_y = 0.0f;
        rh.ray.dir_z = -1.0f;
        rh.ray.tnear = 0.0f;
        rh.ray.tfar = 100.0f;
        rh.ray.mask = ~0u;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(sc, &rh, nullptr);
        return rh;
    };

    const RTCRayHit miss_probe_orig = trace_at_xy(pgA.lo[0] + 0.5f, pgA.lo[1] + 0.25f);
    const RTCRayHit bary_probe_orig = trace_at_xy(pgA.lo[0] + 2.5f, pgA.lo[1] + 0.5f);
    if (miss_probe_orig.hit.geomID != 0u || miss_probe_orig.hit.primID != 0u ||
        bary_probe_orig.hit.geomID != 0u || bary_probe_orig.hit.primID != 0u) {
        return false;
    }

    float *orig_ptr = arr[0];
    arr[0] = reinterpret_cast<float *>(modified.data());
    const RTCRayHit miss_probe_swap = trace_at_xy(pgA.lo[0] + 0.5f, pgA.lo[1] + 0.25f);
    const RTCRayHit bary_probe_swap = trace_at_xy(pgA.lo[0] + 2.5f, pgA.lo[1] + 0.5f);
    arr[0] = orig_ptr;

    const RTCRayHit miss_probe_restored = trace_at_xy(pgA.lo[0] + 0.5f, pgA.lo[1] + 0.25f);
    const bool swap_effect_ok =
        miss_probe_swap.hit.geomID == RTC_INVALID_GEOMETRY_ID &&
        bary_probe_swap.hit.geomID == 0u &&
        bary_probe_swap.hit.primID == 0u &&
        std::fabs(bary_probe_swap.hit.u - bary_probe_orig.hit.u) > 0.05f;
    const bool restored_ok =
        miss_probe_restored.hit.geomID == 0u && miss_probe_restored.hit.primID == 0u;
    return swap_effect_ok && restored_ok;
}

[[nodiscard]] bool run_live_roundtrip_self_test(
    RTCDevice dev,
    const EmbreeOutOfTreeABI &abi,
    RTCScene src_scene,
    const ProbeGeometry &pgA,
    bool compact,
    luisa::string &out_diag) noexcept {
    EmbreeMeshExportState state{};
    state.is_built = true;
    state.option.hint = AccelOption::UsageHint::FAST_BUILD;
    state.option.allow_compaction = compact;
    state.triangle_count = pgA.tris.size();
    state.vertex_count = pgA.verts.size();
    state.vertex_stride = sizeof(ProbeGeometry::Vertex);
    state.committed_rtc_scene = src_scene;

    ExportPlan plan{};
    const auto plan_st = build_export_plan_oot(abi, state, plan);
    if (plan_st != EmbreeArchiveStatus::OK) {
        out_diag = luisa::format("self_test_export_plan_failed(compact={},st={})", compact ? 1 : 0, static_cast<uint32_t>(plan_st));
        return false;
    }

    luisa::vector<std::byte> payload(plan.payload_bytes());
    write_export_payload_oot(abi, plan, payload.data());

    // Create fresh destination scene with dummy 3-vertex/1-triangle mesh and commit it.
    RTCScene dst_scene = rtcNewScene(dev);
    if (dst_scene == nullptr) {
        out_diag = "self_test_dst_scene_null";
        return false;
    }
    auto release_dst = vstd::scope_exit([&]() noexcept {
        if (dst_scene != nullptr) {
            rtcReleaseScene(dst_scene);
        }
    });

    uint32_t flags = static_cast<uint32_t>(RTC_SCENE_FLAG_ROBUST) |
                     static_cast<uint32_t>(RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
    if (compact) {
        flags |= static_cast<uint32_t>(RTC_SCENE_FLAG_COMPACT);
    }
    rtcSetSceneFlags(dst_scene, static_cast<RTCSceneFlags>(flags));
    rtcSetSceneBuildQuality(dst_scene, RTC_BUILD_QUALITY_MEDIUM);

    RTCGeometry dst_geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetGeometryBuildQuality(dst_geom, RTC_BUILD_QUALITY_MEDIUM);
    rtcSetGeometryEnableFilterFunctionFromArguments(dst_geom, true);
    rtcSetGeometryMask(dst_geom, ~0u);
    rtcAttachGeometry(dst_scene, dst_geom);

    EmbreeResolvedTriangleGeometry geom{};
    geom.vertex_buffer = reinterpret_cast<const std::byte *>(pgA.verts.data());
    geom.vertex_offset = 0u;
    geom.vertex_stride = sizeof(ProbeGeometry::Vertex);
    geom.vertex_count = pgA.verts.size();
    geom.triangle_buffer = reinterpret_cast<const std::byte *>(pgA.tris.data());
    geom.triangle_offset = 0u;
    geom.triangle_count = pgA.tris.size();

    EmbreeImportedBVHHandle imported{};
    size_t unpacked_bytes = 0u;
    const auto imp_st = embree_commit_dummy_and_import_bvh_oot(
        abi, dst_scene, dst_geom, geom, luisa::span<const std::byte>{payload.data(), payload.size()}, imported, unpacked_bytes);
    rtcReleaseGeometry(dst_geom);
    if (imp_st != EmbreeArchiveStatus::OK) {
        out_diag = luisa::format("self_test_import_failed(compact={},st={})", compact ? 1 : 0, static_cast<uint32_t>(imp_st));
        return false;
    }

    // Wrap src_scene and dst_scene in transformed TLAS instances and compare 64 rays bit-exact!
    auto make_tlas = [&](RTCScene child) noexcept -> RTCScene {
        RTCScene tl = rtcNewScene(dev);
        rtcSetSceneFlags(tl, static_cast<RTCSceneFlags>(RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS));
        rtcSetSceneBuildQuality(tl, RTC_BUILD_QUALITY_MEDIUM);
        RTCGeometry inst = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(inst, child);
        rtcSetGeometryTimeStepCount(inst, 1u);
        rtcSetGeometryMask(inst, ~0u);
        const float xf[12]{
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
            0.f, 0.f, 1.f,
            3.0f, -2.0f, 1.0f};
        rtcSetGeometryTransform(inst, 0u, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, xf);
        rtcCommitGeometry(inst);
        rtcAttachGeometryByID(tl, inst, 7u);
        rtcReleaseGeometry(inst);
        rtcCommitScene(tl);
        return tl;
    };

    RTCScene tl_ref = make_tlas(src_scene);
    RTCScene tl_imp = make_tlas(dst_scene);
    auto release_tlas = vstd::scope_exit([&]() noexcept {
        if (tl_ref != nullptr) rtcReleaseScene(tl_ref);
        if (tl_imp != nullptr) rtcReleaseScene(tl_imp);
        imported.restore_and_release();
    });

    size_t hit_count = 0u;
    for (int ry = 0; ry < 8; ++ry) {
        for (int rx = 0; rx < 8; ++rx) {
            const float ox = pgA.lo[0] + 3.0f + (static_cast<float>(rx) + 0.37f) * 2.2f;
            const float oy = pgA.lo[1] - 2.0f + (static_cast<float>(ry) + 0.41f) * 2.2f;
            const float oz = pgA.hi[2] + 5.0f;

            RTCRayHit rh_ref{}, rh_imp{};
            auto init_ray = [&](RTCRay &r) noexcept {
                r.org_x = ox;
                r.org_y = oy;
                r.org_z = oz;
                r.dir_x = 0.0f;
                r.dir_y = 0.0f;
                r.dir_z = -1.0f;
                r.tnear = 0.0f;
                r.tfar = 100.0f;
                r.mask = ~0u;
            };
            init_ray(rh_ref.ray);
            init_ray(rh_imp.ray);
            rh_ref.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh_ref.hit.primID = RTC_INVALID_GEOMETRY_ID;
            rh_ref.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            rh_imp.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh_imp.hit.primID = RTC_INVALID_GEOMETRY_ID;
            rh_imp.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            rtcIntersect1(tl_ref, &rh_ref, nullptr);
            rtcIntersect1(tl_imp, &rh_imp, nullptr);
            if (rh_ref.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                ++hit_count;
            }
            if (rh_ref.hit.geomID != rh_imp.hit.geomID ||
                rh_ref.hit.primID != rh_imp.hit.primID ||
                rh_ref.hit.instID[0] != rh_imp.hit.instID[0] ||
                rh_ref.ray.tfar != rh_imp.ray.tfar ||
                rh_ref.hit.u != rh_imp.hit.u ||
                rh_ref.hit.v != rh_imp.hit.v) {
                out_diag = luisa::format(
                    "self_test_intersect_mismatch(compact={},rx={},ry={},prim={}/{})",
                    compact ? 1 : 0, rx, ry, rh_ref.hit.primID, rh_imp.hit.primID);
                return false;
            }

            RTCRay occ_ref{}, occ_imp{};
            init_ray(occ_ref);
            init_ray(occ_imp);
            rtcOccluded1(tl_ref, &occ_ref, nullptr);
            rtcOccluded1(tl_imp, &occ_imp, nullptr);
            if ((occ_ref.tfar < 0.0f) != (occ_imp.tfar < 0.0f)) {
                out_diag = luisa::format("self_test_occluded_mismatch(compact={})", compact ? 1 : 0);
                return false;
            }
        }
    }
    if (hit_count == 0u) {
        out_diag = luisa::format("self_test_zero_hits(compact={})", compact ? 1 : 0);
        return false;
    }
    return true;
}

[[nodiscard]] bool probe_embree_oot_abi(
    RTCDevice dev,
    EmbreeOutOfTreeABI &out_abi,
    luisa::string &out_diag) noexcept {
    out_abi = {};
    if (dev == nullptr) {
        out_diag = "embree4_rtc_device_null";
        return false;
    }
    if (sizeof(void *) != 8u) {
        out_diag = "embree4_requires_64bit";
        return false;
    }
    const ssize_t ver = rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_VERSION);
    if (ver < 40000 || ver >= 50000) {
        out_diag = luisa::format("embree_unsupported_version({})", ver);
        return false;
    }
    if (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED) != 0) {
        out_diag = "embree4_compact_polys_unsupported";
        return false;
    }

    const ProbeGeometry pgA = make_probe_geometry(11.0f, 13.0f, 17.0f);
    const ProbeGeometry pgB = make_probe_geometry(111.0f, 113.0f, 117.0f);

    RTCGeometry gA_nc = nullptr, gB_nc = nullptr, gA_c = nullptr, gB_c = nullptr;
    RTCScene sA_nc = build_probe_scene(dev, pgA, false, gA_nc);
    RTCScene sB_nc = build_probe_scene(dev, pgB, false, gB_nc);
    RTCScene sA_c = build_probe_scene(dev, pgA, true, gA_c);
    RTCScene sB_c = build_probe_scene(dev, pgB, true, gB_c);
    auto release_probes = vstd::scope_exit([&]() noexcept {
        if (sA_nc) rtcReleaseScene(sA_nc);
        if (sB_nc) rtcReleaseScene(sB_nc);
        if (sA_c) rtcReleaseScene(sA_c);
        if (sB_c) rtcReleaseScene(sB_c);
    });
    if (!sA_nc || !sB_nc || !sA_c || !sB_c ||
        !is_readable_memory(sA_nc, 256u) ||
        !is_readable_memory(sB_nc, 256u) ||
        !is_readable_memory(sA_c, 256u) ||
        !is_readable_memory(sB_c, 256u)) {
        out_diag = "embree4_probe_scene_create_or_read_failed";
        return false;
    }

    const auto *pA_nc = reinterpret_cast<const std::byte *>(sA_nc);
    const auto *pB_nc = reinterpret_cast<const std::byte *>(sB_nc);
    const auto *pA_c = reinterpret_cast<const std::byte *>(sA_c);
    const auto *pB_c = reinterpret_cast<const std::byte *>(sB_c);

    // 1. Discover AccelData::bounds offset (64-byte LBBox3fa aligned to 16B).
    size_t found_bounds_ofs = SIZE_MAX;
    for (size_t ofs = 0u; ofs <= 256u; ofs += 8u) {
        if (!is_readable_memory(pA_nc + ofs, 64u) ||
            !is_readable_memory(pB_nc + ofs, 64u) ||
            !is_readable_memory(pA_c + ofs, 64u) ||
            !is_readable_memory(pB_c + ofs, 64u)) {
            continue;
        }
        if (lbbox_matches(pA_nc + ofs, pgA.lo, pgA.hi) &&
            lbbox_matches(pB_nc + ofs, pgB.lo, pgB.hi) &&
            lbbox_matches(pA_c + ofs, pgA.lo, pgA.hi) &&
            lbbox_matches(pB_c + ofs, pgB.lo, pgB.hi)) {
            found_bounds_ofs = ofs;
            break;
        }
    }
    if (found_bounds_ofs == SIZE_MAX) {
        out_diag = "embree4_probe_bounds_offset_not_found";
        return false;
    }

    // 2. Discover Scene::intersectors.ptr, Intersector1::name, AccelData::type, BVH::device/scene/root.
    size_t found_sc_isect_ofs = SIZE_MAX;
    size_t found_isect1_name_ofs = SIZE_MAX;
    size_t found_type_ofs = SIZE_MAX;
    size_t found_bvh_dev_ofs = SIZE_MAX;
    uint32_t nc_bf = 0u, c_bf = 0u;

    auto read_ptr = [](const std::byte *base, size_t ofs) noexcept -> const std::byte * {
        const std::byte *p = nullptr;
        std::memcpy(&p, base + ofs, sizeof(const std::byte *));
        return p;
    };

    const size_t min_bvh_probe_bytes = std::max<size_t>(256u, found_bounds_ofs + 96u);
    uint32_t diag_lbbox = 0u, diag_type = 0u, diag_name = 0u, diag_dev = 0u, diag_root = 0u;
    for (size_t s_ofs = 0u; s_ofs <= 1024u; s_ofs += 8u) {
        if (!is_readable_memory(pA_nc + s_ofs, sizeof(void *)) ||
            !is_readable_memory(pB_nc + s_ofs, sizeof(void *)) ||
            !is_readable_memory(pA_c + s_ofs, sizeof(void *)) ||
            !is_readable_memory(pB_c + s_ofs, sizeof(void *))) {
            continue;
        }
        const std::byte *bA_nc = read_ptr(pA_nc, s_ofs);
        const std::byte *bB_nc = read_ptr(pB_nc, s_ofs);
        const std::byte *bA_c = read_ptr(pA_c, s_ofs);
        const std::byte *bB_c = read_ptr(pB_c, s_ofs);
        if (!is_readable_memory(bA_nc, min_bvh_probe_bytes) ||
            !is_readable_memory(bB_nc, min_bvh_probe_bytes) ||
            !is_readable_memory(bA_c, min_bvh_probe_bytes) ||
            !is_readable_memory(bB_c, min_bvh_probe_bytes)) {
            continue;
        }
        if (!lbbox_matches(bA_nc + found_bounds_ofs, pgA.lo, pgA.hi) ||
            !lbbox_matches(bB_nc + found_bounds_ofs, pgB.lo, pgB.hi) ||
            !lbbox_matches(bA_c + found_bounds_ofs, pgA.lo, pgA.hi) ||
            !lbbox_matches(bB_c + found_bounds_ofs, pgB.lo, pgB.hi)) {
            continue;
        }
        ++diag_lbbox;

        // Check AccelData::type right after LBBox3fa.
        size_t type_ofs = SIZE_MAX;
        uint32_t tA_nc = 0u, tB_nc = 0u, tA_c = 0u, tB_c = 0u;
        for (size_t tofs = found_bounds_ofs + 64u; tofs <= found_bounds_ofs + 88u; tofs += 4u) {
            std::memcpy(&tA_nc, bA_nc + tofs, 4u);
            std::memcpy(&tB_nc, bB_nc + tofs, 4u);
            std::memcpy(&tA_c, bA_c + tofs, 4u);
            std::memcpy(&tB_c, bB_c + tofs, 4u);
            if ((tA_nc == kAccelTypeBVH4 || tA_nc == kAccelTypeBVH8) &&
                tA_nc == tB_nc &&
                tA_c == kAccelTypeBVH4 &&
                tB_c == kAccelTypeBVH4) {
                type_ofs = tofs;
                break;
            }
        }
        if (type_ofs == SIZE_MAX) {
            continue;
        }
        ++diag_type;

        // Find Intersector1::name offset inside Scene::intersectors.
        size_t name_rel_ofs = SIZE_MAX;
        for (size_t nofs = 8u; nofs <= 128u; nofs += 8u) {
            if (!is_readable_memory(pA_nc + s_ofs + nofs, sizeof(void *)) ||
                !is_readable_memory(pB_nc + s_ofs + nofs, sizeof(void *)) ||
                !is_readable_memory(pA_c + s_ofs + nofs, sizeof(void *)) ||
                !is_readable_memory(pB_c + s_ofs + nofs, sizeof(void *))) {
                continue;
            }
            std::string_view svA_nc, svB_nc, svA_c, svB_c;
            if (!try_read_cstring(reinterpret_cast<const char *>(read_ptr(pA_nc, s_ofs + nofs)), 160u, svA_nc) ||
                !try_read_cstring(reinterpret_cast<const char *>(read_ptr(pB_nc, s_ofs + nofs)), 160u, svB_nc) ||
                !try_read_cstring(reinterpret_cast<const char *>(read_ptr(pA_c, s_ofs + nofs)), 160u, svA_c) ||
                !try_read_cstring(reinterpret_cast<const char *>(read_ptr(pB_c, s_ofs + nofs)), 160u, svB_c)) {
                continue;
            }
            uint32_t bf_a = 0u, pk_a = 0u, iv_a = 0u, isa_a = 0u;
            uint32_t bf_b = 0u, pk_b = 0u, iv_b = 0u, isa_b = 0u;
            uint32_t bf_ac = 0u, pk_ac = 0u, iv_ac = 0u, isa_ac = 0u;
            uint32_t bf_bc = 0u, pk_bc = 0u, iv_bc = 0u, isa_bc = 0u;
            if (!parse_intersector1_name(svA_nc, bf_a, pk_a, iv_a, isa_a) ||
                !parse_intersector1_name(svB_nc, bf_b, pk_b, iv_b, isa_b) ||
                !parse_intersector1_name(svA_c, bf_ac, pk_ac, iv_ac, isa_ac) ||
                !parse_intersector1_name(svB_c, bf_bc, pk_bc, iv_bc, isa_bc)) {
                continue;
            }
            const uint32_t expect_nc_bf = (tA_nc == kAccelTypeBVH8) ? 8u : 4u;
            if (bf_a == expect_nc_bf && bf_b == expect_nc_bf &&
                bf_ac == 4u && bf_bc == 4u &&
                pk_a == kPrimKindTriangle4v && pk_b == kPrimKindTriangle4v &&
                pk_ac == kPrimKindTriangle4i && pk_bc == kPrimKindTriangle4i) {
                name_rel_ofs = nofs;
                nc_bf = expect_nc_bf;
                c_bf = 4u;
                break;
            }
        }
        if (name_rel_ofs == SIZE_MAX) {
            continue;
        }
        ++diag_name;

        // Find 8B-aligned BVHN::device followed by BVHN::scene followed by BVHN::root.
        size_t dev_ofs = SIZE_MAX;
        const size_t dev_ofs_start = (type_ofs + 4u + 7u) & (~size_t{7u});
        for (size_t dofs = dev_ofs_start; dofs <= 224u; dofs += 8u) {
            if (read_ptr(bA_nc, dofs) == reinterpret_cast<const std::byte *>(dev) &&
                read_ptr(bA_nc, dofs + 8u) == pA_nc &&
                read_ptr(bB_nc, dofs) == reinterpret_cast<const std::byte *>(dev) &&
                read_ptr(bB_nc, dofs + 8u) == pB_nc &&
                read_ptr(bA_c, dofs) == reinterpret_cast<const std::byte *>(dev) &&
                read_ptr(bA_c, dofs + 8u) == pA_c &&
                read_ptr(bB_c, dofs) == reinterpret_cast<const std::byte *>(dev) &&
                read_ptr(bB_c, dofs + 8u) == pB_c) {
                ++diag_dev;
                uint64_t rA_nc = 0ull, rB_nc = 0ull, rA_c = 0ull, rB_c = 0ull;
                std::memcpy(&rA_nc, bA_nc + dofs + 16u, sizeof(uint64_t));
                std::memcpy(&rB_nc, bB_nc + dofs + 16u, sizeof(uint64_t));
                std::memcpy(&rA_c, bA_c + dofs + 16u, sizeof(uint64_t));
                std::memcpy(&rB_c, bB_c + dofs + 16u, sizeof(uint64_t));
                if (verify_aabb_node_children_union(rA_nc, nc_bf, pgA.lo, pgA.hi) &&
                    verify_aabb_node_children_union(rB_nc, nc_bf, pgB.lo, pgB.hi) &&
                    verify_aabb_node_children_union(rA_c, c_bf, pgA.lo, pgA.hi) &&
                    verify_aabb_node_children_union(rB_c, c_bf, pgB.lo, pgB.hi)) {
                    ++diag_root;
                    dev_ofs = dofs;
                    break;
                }
            }
        }
        if (dev_ofs == SIZE_MAX) {
            continue;
        }

        found_sc_isect_ofs = s_ofs;
        found_isect1_name_ofs = name_rel_ofs;
        found_type_ofs = type_ofs;
        found_bvh_dev_ofs = dev_ofs;
        break;
    }

    if (found_sc_isect_ofs == SIZE_MAX || found_bvh_dev_ofs == SIZE_MAX) {
        out_diag = luisa::format(
            "embree4_probe_bvh_offsets_not_found(b={},t={},n={},d={},r={})",
            diag_lbbox, diag_type, diag_name, diag_dev, diag_root);
        return false;
    }

    // 3. Discover Scene::vertices vector data pointer offset on compact Triangle4i scenes + live test!
    size_t found_verts_ofs = SIZE_MAX;
    uint32_t diag_verts_cand = 0u;
    for (size_t vofs = 0u; vofs <= 2048u; vofs += 8u) {
        if (!is_readable_memory(pA_c + vofs, sizeof(void *)) ||
            !is_readable_memory(pB_c + vofs, sizeof(void *))) {
            continue;
        }
        const std::byte *arrA = read_ptr(pA_c, vofs);
        const std::byte *arrB = read_ptr(pB_c, vofs);
        if (!is_readable_memory(arrA, sizeof(float *)) ||
            !is_readable_memory(arrB, sizeof(float *))) {
            continue;
        }
        if (read_ptr(arrA, 0u) == reinterpret_cast<const std::byte *>(pgA.verts.data()) &&
            read_ptr(arrB, 0u) == reinterpret_cast<const std::byte *>(pgB.verts.data())) {
            ++diag_verts_cand;
            if (verify_compact_vertices_slot_live(sA_c, vofs, pgA)) {
                found_verts_ofs = vofs;
                break;
            }
        }
    }
    if (found_verts_ofs == SIZE_MAX) {
        out_diag = luisa::format("embree4_probe_vertices_slot_failed(cand={})", diag_verts_cand);
        return false;
    }

    out_abi.valid = true;
    out_abi.embree_version_number = static_cast<uint32_t>(ver);
    out_abi.build_config_flags = query_embree_build_config_flags(dev);
    out_abi.cpu_isa_mask = detect_cpu_isa_mask();
    out_abi.accel_bounds_ofs = found_bounds_ofs;
    out_abi.accel_type_ofs = found_type_ofs;
    out_abi.scene_intersectors_ptr_ofs = found_sc_isect_ofs;
    out_abi.intersector1_name_ofs = found_isect1_name_ofs;
    out_abi.bvh_device_ofs = found_bvh_dev_ofs;
    out_abi.bvh_root_ofs = found_bvh_dev_ofs + 16u;
    out_abi.scene_vertices_vec_data_ofs = found_verts_ofs;
    out_abi.supports_bvh4 = true;
    out_abi.supports_bvh8 = (nc_bf == 8u) || (rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED) != 0);
    out_abi.supports_compact_vertices = true;

    // 4. Full live round-trip export -> dummy-commit import -> TLAS transform -> bit-exact ray parity!
    if (!run_live_roundtrip_self_test(dev, out_abi, sA_nc, pgA, false, out_diag) ||
        !run_live_roundtrip_self_test(dev, out_abi, sA_c, pgA, true, out_diag)) {
        out_abi = {};
        return false;
    }

    return true;
}

#endif// LUISA_EMBREE_OOT_ENABLED

[[nodiscard]] AccelArchiveResult map_embree_status(
    EmbreeArchiveStatus st,
    std::string_view op_name) noexcept {
    switch (st) {
        case EmbreeArchiveStatus::OK:
            return {};
        case EmbreeArchiveStatus::INVALID_ARGUMENT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                luisa::format("{} rejected invalid Embree arguments.", op_name));
        case EmbreeArchiveStatus::UNSUPPORTED_LAYOUT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::UNSUPPORTED_LAYOUT,
                luisa::format("{} encountered an unsupported Embree BVH layout or primitive type.", op_name));
        case EmbreeArchiveStatus::CORRUPT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                luisa::format("{} rejected a corrupt Embree BVH payload.", op_name));
        case EmbreeArchiveStatus::INCOMPATIBLE_CONFIG:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::DRIVER_INCOMPATIBLE,
                luisa::format("{} rejected incompatible Embree version, build config, or geometry metadata.", op_name));
        case EmbreeArchiveStatus::INCOMPATIBLE_ISA:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::DEVICE_INCOMPATIBLE,
                luisa::format("{} rejected an Embree BVH requiring unsupported CPU ISA features.", op_name));
        case EmbreeArchiveStatus::BUFFER_TOO_SMALL:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                luisa::format("{} buffer capacity was insufficient.", op_name));
        case EmbreeArchiveStatus::INTERNAL_ERROR:
        default:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                luisa::format("{} failed with internal Embree error code {}.", op_name, static_cast<uint32_t>(st)));
    }
}

}// namespace

EmbreeArchiveStatus embree_import_static_triangle_bvh_oot(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    EmbreeImportedBVHHandle &out_handle,
    size_t &out_unpacked_bytes) noexcept {
    out_unpacked_bytes = 0u;
    out_handle.restore_and_release();

#if !LUISA_EMBREE_OOT_ENABLED
    (void)abi;
    (void)rtc_scene;
    (void)geom;
    (void)bvh_payload;
    return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
#else
    if (!abi.valid || rtc_scene == nullptr ||
        geom.vertex_buffer == nullptr || geom.triangle_buffer == nullptr) {
        return EmbreeArchiveStatus::INVALID_ARGUMENT;
    }
    if (bvh_payload.size() < sizeof(EmbreeStaticTriangleBVHPayloadHeader)) {
        return EmbreeArchiveStatus::CORRUPT;
    }

    EmbreeStaticTriangleBVHPayloadHeader hdr{};
    std::memcpy(&hdr, bvh_payload.data(), sizeof(hdr));
    if (hdr.magic != kEmbreePayloadMagic || hdr.format_version != kEmbreePayloadFormatVersion) {
        return EmbreeArchiveStatus::CORRUPT;
    }
    if (hdr.embree_version_number != abi.embree_version_number ||
        hdr.build_config_flags != abi.build_config_flags) {
        return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
    }
    if ((abi.cpu_isa_mask & hdr.required_isa_mask) != hdr.required_isa_mask) {
        return EmbreeArchiveStatus::INCOMPATIBLE_ISA;
    }
    if ((hdr.branching_factor != 4u && hdr.branching_factor != 8u) ||
        !is_supported_triangle_prim_and_ivariant(hdr.prim_kind, hdr.intersect_variant)) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }
    const size_t expected_node_bytes = (hdr.branching_factor == 8u) ? sizeof(AABBNode8POD) : sizeof(AABBNode4POD);
    const size_t expected_prim_bytes = prim_struct_bytes_for_kind(hdr.prim_kind);
    if (hdr.node_struct_bytes != expected_node_bytes ||
        hdr.prim_struct_bytes != expected_prim_bytes) {
        return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
    }
    if (hdr.vertex_count != geom.vertex_count ||
        hdr.triangle_count != geom.triangle_count ||
        hdr.vertex_stride_bytes != geom.vertex_stride ||
        hdr.vertex_stride_bytes < 12ull ||
        (hdr.vertex_stride_bytes % 4ull) != 0ull) {
        return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
    }

    uint64_t node_bytes_total = 0ull;
    uint64_t leaf_bytes_total = 0ull;
    uint64_t expected_payload_bytes = 0ull;
    if (!checked_mul_add(hdr.node_count, expected_node_bytes, 0ull, node_bytes_total) ||
        !checked_mul_add(hdr.leaf_block_count, expected_prim_bytes, 0ull, leaf_bytes_total) ||
        !checked_mul_add(1ull, node_bytes_total + leaf_bytes_total, sizeof(hdr), expected_payload_bytes) ||
        expected_payload_bytes != bvh_payload.size()) {
        return EmbreeArchiveStatus::CORRUPT;
    }

    InspectSceneResult dummy{};
    if (!inspect_committed_scene_bvh(abi, rtc_scene, dummy)) {
        return EmbreeArchiveStatus::UNSUPPORTED_LAYOUT;
    }
    if (dummy.branching_factor != hdr.branching_factor ||
        dummy.prim_kind != hdr.prim_kind ||
        dummy.intersect_variant != hdr.intersect_variant) {
        return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
    }
    if (hdr.prim_kind == kPrimKindTriangle4i && dummy.vertices_slot == nullptr) {
        return EmbreeArchiveStatus::INTERNAL_ERROR;
    }

    const std::byte *const src_nodes = bvh_payload.data() + sizeof(hdr);
    const std::byte *const src_leaves = src_nodes + static_cast<size_t>(node_bytes_total);

    // Parallel validate all leaf primitive blocks before allocating or modifying the RTCScene.
    std::atomic<bool> leaves_ok{true};
    parallel_for_range(static_cast<size_t>(hdr.leaf_block_count), 8192u, [&](size_t b, size_t e) noexcept {
        for (size_t i = b; i < e; ++i) {
            if (!leaves_ok.load(std::memory_order_relaxed)) {
                return;
            }
            const std::byte *lp = src_leaves + i * expected_prim_bytes;
            bool ok = false;
            if (hdr.prim_kind == kPrimKindTriangle4) {
                Triangle4POD t{};
                std::memcpy(&t, lp, sizeof(t));
                ok = validate_leaf_block_non_compact(t, hdr.triangle_count);
            } else if (hdr.prim_kind == kPrimKindTriangle4v) {
                Triangle4vPOD t{};
                std::memcpy(&t, lp, sizeof(t));
                ok = validate_leaf_block_non_compact(t, hdr.triangle_count);
            } else {
                Triangle4iPOD t{};
                std::memcpy(&t, lp, sizeof(t));
                ok = validate_leaf_block_4i(t, hdr.vertex_count, hdr.triangle_count, hdr.vertex_stride_bytes);
            }
            if (!ok) {
                leaves_ok.store(false, std::memory_order_relaxed);
                return;
            }
        }
    });
    if (!leaves_ok.load(std::memory_order_acquire)) {
        return EmbreeArchiveStatus::CORRUPT;
    }

    const size_t nodes_bytes_aligned = (static_cast<size_t>(node_bytes_total) + 63u) & (~size_t{63u});
    const size_t total_slab_bytes = std::max<size_t>(64u, nodes_bytes_aligned + static_cast<size_t>(leaf_bytes_total));
#if defined(_WIN32)
    auto *slab = static_cast<std::byte *>(_aligned_malloc(total_slab_bytes, 64u));
#else
    auto *slab = static_cast<std::byte *>(std::aligned_alloc(64u, total_slab_bytes));
#endif
    if (slab == nullptr) {
        return EmbreeArchiveStatus::INTERNAL_ERROR;
    }
    if ((reinterpret_cast<uintptr_t>(slab) & 63u) != 0u) {
#if defined(_WIN32)
        _aligned_free(slab);
#else
        std::free(slab);
#endif
        return EmbreeArchiveStatus::INTERNAL_ERROR;
    }

    std::byte *const dst_nodes = slab;
    std::byte *const dst_leaves = slab + nodes_bytes_aligned;

    if (leaf_bytes_total > 0ull) {
        constexpr size_t kLeafCopyChunk = 256ull << 10u;
        const size_t lbytes = static_cast<size_t>(leaf_bytes_total);
        const size_t chunks = (lbytes + kLeafCopyChunk - 1u) / kLeafCopyChunk;
        parallel_for_range(chunks, 4u, [&](size_t b, size_t e) noexcept {
            const size_t off = b * kLeafCopyChunk;
            const size_t len = std::min(lbytes - off, (e - b) * kLeafCopyChunk);
            std::memcpy(dst_leaves + off, src_leaves + off, len);
        });
    }

    const size_t bf = hdr.branching_factor;
    std::atomic<bool> nodes_ok{true};
    parallel_for_range(static_cast<size_t>(hdr.node_count), 2048u, [&](size_t b, size_t e) noexcept {
        for (size_t i = b; i < e; ++i) {
            if (!nodes_ok.load(std::memory_order_relaxed)) {
                return;
            }
            const std::byte *src = src_nodes + i * expected_node_bytes;
            std::byte *dst = dst_nodes + i * expected_node_bytes;
            std::memcpy(dst, src, expected_node_bytes);
            uint64_t relocs[8]{};
            uint64_t native_refs[8]{};
            std::memcpy(relocs, src, bf * sizeof(uint64_t));
            for (size_t c = 0u; c < bf; ++c) {
                if (!decode_reloc_ref(
                        relocs[c],
                        expected_node_bytes,
                        expected_prim_bytes,
                        hdr.node_count,
                        hdr.leaf_block_count,
                        dst_nodes,
                        dst_leaves,
                        native_refs[c])) {
                    nodes_ok.store(false, std::memory_order_relaxed);
                    return;
                }
            }
            std::memcpy(dst, native_refs, bf * sizeof(uint64_t));
        }
    });

    uint64_t new_root = kEmptyNode;
    const bool root_ok =
        nodes_ok.load(std::memory_order_acquire) &&
        decode_reloc_ref(
            hdr.root_encoded_ref,
            expected_node_bytes,
            expected_prim_bytes,
            hdr.node_count,
            hdr.leaf_block_count,
            dst_nodes,
            dst_leaves,
            new_root);

    if (!root_ok) {
#if defined(_WIN32)
        _aligned_free(slab);
#else
        std::free(slab);
#endif
        return EmbreeArchiveStatus::CORRUPT;
    }

    // Save dummy Embree pointers/bounds into RAII handle and swap to the external slab + rebound vertices.
    out_handle.slab = slab;
    out_handle.slab_bytes = total_slab_bytes;
    out_handle.bvh_ptr = dummy.bvh_ptr;
    out_handle.bvh_root_ofs = abi.bvh_root_ofs;
    std::memcpy(&out_handle.saved_dummy_root, static_cast<const std::byte *>(dummy.bvh_ptr) + abi.bvh_root_ofs, sizeof(uint64_t));
    out_handle.bvh_bounds_ofs = abi.accel_bounds_ofs;
    std::memcpy(out_handle.saved_bvh_bounds, static_cast<const std::byte *>(dummy.bvh_ptr) + abi.accel_bounds_ofs, 64u);
    out_handle.has_saved_bvh_bounds = true;
    out_handle.rtc_scene = rtc_scene;
    out_handle.scene_bounds_ofs = abi.accel_bounds_ofs;
    std::memcpy(out_handle.saved_scene_bounds, static_cast<const std::byte *>(rtc_scene) + abi.accel_bounds_ofs, 64u);
    out_handle.has_saved_scene_bounds = true;

    LBBox3faPOD new_lb{};
    std::memcpy(new_lb.lo0, hdr.bounds_lower, sizeof(new_lb.lo0));
    std::memcpy(new_lb.hi0, hdr.bounds_upper, sizeof(new_lb.hi0));
    std::memcpy(new_lb.lo1, hdr.bounds_lower, sizeof(new_lb.lo1));
    std::memcpy(new_lb.hi1, hdr.bounds_upper, sizeof(new_lb.hi1));

    std::memcpy(static_cast<std::byte *>(dummy.bvh_ptr) + abi.bvh_root_ofs, &new_root, sizeof(uint64_t));
    std::memcpy(static_cast<std::byte *>(dummy.bvh_ptr) + abi.accel_bounds_ofs, &new_lb, sizeof(new_lb));
    std::memcpy(static_cast<std::byte *>(rtc_scene) + abi.accel_bounds_ofs, &new_lb, sizeof(new_lb));

    if (hdr.prim_kind == kPrimKindTriangle4i) {
        out_handle.vertices_slot = dummy.vertices_slot;
        out_handle.saved_dummy_verts = *dummy.vertices_slot;
        out_handle.has_saved_verts = true;
        *dummy.vertices_slot = reinterpret_cast<float *>(const_cast<std::byte *>(geom.vertex_buffer + geom.vertex_offset));
    }

    out_unpacked_bytes = static_cast<size_t>(node_bytes_total + leaf_bytes_total);
    return EmbreeArchiveStatus::OK;
#endif
}

EmbreeArchiveStatus embree_commit_dummy_and_import_bvh_oot(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    void *rtc_geometry,
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    EmbreeImportedBVHHandle &out_handle,
    size_t &out_unpacked_bytes) noexcept {
    out_unpacked_bytes = 0u;
    out_handle.restore_and_release();
#if !LUISA_EMBREE_OOT_ENABLED
    (void)abi;
    (void)rtc_scene;
    (void)rtc_geometry;
    (void)geom;
    (void)bvh_payload;
    return EmbreeArchiveStatus::INCOMPATIBLE_CONFIG;
#else
    if (!abi.valid || rtc_scene == nullptr || rtc_geometry == nullptr || bvh_payload.empty()) {
        return EmbreeArchiveStatus::INVALID_ARGUMENT;
    }
    // Commit a minimal 1-triangle dummy mesh so stock Embree creates BVH4/BVH8 and selects
    // genuine native SIMD intersectors + vertices[0] before attaching our external slab.
    alignas(16) static constexpr float kDummyVerts[12]{
        0.f, 0.f, 0.f, 0.f,
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f};
    static constexpr uint32_t kDummyTris[3]{0u, 1u, 2u};
    auto sc = static_cast<RTCScene>(rtc_scene);
    auto gm = static_cast<RTCGeometry>(rtc_geometry);
    rtcSetGeometryTimeStepCount(gm, 1u);
    rtcSetSharedGeometryBuffer(
        gm, RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT3,
        kDummyVerts, 0u, 16u, 3u);
    rtcSetSharedGeometryBuffer(
        gm, RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT3,
        kDummyTris, 0u, 12u, 1u);
    rtcCommitGeometry(gm);
    rtcCommitScene(sc);
    return embree_import_static_triangle_bvh_oot(
        abi, rtc_scene, geom, bvh_payload, out_handle, out_unpacked_bytes);
#endif
}

EmbreeAccelArchiveExt::EmbreeAccelArchiveExt(EmbreeAccelArchiveOps ops) noexcept
    : _ops{std::move(ops)} {
    if (_ops.device_fingerprint.empty()) {
        _ops.device_fingerprint = _ops.backend_kind == AccelArchiveBackendKind::SIMD ?
                                      "embree4_simd_cpu" :
                                      "embree4_fallback_cpu";
    }

    if (const char *env = std::getenv("LUISA_EMBREE_ACCEL_ARCHIVE");
        env != nullptr && std::string_view{env} == "0") {
        _driver_fingerprint = "embree4_archive_disabled_by_env";
        return;
    }
    if (_ops.rtc_device == nullptr) {
        _driver_fingerprint = "embree4_rtc_device_null";
        return;
    }

#if LUISA_EMBREE_OOT_ENABLED
    luisa::string diag;
    if (!probe_embree_oot_abi(static_cast<RTCDevice>(_ops.rtc_device), _abi, diag)) {
        _driver_fingerprint = diag;
        LUISA_WARNING("Embree out-of-tree AccelArchive probe failed closed ({}); rebuild fallback active.", diag);
        return;
    }
    _driver_fingerprint = luisa::format(
        "embree={};cfg=0x{:08x};isa=0x{:08x};bvh4={};bvh8={};oot=1",
        _abi.embree_version_number,
        _abi.build_config_flags,
        _abi.cpu_isa_mask,
        _abi.supports_bvh4 ? 1u : 0u,
        _abi.supports_bvh8 ? 1u : 0u);
    _supported = true;
#elif !LUISA_EMBREE_OOT_X64 && !LUISA_EMBREE_OOT_ARM64
    _driver_fingerprint = "embree4_oot_architecture_unsupported";
#else
    _driver_fingerprint = "embree4_oot_headers_unavailable";
#endif
}

AccelArchiveCapabilities EmbreeAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = _ops.backend_kind;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    caps.device_fingerprint = _ops.device_fingerprint;
    caps.driver_fingerprint = _driver_fingerprint;
    caps.supported = _supported;
    caps.static_triangle_blas = _supported;
    caps.supports_compacted_export = _supported;
    caps.requires_geometry_rebind = true;
    caps.import_is_async_on_stream = false;
    caps.supports_pinned_staging = false;
    caps.supports_direct_file_import = _supported;
    caps.layout_info = _supported ?
                           "embree4_out_of_tree_bvh4_bvh8_static_tri_v1" :
                           "embree4_out_of_tree_static_triangle_bvh_unavailable";
    return caps;
}

EmbreeAccelArchiveExt::~EmbreeAccelArchiveExt() noexcept {
    std::lock_guard lck{_staging_mtx};
    for (auto &slot : _staging_slots) {
        if (slot.ptr != nullptr) {
#if defined(_WIN32)
            _aligned_free(slot.ptr);
#else
            std::free(slot.ptr);
#endif
            slot.ptr = nullptr;
        }
    }
    _staging_slots.clear();
}

AccelArchiveStagingBuffer EmbreeAccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    constexpr size_t kAlign = 65536u;
    const size_t wanted = std::max<size_t>(kAlign, ((min_bytes + kAlign - 1u) / kAlign) * kAlign);
    std::lock_guard lck{_staging_mtx};
    size_t best = _staging_slots.size();
    for (size_t i = 0u; i < _staging_slots.size(); ++i) {
        auto const &s = _staging_slots[i];
        if (!s.in_use && s.capacity >= wanted) {
            if (best == _staging_slots.size() || s.capacity < _staging_slots[best].capacity) {
                best = i;
            }
        }
    }
    if (best < _staging_slots.size()) {
        auto &s = _staging_slots[best];
        s.in_use = true;
        s.lru_tick = ++_staging_tick;
        AccelArchiveStagingBuffer out{};
        out.handle = reinterpret_cast<uint64_t>(s.ptr);
        out.mapped_data = s.ptr;
        out.capacity = s.capacity;
        out.alignment = kAlign;
        out.is_device_local = false;
        out.is_pinned_host = false;
        return out;
    }

    constexpr size_t kMaxSlots = 8u;
    if (_staging_slots.size() >= kMaxSlots) {
        size_t victim = _staging_slots.size();
        for (size_t i = 0u; i < _staging_slots.size(); ++i) {
            if (!_staging_slots[i].in_use) {
                if (victim == _staging_slots.size() || _staging_slots[i].lru_tick < _staging_slots[victim].lru_tick) {
                    victim = i;
                }
            }
        }
        if (victim < _staging_slots.size()) {
#if defined(_WIN32)
            _aligned_free(_staging_slots[victim].ptr);
#else
            std::free(_staging_slots[victim].ptr);
#endif
            if (victim + 1u < _staging_slots.size()) {
                _staging_slots[victim] = _staging_slots.back();
            }
            _staging_slots.pop_back();
        }
    }

#if defined(_WIN32)
    auto *ptr = static_cast<std::byte *>(_aligned_malloc(wanted, kAlign));
#else
    auto *ptr = static_cast<std::byte *>(std::aligned_alloc(kAlign, wanted));
#endif
    if (ptr == nullptr) {
        return {};
    }
    StagingSlot s{};
    s.ptr = ptr;
    s.capacity = wanted;
    s.in_use = true;
    s.lru_tick = ++_staging_tick;
    _staging_slots.push_back(s);

    AccelArchiveStagingBuffer out{};
    out.handle = reinterpret_cast<uint64_t>(ptr);
    out.mapped_data = ptr;
    out.capacity = wanted;
    out.alignment = kAlign;
    out.is_device_local = false;
    out.is_pinned_host = false;
    return out;
}

void EmbreeAccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (buffer.handle != invalid_resource_handle && buffer.handle != 0u) {
        std::lock_guard lck{_staging_mtx};
        for (auto &s : _staging_slots) {
            if (reinterpret_cast<uint64_t>(s.ptr) == buffer.handle ||
                (buffer.mapped_data != nullptr && s.ptr == buffer.mapped_data)) {
                s.in_use = false;
                s.lru_tick = ++_staging_tick;
                buffer = {};
                return;
            }
        }
    }
    AccelArchiveExt::release_staging_buffer(std::move(buffer));
}

AccelArchiveResult EmbreeAccelArchiveExt::_check_compat_and_header(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t meta_unpacked_size,
    uint64_t &out_unpacked_bytes) const noexcept {
    if (compat_blob.size() != sizeof(CompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree AS archive compat blob has unexpected size.");
    }
    CompatBlob compat{};
    std::memcpy(&compat, compat_blob.data(), sizeof(CompatBlob));
    if (compat.magic != kEmbreeCompatMagic ||
        compat.version != kEmbreeCompatVersion ||
        compat.payload_magic != kEmbreePayloadMagic) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree AS archive compat blob magic or version mismatch.");
    }
    if (compat.embree_version_number != _abi.embree_version_number ||
        compat.build_config_flags != _abi.build_config_flags) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            luisa::format(
                "Embree version/config mismatch (archive ver={} cfg=0x{:08x}, active ver={} cfg=0x{:08x}).",
                compat.embree_version_number,
                compat.build_config_flags,
                _abi.embree_version_number,
                _abi.build_config_flags));
    }
    if ((_abi.cpu_isa_mask & compat.enabled_cpu_features) != compat.enabled_cpu_features) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            "Embree AS archive requires CPU ISA features not enabled on active device.");
    }
    if (compat.unpacked_as_bytes != meta_unpacked_size) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree compat unpacked_as_bytes disagrees with envelope unpacked_as_size.");
    }
    if (payload.size() < sizeof(EmbreeStaticTriangleBVHPayloadHeader) || payload.size() > kMaxEmbreePayloadBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Embree BVH payload has invalid byte size.");
    }
    uint32_t payload_magic = 0u;
    std::memcpy(&payload_magic, payload.data(), sizeof(uint32_t));
    if (payload_magic != kEmbreePayloadMagic) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Embree BVH payload header magic mismatch.");
    }
    out_unpacked_bytes = compat.unpacked_as_bytes;
    return {};
}

AccelArchiveResult EmbreeAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree out-of-tree static triangle BVH archive is unavailable or disabled.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        _ops.backend_kind,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t unpacked_bytes = 0u;
    auto check_res = _check_compat_and_header(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        unpacked_bytes);
    if (!check_res) {
        return check_res;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = unpacked_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult EmbreeAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree out-of-tree static triangle BVH archive is unavailable or disabled.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }
    if (_ops.validate_stream && !_ops.validate_stream(_ops.user_ctx, stream_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle must be a valid stream on the active device.");
    }
    if (mesh_handle == invalid_resource_handle || mesh_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh handle must be valid.");
    }

    // Drain queued CPU stream builds before inspecting and serializing Embree's BVH.
    _ops.synchronize_stream(_ops.user_ctx, stream_handle);

    EmbreeMeshExportState state{};
    auto mesh_res = _ops.get_mesh_export_state(_ops.user_ctx, mesh_handle, state);
    if (!mesh_res) {
        return mesh_res;
    }
    if (!options_match(state.option, desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Mesh AccelOption does not match AccelArchiveMeshDesc::option.");
    }
    if (!state.is_built || state.committed_rtc_scene == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "Mesh has not been built yet.");
    }
    if (state.triangle_count != desc.triangle_count ||
        (state.vertex_count != 0u && state.vertex_count != desc.vertex_count) ||
        (state.vertex_stride != 0u && state.vertex_stride != desc.vertex_stride)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "Mesh geometry counts (tris={}, verts={}, stride={}) do not match expected ({}, {}, {}).",
                state.triangle_count,
                state.vertex_count,
                state.vertex_stride,
                desc.triangle_count,
                desc.vertex_count,
                desc.vertex_stride));
    }

#if !LUISA_EMBREE_OOT_ENABLED
    return accel_archive::make_archive_error(
        AccelArchiveStatus::UNSUPPORTED_FEATURE,
        "Embree 4 out-of-tree archiving disabled at compile time.");
#else
    ExportPlan plan{};
    const auto plan_st = build_export_plan_oot(_abi, state, plan);
    if (plan_st != EmbreeArchiveStatus::OK) {
        return map_embree_status(plan_st, "embree_export_static_triangle_bvh_oot");
    }
    const size_t required_bytes = plan.payload_bytes();
    const uint64_t unpacked_bytes = plan.unpacked_as_bytes();
    if (required_bytes < sizeof(EmbreeStaticTriangleBVHPayloadHeader) || required_bytes > kMaxEmbreePayloadBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Embree serialized BVH size is outside valid bounds.");
    }

    auto payload_staging = acquire_staging_buffer(required_bytes);
    if (payload_staging.mapped_data == nullptr || payload_staging.capacity < required_bytes) {
        release_staging_buffer(std::move(payload_staging));
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to allocate Embree export staging buffer.");
    }
    auto release_payload_staging = vstd::scope_exit([&]() noexcept {
        release_staging_buffer(std::move(payload_staging));
    });

    write_export_payload_oot(_abi, plan, payload_staging.mapped_data);
    const luisa::span<const std::byte> payload{payload_staging.mapped_data, required_bytes};

    CompatBlob compat{};
    compat.magic = kEmbreeCompatMagic;
    compat.version = kEmbreeCompatVersion;
    compat.embree_version_number = _abi.embree_version_number;
    compat.build_config_flags = _abi.build_config_flags;
    compat.enabled_cpu_features = plan.inspected.required_isa_mask;
    compat.payload_magic = kEmbreePayloadMagic;
    compat.unpacked_as_bytes = unpacked_bytes;

    const luisa::span<const std::byte> compat_span{
        reinterpret_cast<const std::byte *>(&compat),
        sizeof(CompatBlob)};
    uint64_t checked_unpacked = 0u;
    if (auto self_check = _check_compat_and_header(
            compat_span, payload, compat.unpacked_as_bytes, checked_unpacked);
        !self_check) {
        return self_check;
    }

    const bool is_compacted = desc.option.allow_compaction;
    auto encode_res = accel_archive::encode_envelope(
        _ops.backend_kind,
        desc,
        is_compacted,
        compat.unpacked_as_bytes,
        compat_span,
        payload,
        out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = compat.unpacked_as_bytes;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
#endif
}

AccelArchiveResult EmbreeAccelArchiveExt::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveImportOptions default_opts{};
    return import_mesh_archive_with_options(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        archive_bytes,
        default_opts);
}

AccelArchiveResult EmbreeAccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &options) noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree out-of-tree static triangle BVH archive is unavailable or disabled.");
    }
    if (_ops.validate_stream && !_ops.validate_stream(_ops.user_ctx, stream_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle must be a valid stream on the active device.");
    }
    if (mesh_handle == invalid_resource_handle || mesh_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh handle must be valid.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    uint8_t expected_payload_checksum[16]{};
    auto env_res = accel_archive::decode_envelope_header_and_compat(
        archive_bytes,
        _ops.backend_kind,
        expected_desc,
        view,
        expected_payload_checksum);
    if (!env_res) {
        return env_res;
    }

    uint64_t header_unpacked = 0u;
    if (auto check_res = _check_compat_and_header(
            view.compat_blob,
            view.payload,
            view.meta.unpacked_as_size,
            header_unpacked);
        !check_res) {
        return check_res;
    }

    if (options.verify_payload_checksum) {
        if (auto p = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum); !p) {
            return p;
        }
    }

    EmbreeResolvedTriangleGeometry resolved_geom{};
    auto geom_res = _ops.resolve_geometry_binding(
        _ops.user_ctx,
        mesh_handle,
        expected_desc,
        geometry_binding,
        resolved_geom);
    if (!geom_res) {
        return geom_res;
    }

    // Drain any prior stream work before rebinding buffers and populating the mesh's RTCScene.
    _ops.synchronize_stream(_ops.user_ctx, stream_handle);

    size_t unpacked_bytes = 0u;
    EmbreeArchiveStatus embree_st = EmbreeArchiveStatus::OK;
    auto import_res = _ops.import_mesh_bvh(
        _ops.user_ctx,
        mesh_handle,
        resolved_geom,
        view.payload,
        _abi,
        unpacked_bytes,
        embree_st);
    if (!import_res) {
        return import_res;
    }
    if (embree_st != EmbreeArchiveStatus::OK) {
        return map_embree_status(embree_st, "embree_import_static_triangle_bvh_oot");
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    res.unpacked_as_size = unpacked_bytes != 0u ? unpacked_bytes : header_unpacked;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "imported_immediately";
    return res;
}

}// namespace luisa::compute
