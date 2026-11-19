// Test for the shared, reference-counted lifetime of `Resource` and its subclasses.
// This test covers:
// - compile-time proof that every `Resource` subclass is copy/move constructible
//   and copy/move assignable
// - the "zero overhead" single-owner path: a resource that is never copied keeps
//   `rc == nullptr`, so it allocates no `ResourceRefCount`
// - copy construction / copy assignment sharing one underlying API object (same
//   handle, same uid, `reference_count()` grows)
// - move construction / move assignment transferring the ownership instead of
//   sharing it
// - the object staying alive and usable while at least one copy exists, and being
//   torn down exactly once by the last live owner
// - `release()` detaching one instance from the shared block
// - self assignment, invalid-resource assignment, and re-assignment over a live
//   resource

#include "ut/ut.hpp"
#include "test_device.h"

#include <barrier>
#include <thread>

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/sparse_buffer.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/sparse_volume.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>
#include <luisa/runtime/rtx/aabb.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/runtime/rtx/procedural_primitive.h>
#include <luisa/runtime/rtx/motion_instance.h>
#include <luisa/runtime/rtx/opacity_micromap.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

/// Number of times the observable teardown hook below has run.
int g_destroy_count = 0;

/// A minimal `Resource` subclass with an observable `destroy()`, so the test can
/// prove that the last live owner tears the object down exactly once.
class TrackedResource final : public Resource {

private:
    size_t _size_bytes{0u};

public:
    TrackedResource() noexcept = default;
    explicit TrackedResource(DeviceInterface *device,
                             const BufferCreationInfo &info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _size_bytes{info.total_size_bytes} {}

    ~TrackedResource() noexcept override { _release(); }
    TrackedResource(TrackedResource &&) noexcept = default;
    TrackedResource(TrackedResource const &) noexcept = default;
    TrackedResource &operator=(TrackedResource &&) noexcept = default;
    TrackedResource &operator=(TrackedResource const &) noexcept = default;

    using Resource::operator bool;
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }

protected:
    void destroy() noexcept override {
        if (*this) {
            ++g_destroy_count;
            device()->destroy_buffer(handle());
        }
    }
};

/// Two instances that share one ownership block point at the same API object.
template<typename R>
void expect_same_object(const R &lhs, const R &rhs) {
    expect(static_cast<bool>(lhs)) << "lhs must be valid";
    expect(static_cast<bool>(rhs)) << "rhs must be valid";
    expect(lhs.handle() == rhs.handle()) << "copies must share the device handle";
    expect(lhs.uid() == rhs.uid()) << "copies must share the resource uid";
    expect(lhs.tag() == rhs.tag()) << "copies must share the resource tag";
    expect(lhs.device() == rhs.device()) << "copies must share the device";
}

/// Every `Resource` subclass is now copyable and movable, and keeps the old
/// move-only invariants only where the payload itself cannot be copied. This is a
/// compile-time check, so it also covers the types that need a backend feature
/// (sparse resources, depth buffers, raster shaders, swapchains, dstorage files).
void check_all_subclasses_are_shared_handles() {
    using namespace luisa::compute;
#define LUISA_TEST_SHARED_HANDLE(...)                          \
    static_assert(std::is_copy_constructible_v<__VA_ARGS__>,   \
                  #__VA_ARGS__ " must be copy constructible"); \
    static_assert(std::is_copy_assignable_v<__VA_ARGS__>,      \
                  #__VA_ARGS__ " must be copy assignable");    \
    static_assert(std::is_move_constructible_v<__VA_ARGS__>,   \
                  #__VA_ARGS__ " must be move constructible"); \
    static_assert(std::is_move_assignable_v<__VA_ARGS__>,      \
                  #__VA_ARGS__ " must be move assignable")

    LUISA_TEST_SHARED_HANDLE(Buffer<float>);
    LUISA_TEST_SHARED_HANDLE(ByteBuffer);
    LUISA_TEST_SHARED_HANDLE(Image<float>);
    LUISA_TEST_SHARED_HANDLE(Volume<float>);
    LUISA_TEST_SHARED_HANDLE(Event);
    LUISA_TEST_SHARED_HANDLE(TimelineEvent);
    LUISA_TEST_SHARED_HANDLE(IndirectDispatchBuffer);
    LUISA_TEST_SHARED_HANDLE(BindlessArray);
    LUISA_TEST_SHARED_HANDLE(Stream);
    LUISA_TEST_SHARED_HANDLE(Swapchain);
    LUISA_TEST_SHARED_HANDLE(ShaderBase);
    LUISA_TEST_SHARED_HANDLE(Shader<1, Buffer<float>>);
    LUISA_TEST_SHARED_HANDLE(DepthBuffer);
    LUISA_TEST_SHARED_HANDLE(RasterShader<int>);
    LUISA_TEST_SHARED_HANDLE(SparseBuffer<float>);
    LUISA_TEST_SHARED_HANDLE(SparseTexture);
    LUISA_TEST_SHARED_HANDLE(SparseImage<float>);
    LUISA_TEST_SHARED_HANDLE(SparseVolume<float>);
    LUISA_TEST_SHARED_HANDLE(SparseBufferHeap);
    LUISA_TEST_SHARED_HANDLE(SparseTextureHeap);
    LUISA_TEST_SHARED_HANDLE(Accel);
    LUISA_TEST_SHARED_HANDLE(Mesh);
    LUISA_TEST_SHARED_HANDLE(Curve);
    LUISA_TEST_SHARED_HANDLE(ProceduralPrimitive);
    LUISA_TEST_SHARED_HANDLE(MotionInstance);
    LUISA_TEST_SHARED_HANDLE(OpacityMicromap);
    LUISA_TEST_SHARED_HANDLE(DStorageFile);
#undef LUISA_TEST_SHARED_HANDLE

    // The base-class assignment stays protected. `Tag` is not a type identity --
    // `Buffer`/`ByteBuffer`/`IndirectDispatchBuffer` all use Tag::BUFFER,
    // `Image`/`Volume` both use Tag::TEXTURE, and the sparse image/volume derive
    // from the sparse texture -- so a `Resource &` assignment would be a silent
    // type confusion that the tag check cannot catch. Assigning the concrete
    // handles (above) is still fine, because their own operators are type-exact.
    static_assert(!std::is_assignable_v<Resource &, const Buffer<float> &>);
    static_assert(!std::is_assignable_v<Resource &, Buffer<float> &&>);
    static_assert(!std::is_assignable_v<Resource &, const Image<float> &>);
    static_assert(!std::is_assignable_v<Resource &, const Volume<float> &>);
    static_assert(!std::is_assignable_v<Resource &, const ByteBuffer &>);
    static_assert(!std::is_assignable_v<Resource &, const IndirectDispatchBuffer &>);
}

void test_buffer_lifetime(Device &device) {
    Stream stream = device.create_stream();
    auto buffer = device.create_buffer<float>(16u);
    expect(buffer.reference_count() == 1)
        << "a resource that was never copied must stay a single owner (no block)";

    // --- copy construction ---------------------------------------------------
    auto copy = buffer;
    expect_same_object(buffer, copy);
    expect(buffer.reference_count() == 2);
    expect(copy.reference_count() == 2);

    // --- copy assignment ----------------------------------------------------
    Buffer<float> assigned;
    expect(!assigned);
    expect(assigned.reference_count() == 1)
        << "an invalid resource never allocates a reference-count block";
    assigned = buffer;
    expect_same_object(buffer, assigned);
    expect(buffer.reference_count() == 3);

    // --- move construction hands the ownership over -------------------------
    auto moved = std::move(copy);
    expect(!copy) << "a moved-from resource must be invalid";
    expect(copy.reference_count() == 1) << "a moved-from resource owns nothing";
    expect_same_object(buffer, moved);
    expect(buffer.reference_count() == 3) << "moving must not change the count";

    // --- move assignment ----------------------------------------------------
    Buffer<float> moved_into;
    moved_into = std::move(moved);
    expect(!moved) << "a moved-from resource must be invalid";
    expect_same_object(buffer, moved_into);
    expect(buffer.reference_count() == 3) << "moving must not change the count";

    // --- self assignment is a no-op -----------------------------------------
    {
        auto &self = buffer;
        buffer = self;
    }
    expect(buffer.reference_count() == 3) << "self copy assignment must not leak";
    {
        auto &self = buffer;
        buffer = std::move(self);
    }
    expect(static_cast<bool>(buffer)) << "self move assignment must not destroy";
    expect(buffer.reference_count() == 3) << "self move assignment must not leak";

    // --- the object survives as long as one copy is alive -------------------
    luisa::vector<float> host(16u);
    for (auto i = 0u; i < host.size(); i++) { host[i] = 10.0f + static_cast<float>(i); }
    stream << buffer.copy_from(luisa::span{host}) << synchronize();
    {
        auto transient = buffer;
        expect(transient.reference_count() == 4);
    }
    expect(buffer.reference_count() == 3) << "a dead copy must drop its reference";
    {
        luisa::vector<float> readback(16u, -1.0f);
        stream << moved_into.copy_to(luisa::span{readback}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < host.size(); i++) {
            ok &= std::abs(readback[i] - host[i]) < 1e-6f;
        }
        expect(ok) << "a surviving copy must keep the underlying buffer intact";
    }

    // --- writing through one copy is visible through another ----------------
    {
        Kernel1D write_constant = [](BufferFloat b, Float value) noexcept {
            b.write(dispatch_id().x, value);
        };
        auto shader = device.compile(write_constant);
        stream << shader(buffer, 42.0f).dispatch(host.size()) << synchronize();
        luisa::vector<float> readback(16u, -1.0f);
        stream << assigned.copy_to(luisa::span{readback}) << synchronize();
        auto ok = true;
        for (auto v : readback) { ok &= std::abs(v - 42.0f) < 1e-6f; }
        expect(ok) << "copies must alias one object, not clone it";
    }

    // --- re-assignment over a live resource releases the old one -----------
    {
        auto other = device.create_buffer<float>(16u);
        expect(other.reference_count() == 1);
        other = buffer;
        expect_same_object(buffer, other);
        expect(buffer.reference_count() == 4)
            << "assigning over a live resource must drop its old reference";
    }
    expect(buffer.reference_count() == 3);

    // --- release() hands the ownership over, so it needs a sole owner -----
    {
        auto sole = device.create_buffer<float>(16u);
        expect(sole.reference_count() == 1);
        auto info = sole.release();
        expect(info.valid());
        expect(!sole) << "a released resource must be invalid";
        expect(sole.reference_count() == 1);
        // give the ownership of the released handle to somebody that frees it
        TrackedResource reclaim{device.impl(), info};
        expect(reclaim.size_bytes() == 16u * sizeof(float));
        expect(reclaim.reference_count() == 1);
    }
    expect(buffer.reference_count() == 3) << "release() must not touch other resources";
}

void test_destroy_exactly_once(Device &device) {
    auto info = device.create_buffer<float>(64u).release();
    expect(info.valid()) << "a released buffer must hand over a valid creation info";
    g_destroy_count = 0;
    {
        TrackedResource first{device.impl(), info};
        expect(static_cast<bool>(first));
        expect(first.size_bytes() == 64u * sizeof(float));
        expect(first.reference_count() == 1);
        {
            auto second = first;
            expect(first.reference_count() == 2);
            {
                TrackedResource third;
                third = first;
                expect(first.reference_count() == 3);
            }
            expect(g_destroy_count == 0) << "a dead copy must not destroy the object";
            expect(first.reference_count() == 2);
        }
        expect(g_destroy_count == 0) << "the object must outlive its dead copies";
        expect(first.reference_count() == 1);
    }
    expect(g_destroy_count == 1)
        << "the last live owner must destroy the object once, got "
        << g_destroy_count;
}

void test_invalid_resource_has_no_block(Device &device) {
    static_cast<void>(device);
    Buffer<float> empty;
    expect(!empty);
    auto copy = empty;
    expect(!copy) << "copying an invalid resource stays invalid";
    expect(empty.reference_count() == 1);
    expect(copy.reference_count() == 1);
    expect(empty.uid() == copy.uid()) << "even an invalid copy keeps the same uid";
    Image<float> image;
    expect(!image);
    image = Image<float>{};
    expect(!image);
}

void test_typed_resources(Device &device) {
    Stream stream = device.create_stream();

    // ByteBuffer
    {
        auto bytes = device.create_byte_buffer(64u);
        auto copy = bytes;
        expect_same_object(bytes, copy);
        expect(bytes.reference_count() == 2);
        expect(copy.view().size_bytes() == 64u);
    }

    // Image
    {
        auto image = device.create_image<float>(PixelStorage::FLOAT4, 4u, 4u);
        auto copy = image;
        expect_same_object(image, copy);
        expect(copy.size().x == 4u && copy.size().y == 4u);
        expect(copy.storage() == PixelStorage::FLOAT4);
        expect(image.reference_count() == 2);
        Image<float> assigned;
        assigned = image;
        expect(assigned.reference_count() == 3);
    }

    // Volume
    {
        auto volume = device.create_volume<float>(PixelStorage::FLOAT1, 4u, 4u, 4u);
        auto copy = volume;
        expect_same_object(volume, copy);
        expect(copy.size().x == 4u && copy.size().y == 4u && copy.size().z == 4u);
        expect(volume.reference_count() == 2);
    }

    // Event / TimelineEvent
    {
        auto event = device.create_event();
        auto event_copy = event;
        expect_same_object(event, event_copy);
        expect(event.reference_count() == 2);
        stream << event_copy.signal() << event.wait() << synchronize();

        auto timeline = device.create_timeline_event();
        auto timeline_copy = timeline;
        expect_same_object(timeline, timeline_copy);
        expect(timeline.reference_count() == 2);
        stream << timeline_copy.signal(1u) << timeline.wait(1u) << synchronize();
    }

    // IndirectDispatchBuffer
    {
        auto dispatch_buffer = device.create_indirect_dispatch_buffer(4u);
        auto copy = dispatch_buffer;
        expect_same_object(dispatch_buffer, copy);
        expect(copy.capacity() == 4u);
        expect(dispatch_buffer.reference_count() == 2);
    }

    // BindlessArray
    {
        auto bindless = device.create_bindless_array(16u);
        auto copy = bindless;
        expect_same_object(bindless, copy);
        expect(copy.size() == 16u);
        expect(bindless.reference_count() == 2);
    }

    // Shader (compiled kernel)
    {
        Kernel1D trivial = [](BufferFloat b) noexcept {
            b.write(dispatch_id().x, 1.0f);
        };
        auto shader = device.compile(trivial);
        expect(shader.reference_count() == 1);
        auto copy = shader;
        expect_same_object(shader, copy);
        expect(shader.reference_count() == 2);
        auto buffer = device.create_buffer<float>(4u);
        stream << copy(buffer).dispatch(4u) << synchronize();
        luisa::vector<float> readback(4u, 0.0f);
        stream << buffer.copy_to(luisa::span{readback}) << synchronize();
        auto ok = true;
        for (auto v : readback) { ok &= std::abs(v - 1.0f) < 1e-6f; }
        expect(ok) << "a copied shader must still dispatch";
    }
}

void test_accel_geometry(Device &device) {
    Stream stream = device.create_stream();

    luisa::vector<float3> vertices{{0.0f, 0.0f, 0.0f},
                                   {1.0f, 0.0f, 0.0f},
                                   {0.0f, 1.0f, 0.0f}};
    luisa::vector<Triangle> triangles{Triangle{0u, 1u, 2u}};
    luisa::vector<float4> control_points{{0.0f, 0.0f, 0.0f, 1.0f},
                                         {1.0f, 0.0f, 0.0f, 1.0f},
                                         {2.0f, 1.0f, 0.0f, 1.0f},
                                         {3.0f, 1.0f, 0.0f, 1.0f}};
    luisa::vector<uint> segments{0u, 1u};
    luisa::vector<AABB> aabbs{AABB{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}};

    auto v_buffer = device.create_buffer<float3>(vertices.size());
    auto t_buffer = device.create_buffer<Triangle>(triangles.size());
    auto cp_buffer = device.create_buffer<float4>(control_points.size());
    auto seg_buffer = device.create_buffer<uint>(segments.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    stream << v_buffer.copy_from(luisa::span{vertices})
           << t_buffer.copy_from(luisa::span{triangles})
           << cp_buffer.copy_from(luisa::span{control_points})
           << seg_buffer.copy_from(luisa::span{segments})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << synchronize();

    // Mesh
    auto mesh = device.create_mesh(v_buffer, t_buffer);
    auto mesh_copy = mesh;
    expect_same_object(mesh, mesh_copy);
    expect(mesh.reference_count() == 2);
    expect(mesh_copy.triangle_count() == 1u);
    // the BLAS must be built before a TLAS can reference it
    stream << mesh_copy.build() << synchronize();

    // Curve: not every backend implements it (dx and vk do not), and an unsupported
    // one fails loudly instead of handing out an invalid resource, so skip it there.
    // `MotionInstance` is in the same boat (only implemented by a subset of the
    // backends as well), so its shared lifetime is only covered at compile time.
    auto backend = device.backend_name();
    if (backend != "dx" && backend != "vk") {
        auto curve = device.create_curve(CurveBasis::CUBIC_BSPLINE, cp_buffer, seg_buffer);
        auto curve_copy = curve;
        expect_same_object(curve, curve_copy);
        expect(curve.reference_count() == 2);
        expect(curve_copy.segment_count() == segments.size());
    }

    // ProceduralPrimitive
    auto primitive = device.create_procedural_primitive(aabb_buffer);
    auto primitive_copy = primitive;
    expect_same_object(primitive, primitive_copy);
    expect(primitive.reference_count() == 2);
    expect(primitive_copy.aabb_buffer().size() == aabbs.size());

    // Accel: the copies must share the very same acceleration structure
    auto accel = device.create_accel();
    auto accel_copy = accel;
    expect_same_object(accel, accel_copy);
    expect(accel.reference_count() == 2);
    // NOTE: the copies share the device-side acceleration structure, but not the
    // frontend bookkeeping (the pending-modification list and its instance count),
    // so a mutation has to be recorded and committed through one instance.
    accel_copy.emplace_back(mesh_copy);
    expect(accel_copy.size() == 1u);
    expect(accel.size() == 0u) << "frontend bookkeeping is per instance";
    expect(accel_copy.dirty());
    stream << accel_copy.build() << synchronize();
    expect(!accel_copy.dirty());
}

void test_sparse_resources(Device &device) {
    // Sparse resources are only available on some backends, and some backends only
    // support a subset of the family (for example CUDA has sparse buffers but no
    // sparse textures). Unsupported ones are reported as an invalid resource instead
    // of failing loudly, so probe each one and only check what the backend gave us.
    auto heap = device.allocate_sparse_buffer_heap(64_k);
    if (!heap) {
        LUISA_INFO("Skipping the sparse resource lifetime check on backend '{}'.",
                   device.backend_name());
        return;
    }
    {
        auto copy = heap;
        expect_same_object(heap, copy);
        expect(heap.reference_count() == 2);
    }
    if (auto texture_heap = device.allocate_sparse_texture_heap(64_k)) {
        auto copy = texture_heap;
        expect_same_object(texture_heap, copy);
        expect(texture_heap.reference_count() == 2);
    }
    if (auto sparse_buffer = device.create_sparse_buffer<float>(64u)) {
        auto copy = sparse_buffer;
        expect_same_object(sparse_buffer, copy);
        expect(sparse_buffer.reference_count() == 2);
        // a backend may round the allocation up to a tile boundary, so compare the
        // copy against the source rather than against the requested element count
        expect(copy.size() == sparse_buffer.size());
        expect(copy.size() >= 64u);
    }
    // A sparse level can only be tiled when it is at least one tile (64 KiB) large; a
    // smaller base level would start in the "mip tail", which the frontend cannot
    // express. So use a base level of a few tiles and a single mip level.
    if (auto sparse_image = device.create_sparse_image<float>(PixelStorage::FLOAT4, 128u, 128u)) {
        auto copy = sparse_image;
        expect_same_object(sparse_image, copy);
        expect(sparse_image.reference_count() == 2);
        expect(copy.size().x == 128u && copy.size().y == 128u);
    }
    if (auto sparse_volume = device.create_sparse_volume<float>(PixelStorage::FLOAT1, 32u, 32u, 32u)) {
        auto copy = sparse_volume;
        expect_same_object(sparse_volume, copy);
        expect(sparse_volume.reference_count() == 2);
        expect(copy.size().x == 32u && copy.size().y == 32u && copy.size().z == 32u);
    }
}

}// namespace


// Copies of one handle may be made from several threads at once. Promoting a
// single-owner resource to shared ownership therefore has to pick a single winner:
// two blocks would leak one of them, and a lost promotion would free a block that
// another thread is still counting on.
void test_concurrent_copies(Device &device) {
    constexpr auto thread_count = 8u;
    constexpr auto rounds = 32u;

    luisa::vector<Buffer<float>> sources;
    sources.reserve(rounds);
    for (auto i = 0u; i < rounds; i++) {
        sources.emplace_back(device.create_buffer<float>(16u));
        expect(sources.back().reference_count() == 1)
            << "a fresh resource must not own a reference-count block yet";
    }

    // Two rendezvous per round, so every worker copies the *same* single-owner handle
    // at the same moment, and no copy outlives its round.
    std::barrier before{thread_count};
    std::barrier after{thread_count};
    std::atomic<uint64_t> copies{0u};
    std::atomic<size_t> max_seen{0u};
    std::atomic<uint32_t> out_of_range{0u};

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto t = 0u; t < thread_count; t++) {
        threads.emplace_back([&] {
            for (auto r = 0u; r < rounds; r++) {
                before.arrive_and_wait();
                {
                    auto copy = sources[r];
                    auto count = sources[r].reference_count();
                    if (count < 2u || count > thread_count + 1u) {
                        out_of_range.fetch_add(1u, std::memory_order_relaxed);
                    }
                    auto previous = max_seen.load();
                    while (previous < count &&
                           !max_seen.compare_exchange_weak(previous, count)) {}
                    copies.fetch_add(1u, std::memory_order_relaxed);
                }
                after.arrive_and_wait();
            }
        });
    }
    for (auto &thread : threads) { thread.join(); }

    expect(copies.load() == static_cast<uint64_t>(thread_count) * rounds);
    expect(out_of_range.load() == 0u)
        << "reference count out of range during a concurrent promotion";
    expect(max_seen.load() >= 2u && max_seen.load() <= thread_count + 1u);
    for (auto i = 0u; i < rounds; i++) {
        expect(sources[i].reference_count() == 1)
            << "the block must be back to a single owner once every copy is gone";
        expect(static_cast<bool>(sources[i]))
            << "a concurrently copied handle must stay alive";
    }

    // The object must still be usable: a lost or premature teardown would show up here.
    Stream stream = device.create_stream();
    luisa::vector<float> host(16u, 3.0f);
    stream << sources[0].copy_from(luisa::span{host}) << synchronize();
}


// Retiring a handle through a completion callback: the device object must survive the
// commands that still reference it and be released only once the GPU is done, without the
// submitting thread calling `synchronize()`.
void test_deferred_destroy(Device &device) {
    Stream stream = device.create_stream();
    Event event = device.create_event();

    std::atomic<uint32_t> fired{0u};
    std::atomic<bool> alive_in_callback{false};

    CommandList list = CommandList::create(1u, 1u);
    {
        auto buffer = device.create_buffer<float>(16u);
        luisa::vector<float> host(16u, 5.0f);
        list << buffer.copy_from(luisa::span{host});
        list.add_callback([buffer = std::move(buffer), &fired, &alive_in_callback]() mutable {
            alive_in_callback.store(static_cast<bool>(buffer), std::memory_order_relaxed);
            fired.fetch_add(1u, std::memory_order_release);
            buffer = {};// released here, on the backend's callback thread
        });
        expect(!buffer) << "the callback must own the only handle";
    }
    stream << list.commit() << event.signal();
    event.synchronize();// the stream drained, so the callback has run

    expect(fired.load(std::memory_order_acquire) == 1u)
        << "the completion callback must run exactly once";
    expect(alive_in_callback.load(std::memory_order_relaxed))
        << "the handle must still be valid inside the callback";

    // The same shape without an explicit CommandList: a plain chained callback.
    std::atomic<uint32_t> chained{0u};
    {
        auto another = device.create_buffer<float>(16u);
        luisa::vector<float> host(16u, 1.0f);
        stream << another.copy_from(luisa::span{host})
               << [b = std::move(another), &chained]() mutable {
                      chained.fetch_add(1u, std::memory_order_release);
                      b = {};
                  };
        expect(!another) << "the chained callback must own the only handle";
    }
    stream << event.signal();
    event.synchronize();
    expect(chained.load(std::memory_order_acquire) == 1u)
        << "a chained `stream << lambda` must run as a completion callback";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;

    check_all_subclasses_are_shared_handles();

    "buffer_shared_lifetime"_test = [&] { test_buffer_lifetime(device); };
    "destroy_exactly_once"_test = [&] { test_destroy_exactly_once(device); };
    "invalid_resource_has_no_block"_test = [&] { test_invalid_resource_has_no_block(device); };
    "typed_resources"_test = [&] { test_typed_resources(device); };
    "accel_geometry"_test = [&] { test_accel_geometry(device); };
    "sparse_resources"_test = [&] { test_sparse_resources(device); };
    "concurrent_copies"_test = [&] { test_concurrent_copies(device); };
    "deferred_destroy"_test = [&] { test_deferred_destroy(device); };
    return 0;
}
