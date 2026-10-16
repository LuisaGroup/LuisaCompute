// Batched (multiple) dispatch test.
//
// A single ShaderDispatchCommand can carry several dispatch sizes
// (see ShaderInvoke::dispatch(luisa::span<const uint3>)). Backends must
// launch one grid per element: the device-side dispatch_size() must equal
// the corresponding span element, kernel_id() must equal the span index
// (matching the DX and Vulkan backends), zero-sized elements must be
// skipped without shifting the kernel ids of later elements, and each
// grid must be sized for its own dispatch rather than reusing another
// element's grid.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_multi_dispatch(Device &device) {
    log_level_verbose();

    constexpr std::array dispatches{
        uint3{3u, 1u, 1u},
        uint3{0u, 0u, 0u},   // skipped; kernel ids of later dispatches must not shift
        uint3{17u, 1u, 1u},
        uint3{4097u, 1u, 1u},// forces multiple blocks for any block size <= 1024
    };
    constexpr auto sentinel = 0xffffffffu;

    auto stream = device.create_stream();
    auto output = device.create_buffer<uint2>(dispatches.size());
    std::array<uint2, dispatches.size()> result{};
    result.fill(make_uint2(sentinel, sentinel));
    stream << output.copy_from(luisa::span{result});

    // the last thread of each grid records {kernel_id, dispatch_size.x}
    // so a wrong per-launch grid size leaves the slot at the sentinel
    Kernel1D kernel = [&](BufferUInt2 out) noexcept {
        $if (dispatch_id().x == dispatch_size().x - 1u) {
            out.write(kernel_id(), make_uint2(kernel_id(), dispatch_size().x));
        };
    };
    auto shader = device.compile(kernel);

    stream << shader(output).dispatch(luisa::span{dispatches})
           << output.copy_to(luisa::span{result})
           << synchronize();

    for (auto i = 0u; i < dispatches.size(); ++i) {
        if (any(dispatches[i] == make_uint3(0u))) {
            expect(all(result[i] == make_uint2(sentinel, sentinel)))
                << "zero-sized dispatch " << i << " must be skipped";
        } else {
            expect(all(result[i] == make_uint2(i, dispatches[i].x)))
                << "dispatch " << i << " must observe span index and size";
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    test_multi_dispatch(dc->device);
}
