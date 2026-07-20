// Tests for backend-independent motion-instance keyframes and ownership.
// This test covers:
// - MATRIX keyframes start as exact identity matrices.
// - SRT keyframes start with canonical identity components.
// - Backend handles are destroyed exactly once across moves.

#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/remote/client_interface.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/motion_instance.h>
#include <luisa/runtime/rtx/triangle.h>

#include <cmath>
#include <cstdint>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

class MotionTestDevice final : public ClientInterface {
private:
    uint64_t _next_handle{1u};
    size_t _motion_create_count{};
    luisa::vector<uint64_t> _destroyed_motion_handles;

    [[nodiscard]] ResourceCreationInfo _create_resource() noexcept {
        return ResourceCreationInfo{_next_handle++, nullptr};
    }

public:
    explicit MotionTestDevice(Context context) noexcept
        : ClientInterface{std::move(context), nullptr} {}

    [[nodiscard]] BufferCreationInfo create_buffer(
        const Type *element, size_t element_count, void *) noexcept override {
        auto stride = element == nullptr ? 1u : element->size();
        BufferCreationInfo info{};
        info.handle = _next_handle++;
        info.native_handle = nullptr;
        info.element_stride = stride;
        info.total_size_bytes = stride * element_count;
        return info;
    }

    [[nodiscard]] BufferCreationInfo create_buffer(
        const ir::CArc<ir::Type> *, size_t, void *) noexcept override {
        return BufferCreationInfo::make_invalid();
    }

    void destroy_buffer(uint64_t) noexcept override {}

    [[nodiscard]] ResourceCreationInfo create_mesh(
        const AccelOption &) noexcept override {
        return _create_resource();
    }

    void destroy_mesh(uint64_t) noexcept override {}

    [[nodiscard]] ResourceCreationInfo create_motion_instance(
        const AccelMotionOption &) noexcept override {
        _motion_create_count++;
        return _create_resource();
    }

    void destroy_motion_instance(uint64_t handle) noexcept override {
        _destroyed_motion_handles.emplace_back(handle);
    }

    [[nodiscard]] auto motion_create_count() const noexcept {
        return _motion_create_count;
    }

    [[nodiscard]] auto destroyed_motion_handles() const noexcept {
        return luisa::span{_destroyed_motion_handles};
    }
};

[[nodiscard]] luisa::shared_ptr<MotionTestDevice> make_test_backend(
    luisa::string_view program_path) {
    return luisa::make_shared<MotionTestDevice>(Context{program_path});
}

[[nodiscard]] Mesh make_test_mesh(Device &device) {
    auto vertices = device.create_buffer<float3>(3u);
    auto triangles = device.create_buffer<Triangle>(1u);
    return device.create_mesh(vertices, triangles);
}

void expect_matrix_identity(const MotionInstance &instance, size_t keyframe_count) {
    auto keyframes = instance.keyframes_matrix();
    expect(keyframes.size() == keyframe_count);
    for (auto &&keyframe : keyframes) {
        for (auto column = 0u; column < 4u; column++) {
            for (auto row = 0u; row < 4u; row++) {
                auto expected = column == row ? 1.0f : 0.0f;
                expect(std::abs(keyframe[column][row] - expected) < 1.0e-7f);
            }
        }
    }
}

void expect_srt_identity(const MotionInstance &instance, size_t keyframe_count) {
    auto keyframes = instance.keyframes_srt();
    expect(keyframes.size() == keyframe_count);
    for (auto &&keyframe : keyframes) {
        for (auto value : keyframe.pivot) { expect(std::abs(value) < 1.0e-7f); }
        for (auto i = 0u; i < 3u; i++) { expect(std::abs(keyframe.quaternion[i]) < 1.0e-7f); }
        expect(std::abs(keyframe.quaternion[3] - 1.0f) < 1.0e-7f);
        for (auto value : keyframe.scale) { expect(std::abs(value - 1.0f) < 1.0e-7f); }
        for (auto value : keyframe.shear) { expect(std::abs(value) < 1.0e-7f); }
        for (auto value : keyframe.translation) { expect(std::abs(value) < 1.0e-7f); }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto program_path = luisa::string{argc > 0 && argv[0] != nullptr ? argv[0] : "."};

    "motion instance keyframes start at canonical identities"_test = [program_path] {
        constexpr auto keyframe_count = 3u;
        auto backend = make_test_backend(program_path);
        {
            Device device{backend};
            auto mesh = make_test_mesh(device);

            AccelMotionOption matrix_option{};
            matrix_option.keyframe_count = keyframe_count;
            matrix_option.mode = AccelMotionMode::MATRIX;
            auto matrix_instance = device.create_motion_instance(mesh, matrix_option);
            expect_matrix_identity(matrix_instance, keyframe_count);

            AccelMotionOption srt_option{};
            srt_option.keyframe_count = keyframe_count;
            srt_option.mode = AccelMotionMode::SRT;
            auto srt_instance = device.create_motion_instance(mesh, srt_option);
            expect_srt_identity(srt_instance, keyframe_count);
        }
        expect(backend->motion_create_count() == 2u);
        expect(backend->destroyed_motion_handles().size() == 2u);
    };

    "motion instance destroys its backend handle exactly once"_test = [program_path] {
        auto backend = make_test_backend(program_path);
        uint64_t handle{};
        {
            Device device{backend};
            auto mesh = make_test_mesh(device);
            AccelMotionOption option{};
            option.keyframe_count = 2u;
            auto instance = device.create_motion_instance(mesh, option);
            handle = instance.handle();
            expect(backend->destroyed_motion_handles().empty());
        }
        auto destroyed = backend->destroyed_motion_handles();
        expect(destroyed.size() == 1u);
        expect(destroyed.front() == handle);
    };

    "motion instance move construction transfers ownership"_test = [program_path] {
        auto backend = make_test_backend(program_path);
        uint64_t handle{};
        {
            Device device{backend};
            auto mesh = make_test_mesh(device);
            AccelMotionOption option{};
            option.keyframe_count = 2u;
            auto source = device.create_motion_instance(mesh, option);
            handle = source.handle();
            {
                auto target = std::move(source);
                expect(!source);
                expect(target.handle() == handle);
                expect(backend->destroyed_motion_handles().empty());
            }
            expect(backend->destroyed_motion_handles().size() == 1u);
        }
        auto destroyed = backend->destroyed_motion_handles();
        expect(destroyed.size() == 1u);
        expect(destroyed.front() == handle);
    };

    "motion instance move assignment retires and adopts handles"_test = [program_path] {
        auto backend = make_test_backend(program_path);
        uint64_t old_handle{};
        uint64_t adopted_handle{};
        {
            Device device{backend};
            auto mesh = make_test_mesh(device);
            AccelMotionOption option{};
            option.keyframe_count = 2u;
            auto target = device.create_motion_instance(mesh, option);
            auto source = device.create_motion_instance(mesh, option);
            old_handle = target.handle();
            adopted_handle = source.handle();
            target = std::move(source);
            expect(!source);
            expect(target.handle() == adopted_handle);
            auto destroyed = backend->destroyed_motion_handles();
            expect(destroyed.size() == 1u);
            expect(destroyed.front() == old_handle);
        }
        auto destroyed = backend->destroyed_motion_handles();
        expect(destroyed.size() == 2u);
        expect(destroyed[0u] == old_handle);
        expect(destroyed[1u] == adopted_handle);
    };

    "motion instance self move preserves ownership"_test = [program_path] {
        auto backend = make_test_backend(program_path);
        uint64_t handle{};
        {
            Device device{backend};
            auto mesh = make_test_mesh(device);
            AccelMotionOption option{};
            option.keyframe_count = 2u;
            auto instance = device.create_motion_instance(mesh, option);
            handle = instance.handle();
            instance = std::move(instance);
            expect(instance.valid());
            expect(instance.handle() == handle);
            expect(backend->destroyed_motion_handles().empty());
        }
        auto destroyed = backend->destroyed_motion_handles();
        expect(destroyed.size() == 1u);
        expect(destroyed.front() == handle);
    };
}
