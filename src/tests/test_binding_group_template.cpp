#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

template<class T>
struct TArguments {
    Image<T> image;
    uint2 resolution;
};
template<class T>
struct TArgumentsView {
    ImageView<T> image;
    uint2 resolution;
};
template<class T>
struct TNestedArguments {
    TArgumentsView<T> args;
    Image<T> image;
};

#define TEMPLATE_T()  \
    template<class T> \
        requires is_legal_image_element<T>

LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TArguments<T>, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        this->image->write(coord, color);
    }
};
LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TArgumentsView<T>, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        this->image->write(coord, color);
    }
};
LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TNestedArguments<T>, args, image) {
    void blit(const UInt2 &coord) noexcept {
        auto color = this->args.image.read(coord).xyz();
        this->image->write(coord, make_float4(1.f - color, 1.f));
    }
};
using Arguments = TArguments<float>;
using ArgumentsView = TArgumentsView<float>;
using NestedArguments = TNestedArguments<float>;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }

    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    Callable color = [](UInt2 coord, Var<Arguments> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Callable color_with_view = [](UInt2 coord, Var<ArgumentsView> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Kernel2D kernel = [&color](Var<Arguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color(coord, args));
    };

    Kernel2D kernel_with_view = [&color_with_view](Var<ArgumentsView> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color_with_view(coord, args));
    };

    Kernel2D kernel_with_nested = [](Var<NestedArguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->blit(coord);
    };

    auto shader = device.compile(kernel);
    auto shader_with_view = device.compile(kernel_with_view);
    auto shader_with_nested = device.compile(kernel_with_nested);

    Arguments args{
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024)),
        .resolution = make_uint2(1024, 1024)};

    ArgumentsView args_view{
        .image = args.image.view(),
        .resolution = args.resolution};

    NestedArguments args_nested{
        .args = args_view,
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024))};

    luisa::vector<std::byte> host_image(args.image.view().size_bytes());

    // simple binding group
    stream << shader(args).dispatch(args.resolution)
           << args.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_binding_group.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);

    // binding group with view
    stream << shader_with_view(args_view).dispatch(args_view.resolution)
           << args.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_binding_group_with_view.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);

    // nested binding group
    stream << shader_with_nested(args_nested).dispatch(args_nested.image.view().size())
           << args_nested.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_binding_group_nested.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);
}
