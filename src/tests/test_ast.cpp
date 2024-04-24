#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    luisa::vector<Attribute> attris;
    attris.emplace_back("fuck", "shit");
    auto t = Type::buffer(Type::of<float>(), attris);
    LUISA_INFO("{}", t->description());
}
