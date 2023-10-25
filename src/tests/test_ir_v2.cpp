#include <luisa/ir_v2/ir_v2.h>
using namespace luisa::compute;
int main() {
    auto pool = luisa::make_shared<ir_v2::Pool>();
    auto builder = ir_v2::IrBuilder{pool};
    auto x = builder.const_(1.0f);
    auto y = builder.const_(2.0f);
    luisa::vector<ir_v2::Node *> args{x, y};
    auto z = builder.call(ir_v2::FuncTag::ADD, luisa::span{args}, Type::of<float>());
}