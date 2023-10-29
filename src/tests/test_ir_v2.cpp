#include <luisa/ir_v2/ir_v2.h>
#include <luisa/dsl/syntax.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
using namespace luisa::compute;
using namespace luisa::compute::ir_v2;
struct Foo {
    int x;
    float y;
};
LUISA_STRUCT(Foo, x, y) {};

int main(int argc, char *argv[]) {
    log_level_verbose();
    auto mod = KernelModule();
    {
        auto &pool = mod.pool;
        auto buf = pool->alloc<Node>(Instruction(InstructionTag::BUFFER), Type::of<Foo>());
        mod.args.push_back(buf);
        auto builder = IrBuilder{pool};
        auto dispatch_id = builder.call(FuncTag::DISPATCH_ID, {}, Type::of<uint3>());
        auto tid = builder.extract_element(dispatch_id, luisa::span{std::array{0u}}, Type::of<uint>());
        auto foo = builder.local(builder.call(FuncTag::BUFFER_READ, luisa::span{std::array{buf, tid}}, Type::of<Foo>()));
        auto foo_x = builder.gep(foo, luisa::span{std::array{0u}}, Type::of<int>());
        builder.update(foo_x, builder.const_(int(1)));
        (void)builder.call(FuncTag::BUFFER_WRITE, luisa::span{std::array<const Node*, 3>{buf, tid, foo}}, Type::of<void>());
        mod.entry = std::move(builder).finish();
    }
    LUISA_VERBOSE("{}", dump_human_readable(mod));
    return 0;
}