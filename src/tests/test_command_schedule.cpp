//
// Created by ChenXin on 2021/12/9.
//

#include <luisa-compute.h>
#include <runtime/command_reorder_visitor.h>
#include <nlohmann/json.hpp>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    CommandReorderVisitor commandReorderVisitor(device.impl());
    commandReorderVisitor.clear();
    auto CreateBuffer = [&](size_t byte_size) {
        return device.create_buffer<float>(byte_size / 4);
    };
    auto b139910841368576 = CreateBuffer(67108864);
    auto b139910036062208 = CreateBuffer(536870912);
    auto b139909230755840 = CreateBuffer(268435456);
    auto b139910640041984 = CreateBuffer(67108864);
    auto b139912585532928 = CreateBuffer(2552 + 4);
    auto b139912585528832 = CreateBuffer(2552 + 4);
    auto b139914465247232 = CreateBuffer(67108864);
    auto b139911713783808 = CreateBuffer(201326592);
    auto b139911512457216 = CreateBuffer(201326592);
    auto b139912585507840 = CreateBuffer(12400);
    auto b139910707150848 = CreateBuffer(67108864);
    auto b139910908477440 = CreateBuffer(67108864);
    auto b139911311130624 = CreateBuffer(201326592);
    auto b139911244021760 = CreateBuffer(67108864);
    auto b139910975586304 = CreateBuffer(268435456);
    auto ba = device.create_bindless_array();
    ba.emplace(0u, b139909230755840);
    ba.emplace(1u, b139912585507840);
    Kernel1D k0 = [&] {
        auto x0 = b139910841368576.read(0u);
        auto x1 = b139910036062208.read(0u);
        auto x2 = b139909230755840.read(0u);
        auto x3 = b139910640041984.read(0u);
        auto x4 = b139912585532928.read(0u);
        auto x5 = b139914465247232.read(0u);
        auto x6 = b139911713783808.read(0u);
        auto x7 = b139911512457216.read(0u);
        auto x8 = b139912585507840.read(0u);
        b139911512457216.write(0u, 0.f);
        auto x = ba.buffer<uint>(0u).read(0u);
    };
    Kernel1D k1 = [&] {
        Var a0 = b139910841368576.read(0);
        Var a1 = b139910036062208.read(0);
        Var a2 = b139909230755840.read(0);
        Var a3 = b139910707150848.read(0);
        Var a4 = b139912585528832.read(0);
        Var a5 = b139912585507840.read(0);
        Var a6 = b139910908477440.read(0);
        b139910908477440.write(0, 1.0f);
        b139911311130624.write(0, 1.0f);
        b139911244021760.write(0, 1.0f);
        b139910975586304.write(0, 1.0f);
        auto x = ba.buffer<uint>(0u).read(0u);
    };
    auto s0 = device.compile(k0);
    auto s1 = device.compile(k1);

    for (auto r = 0u; r < 10u; r++) {
        luisa::vector<Command *> cmd(2);
        cmd[0] = s0().dispatch(1);
        cmd[1] = s1().dispatch(1);

        CommandScheduler s{device.impl()};
        s.add(cmd[0]);
        s.add(cmd[1]);
        auto lists = s.schedule();
        assert(lists.size() == 1);

        for (auto &&i : cmd) {
            i->accept(commandReorderVisitor);
        }
        auto reordered_list = commandReorderVisitor.command_lists();
        assert(reordered_list.size() == 1);
        assert(reordered_list[0].size() == 2);
        commandReorderVisitor.clear();
    }
    return 0;
}
