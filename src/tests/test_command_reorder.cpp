//
// Created by ChenXin on 2021/12/9.
//

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#else
    auto device = context.create_device("ispc");
#endif

    auto width = 1920u, height = 1080u;

    auto texture = device.create_image<float>(PixelStorage::BYTE4, width, height);
    auto texture1 = device.create_image<float>(PixelStorage::BYTE4, width, height);
    auto texture2 = device.create_image<float>(PixelStorage::BYTE4, width, height);

    auto buffer = device.create_buffer<float>(width * height);
    auto buffer1 = device.create_buffer<float>(width * height);
    auto buffer2 = device.create_buffer<float>(width * height);

    Kernel1D kernel = [](BufferFloat buffer) noexcept {
        buffer.write(0u, 1.f);
    };
    auto shader = device.compile(kernel);

    CommandReorderVisitor commandReorderVisitor(device.impl(), 100);

    {
        CommandList feed;
        feed.append(shader(buffer).dispatch(1024u));
        feed.append(buffer.copy_to(nullptr));
        for (auto cmd : feed) { cmd->accept(commandReorderVisitor); }
        auto reordered_lists = commandReorderVisitor.getCommandLists();
        LUISA_INFO("Size: {}.", reordered_lists.size());
        assert(reordered_lists.size() == 2u);
    }

    {
        auto bindless_array = device.create_bindless_array(3);
        bindless_array.emplace(0, buffer);
        bindless_array.emplace(1, buffer1);
        bindless_array.emplace(2, texture1, Sampler());

        CommandList feed;

        /*
         *          -- buffer2
         *        /
         * buffer -- buffer1 --
         *        \             \
         *          -----------  bindless_array
         *        /
         * texture1 -- texture -- texture2
         */
        feed.append(texture.copy_from(texture1));
        feed.append(texture2.copy_from(texture));
        feed.append(buffer1.copy_from(buffer));
        feed.append(buffer2.copy_from(buffer));
        feed.append(bindless_array.update());

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 2);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 3);
        assert(size[1] == 2);
    }

    {
        auto bindless_array = device.create_bindless_array(3);
        bindless_array.emplace(0, buffer);
        bindless_array.emplace(1, buffer1);
        bindless_array.emplace(2, texture1, Sampler());

        CommandList feed;

        /*
         *          -- buffer2
         *        /
         * buffer -- buffer1 --
         *        \             \
         *          -----------  bindless_array
         *        /
         * texture1 -- texture -- texture2
         */
        feed.append(texture.copy_from(texture1));
        feed.append(texture2.copy_from(texture));
        feed.append(buffer1.copy_from(buffer));
        feed.append(bindless_array.update());
        feed.append(buffer2.copy_from(buffer));
        /*
         * the same with the last test
         * but bindless_array is inserted before buffer2's copy
         */

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 3);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 2);
        assert(size[1] == 2);
        assert(size[2] == 1);
    }

    {
        auto bindless_array = device.create_bindless_array(2);
        bindless_array.emplace(0, buffer1);
        bindless_array.emplace(1, texture1, Sampler());

        CommandList feed;

        /*
         *          -- buffer2
         *        /
         * buffer -- buffer1 --
         *                      \
         *          ------------ bindless_array
         *        /
         * texture1 -- texture -- texture2
         */
        feed.append(texture.copy_from(texture1));
        feed.append(texture2.copy_from(texture));
        feed.append(buffer1.copy_from(buffer));
        feed.append(bindless_array.update());
        feed.append(buffer2.copy_from(buffer));

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 2);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 3);
        assert(size[1] == 2);
    }

    {
        CommandList feed;

        /*
         *                               -- buffer2
         *                             /
         * buffer -- buffer1 -- buffer -- buffer1
         */
        feed.append(buffer1.copy_from(buffer));
        feed.append(buffer.copy_from(buffer1));
        feed.append(buffer1.copy_from(buffer));
        feed.append(buffer2.copy_from(buffer));

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 3);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 1);
        assert(size[1] == 1);
        assert(size[2] == 2);
    }

    {
        CommandList feed;

        /*
         * texture ------           ----- shader2
         *                \       /
         * texture1 ------ shader ------- shader1
         *               /
         * texture2 ----
         */

        Kernel2D kernel = [](ImageFloat texture, ImageFloat texture1, ImageFloat texture2) {
            Var xy = dispatch_id().xy();
            Var data = texture1.read(xy).xyz();
            texture.write(xy, make_float4(data, 1.0f));
            texture2.write(xy, make_float4(data, 1.0f));
        };
        Kernel2D kernel1 = [](ImageFloat texture, ImageFloat texture1) {
            Var xy = dispatch_id().xy();
            Var data = texture.read(xy).xyz();
            texture1.write(xy, make_float4(data, 1.0f));
        };
        Kernel2D kernel2 = [](ImageFloat texture, ImageFloat texture2) {
            Var xy = dispatch_id().xy();
            Var data = texture.read(xy).xyz();
            texture2.write(xy, make_float4(data, 1.0f));
        };

        auto shader = device.compile(kernel);
        auto shader1 = device.compile(kernel1);
        auto shader2 = device.compile(kernel2);

        feed.append(shader(texture, texture1, texture2).dispatch(width, height));
        feed.append(shader1(texture, texture1).dispatch(width, height));
        feed.append(shader2(texture, texture2).dispatch(width, height));

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 2);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 1);
        assert(size[1] == 2);
    }

    {
        CommandList feed;

        auto vertex_buffer = device.create_buffer<Vector<float, 3>>(3);
        auto triangle_buffer = device.create_buffer<Triangle>(1);
        auto vertex_buffer1 = device.create_buffer<Vector<float, 3>>(3);
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
        auto mesh1 = device.create_mesh(vertex_buffer1, triangle_buffer);
        auto accel = device.create_accel();
        accel.emplace_back(mesh);
        assert(device.impl()->is_mesh_in_accel(accel.handle(), mesh.handle()));

        /*
         * vertex_buffer -------
         *                       \
         * triangle_buffer ------- mesh ------  ------- accel
         *                                     \
         * vertex_buffer1 ---------------------  ------ mesh1
         */

        feed.append(mesh.build());
        feed.append(mesh1.build());
        feed.append(accel.build());

        for (auto command : feed) {
            command->accept(commandReorderVisitor);
        }
        luisa::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();
        assert(reordered_list.size() == 2);
        luisa::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 1);
        assert(size[1] == 2);
    }

    return 0;
}