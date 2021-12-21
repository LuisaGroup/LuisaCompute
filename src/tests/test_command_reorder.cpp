//
// Created by ChenXin on 2021/12/9.
//

#include <vector>

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

    CommandReorderVisitor commandReorderVisitor(device.impl(), 100);

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
        std::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 2);
        std::vector<int> size(reordered_list.size(), 0);
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
        std::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 3);
        std::vector<int> size(reordered_list.size(), 0);
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
        std::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 2);
        std::vector<int> size(reordered_list.size(), 0);
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
        std::vector<CommandList> reordered_list = commandReorderVisitor.getCommandLists();

        assert(reordered_list.size() == 3);
        std::vector<int> size(reordered_list.size(), 0);
        for (auto i = 0; i < reordered_list.size(); ++i) {
            auto &command_list = reordered_list[i];
            for (auto command : command_list)
                ++size[i];
        }
        assert(size[0] == 1);
        assert(size[1] == 1);
        assert(size[2] == 2);
    }

    return 0;
}