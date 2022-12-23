//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command.h>
#include <runtime/command_list.h>

namespace luisa::compute {

void CommandList::_recycle() noexcept {
    _commands.clear();
}

void CommandList::append(luisa::unique_ptr<Command>&& cmd) noexcept {
    _commands.emplace_back(std::move(cmd));
}

luisa::vector<luisa::unique_ptr<Command>> CommandList::steal_commands() noexcept {
    return std::move(_commands);
}

CommandList::CommandList(CommandList &&another) noexcept = default;

CommandList &CommandList::operator=(CommandList &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        _recycle();
        _commands = std::move(rhs._commands);
    }
    return *this;
}

CommandList::~CommandList() noexcept { _recycle(); }

//class CommandDumpVisitor : CommandVisitor {
//
//private:
//    nlohmann::json *_json{nullptr};
//
//private:
//    void visit(const BufferUploadCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "buffer_upload";
//        json["handle"] = command->handle();
//        json["offset"] = command->offset();
//        json["size"] = command->size();
//        json["data"] = luisa::format("{}", command->data());
//    }
//    void visit(const BufferDownloadCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "buffer_download";
//        json["handle"] = command->handle();
//        json["offset"] = command->offset();
//        json["size"] = command->size();
//        json["data"] = luisa::format("{}", command->data());
//    }
//    void visit(const BufferCopyCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "buffer_copy";
//        json["src_handle"] = command->src_handle();
//        json["dst_handle"] = command->dst_handle();
//        json["src_offset"] = command->src_offset();
//        json["dst_offset"] = command->dst_offset();
//        json["size"] = command->size();
//    }
//    void visit(const BufferToTextureCopyCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "buffer_to_texture_copy";
//        json["buffer_handle"] = command->buffer();
//        json["buffer_offset"] = command->buffer_offset();
//        json["texture_handle"] = command->texture();
//        json["texture_level"] = command->level();
//        json["size"] = std::array{command->size().x, command->size().y, command->size().z};
//    }
//    void visit(const ShaderDispatchCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "shader_dispatch";
//        json["handle"] = command->handle();
//        json["dispatch_size"] = std::array{command->dispatch_size().x,
//                                           command->dispatch_size().y,
//                                           command->dispatch_size().z};
//        auto arg_array = nlohmann::json::array();
//        constexpr auto usage_string = [](Usage usage) noexcept {
//            switch (usage) {
//                case Usage::NONE: return "none";
//                case Usage::READ: return "read";
//                case Usage::WRITE: return "write";
//                case Usage::READ_WRITE: return "read_write";
//            }
//            return "unknown";
//        };
//        command->decode([&arg_array, command, usage_string](auto arg) noexcept {
//            using T = std::decay_t<decltype(arg)>;
//            if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
//                arg_array.emplace_back(nlohmann::json{{"type", "buffer"},
//                                                      {"handle", arg.handle},
//                                                      {"offset", arg.offset},
//                                                      {"size", arg.size}});
//            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
//                arg_array.emplace_back(nlohmann::json{{"type", "texture"},
//                                                      {"handle", arg.handle},
//                                                      {"level", arg.level}});
//            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
//                arg_array.emplace_back(nlohmann::json{{"type", "bindless_array"},
//                                                      {"handle", arg.handle}});
//            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
//                arg_array.emplace_back(nlohmann::json{{"type", "accel"},
//                                                      {"handle", arg.handle}});
//            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::UniformArgument>) {
//                arg_array.emplace_back(nlohmann::json{{"type", "uniform"},
//                                                      {"date", luisa::format("{}", static_cast<const void *>(arg.data))},
//                                                      {"size", arg.size}});
//            } else {
//                static_assert(always_false_v<T>, "Invalid argument type");
//            }
//        });
//        json["arguments"] = std::move(arg_array);
//    }
//    void visit(const TextureUploadCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "texture_upload";
//        json["handle"] = command->handle();
//        json["level"] = command->level();
//        json["data"] = luisa::format("{}", command->data());
//        json["size"] = std::array{command->size().x, command->size().y, command->size().z};
//    }
//    void visit(const TextureDownloadCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "texture_download";
//        json["handle"] = command->handle();
//        json["level"] = command->level();
//        json["data"] = luisa::format("{}", command->data());
//        json["size"] = std::array{command->size().x, command->size().y, command->size().z};
//    }
//    void visit(const TextureCopyCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "texture_copy";
//        json["src_handle"] = command->src_handle();
//        json["src_level"] = command->src_level();
//        json["dst_handle"] = command->dst_handle();
//        json["dst_level"] = command->dst_level();
//        json["size"] = std::array{command->size().x, command->size().y, command->size().z};
//    }
//    void visit(const TextureToBufferCopyCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "texture_to_buffer_copy";
//        json["texture_handle"] = command->texture();
//        json["texture_level"] = command->level();
//        json["buffer_handle"] = command->buffer();
//        json["buffer_offset"] = command->buffer_offset();
//        json["size"] = std::array{command->size().x, command->size().y, command->size().z};
//    }
//    void visit(const AccelBuildCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "accel_build";
//        json["accel_handle"] = command->handle();
//        json["instance_count"] = command->instance_count();
//        switch (command->request()) {
//            case AccelBuildRequest::PREFER_UPDATE:
//                json["request"] = "prefer_update";
//                break;
//            case AccelBuildRequest::FORCE_BUILD:
//                json["request"] = "force_build";
//                break;
//        }
//        auto mods = nlohmann::json::array();
//        for (auto &mod : command->modifications()) {
//            auto &m = mods.emplace_back(nlohmann::json{});
//            m["index"] = mod.index;
//            m["flags"] = mod.flags;
//            m["mesh"] = mod.mesh;
//            m["affine"] = std::array{
//                mod.affine[0], mod.affine[1], mod.affine[2],
//                mod.affine[3], mod.affine[4], mod.affine[5],
//                mod.affine[6], mod.affine[7], mod.affine[8],
//                mod.affine[9], mod.affine[10], mod.affine[11]};
//        }
//        json["modifiers"] = std::move(mods);
//    }
//    void visit(const MeshBuildCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "mesh_build";
//        json["handle"] = command->handle();
//        switch (command->request()) {
//            case AccelBuildRequest::PREFER_UPDATE:
//                json["request"] = "prefer_update";
//                break;
//            case AccelBuildRequest::FORCE_BUILD:
//                json["request"] = "force_build";
//                break;
//        }
//        json["vertex_buffer"] = command->vertex_buffer();
//        json["triangle_buffer"] = command->triangle_buffer();
//    }
//    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
//        auto &json = _json->emplace_back(nlohmann::json{});
//        json["type"] = "bindless_array_update";
//        json["handle"] = command->handle();
//    }
//
//public:
//    [[nodiscard]] nlohmann::json dump(const CommandList &list) noexcept {
//        auto json = nlohmann::json::array();
//        _json = &json;
//        for (auto cmd : list) { cmd->accept(*this); }
//        _json = nullptr;
//        return json;
//    }
//};
//
//nlohmann::json CommandList::dump_json() const noexcept {
//    CommandDumpVisitor visitor;
//    return visitor.dump(*this);
//}
//
//void CommandList::reserve(size_t size) noexcept {
//    _commands.reserve(size);
//}

}// namespace luisa::compute
