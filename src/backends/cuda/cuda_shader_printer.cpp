#include "cuda_command_encoder.h"
#include "cuda_shader_printer.h"

namespace luisa::compute::cuda {

class CUDAShaderPrinter::Callback : public CUDACallbackContext {

private:
    const CUDAShaderPrinter *_printer;
    const void *_data;

public:
    Callback(const CUDAShaderPrinter *printer, const void *data) noexcept
        : _printer{printer}, _data{data} {}

private:
    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<Callback> pool;
        return pool;
    }

public:
    [[nodiscard]] static auto create(const CUDAShaderPrinter *printer,
                                     const void *data) noexcept {
        return _pool().create(printer, data);
    }

public:
    void recycle() noexcept override {
        _printer->_do_print(_data);
        _pool().destroy(this);
    }
};

class CUDAShaderPrinter::Formatter {

public:
    using Primitive = luisa::variant<
        Type::Tag, luisa::string>;

private:
    size_t _size{};
    luisa::vector<size_t> _offsets;
    luisa::vector<Primitive> _primitives;

public:
    Formatter(luisa::string_view fmt, const Type *arg_pack) noexcept {
        LUISA_ASSERT(arg_pack->members().size() >= 2u &&
                         arg_pack->members()[0] == Type::of<uint>() &&
                         arg_pack->members()[1] == Type::of<uint>(),
                     "Invalid argument pack for shader printer.");
        // TODO: parse the fmt string
        auto offset = static_cast<size_t>(8u);
        auto args = arg_pack->members().subspan(2u);
        luisa::string s;
        luisa::string f;
        auto commit_s = [this, &s] {
            if (!s.empty()) {
                _offsets.push_back(0u);
                _primitives.emplace_back(s);
                s.clear();
            }
        };
        while (!fmt.empty()) {
            auto c = fmt.front();
            fmt.remove_prefix(1u);
            if (c == '{') {
                if (!f.empty()) {
                    LUISA_ERROR_WITH_LOCATION("Invalid format string.");
                } else if (fmt.front() == '{') {// escape
                    fmt.remove_prefix(1u);
                    s.push_back('{');
                } else {
                    f.push_back('{');
                    commit_s();
                }
            } else if (c == '}') {
                if (!f.empty()) {// end of format group
                    f.push_back('}');
                    LUISA_ASSERT(f == "{}", "Unsupported format string '{}'.", f);// TODO: support more formats?
                    LUISA_ASSERT(!args.empty(), "Not enough arguments for shader printer.");
                    auto arg = args.front();
                    args = args.subspan(1u);
                    auto encode = [this, &s, &commit_s](auto &&self, size_t offset, const Type *arg) noexcept -> void {
                        if (arg->is_scalar()) {
                            _offsets.push_back(offset);
                            _primitives.emplace_back(arg->tag());
                        } else if (arg->is_vector()) {
                            s.push_back('(');
                            commit_s();
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, arg->element());
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += arg->element()->size();
                            }
                            s.push_back(')');
                            commit_s();
                        } else if (arg->is_array()) {
                            s.push_back('[');
                            commit_s();
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, arg->element());
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += arg->element()->size();
                            }
                            s.push_back(']');
                            commit_s();
                        } else if (arg->is_matrix()) {
                            s.push_back('(');
                            commit_s();
                            auto column = Type::vector(arg->element(), arg->dimension());
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, column);
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += column->size();
                            }
                            s.push_back(')');
                            commit_s();
                        } else if (arg->is_structure()) {
                            s.push_back('{');
                            commit_s();
                            for (auto i = 0u; i < arg->members().size(); i++) {
                                auto member = arg->members()[i];
                                offset = luisa::align(offset, member->alignment());
                                self(self, offset, member);
                                if (i + 1u < arg->members().size()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += member->size();
                            }
                            s.push_back('}');
                            commit_s();
                        } else {
                            LUISA_ERROR_WITH_LOCATION(
                                "Invalid argument type '{}' for printing.",
                                arg->description());
                        }
                    };
                    offset = luisa::align(offset, arg->alignment());
                    encode(encode, offset, arg);
                    offset += arg->size();
                    f.clear();
                } else {// not in a format group, only escape is allowed
                    if (fmt.front() == '}') {
                        fmt.remove_prefix(1u);
                        s.push_back('}');
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid format string.");
                    }
                }
            } else {
                if (!f.empty()) {
                    f.push_back(c);
                } else {
                    s.push_back(c);
                }
            }
        }
        LUISA_ASSERT(f.empty(), "Invalid format string.");
        if (!args.empty()) {
            LUISA_WARNING_WITH_LOCATION(
                "Too many arguments for shader printer. Ignored.");
        }
        commit_s();
        _size = luisa::align(offset, arg_pack->alignment());
        LUISA_ASSERT(_size == arg_pack->size(), "Invalid argument pack for shader printer.");
        // optimize the format
        luisa::vector<Primitive> primitives;
        luisa::vector<size_t> offsets;
        primitives.reserve(_primitives.size());
        offsets.reserve(_offsets.size());
        for (auto i = 0u; i < _primitives.size(); i++) {
            luisa::visit(
                [&](auto &&p) noexcept {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, Type::Tag>) {
                        primitives.emplace_back(p);
                        offsets.emplace_back(_offsets[i]);
                    } else {
                        static_assert(std::is_same_v<T, luisa::string>);
                        if (primitives.empty() || !luisa::holds_alternative<luisa::string>(primitives.back())) {
                            primitives.emplace_back(p);
                            offsets.emplace_back(_offsets[i]);
                        } else {
                            luisa::get<luisa::string>(primitives.back()).append(p);
                        }
                    }
                },
                _primitives[i]);
        }
        _primitives = std::move(primitives);
        _offsets = std::move(offsets);
    }
    ~Formatter() noexcept = default;

public:
    bool operator()(luisa::string &scratch, luisa::span<const std::byte> item) const noexcept {
        if (item.size() != _size) { return false; }
        for (auto i = 0u; i < _offsets.size(); i++) {
            auto data = item.data() + _offsets[i];
            luisa::visit(
                [&](auto &&p) noexcept {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, Type::Tag>) {
                        auto print_primitive = [&](auto &&p) noexcept {
                            luisa::format_to(std::back_inserter(scratch), "{}", p);
                        };
                        switch (p) {
                            case Type::Tag::INT8: print_primitive(*reinterpret_cast<const int8_t *>(data)); break;
                            case Type::Tag::UINT8: print_primitive(*reinterpret_cast<const uint8_t *>(data)); break;
                            case Type::Tag::INT16: print_primitive(*reinterpret_cast<const int16_t *>(data)); break;
                            case Type::Tag::UINT16: print_primitive(*reinterpret_cast<const uint16_t *>(data)); break;
                            case Type::Tag::INT32: print_primitive(*reinterpret_cast<const int32_t *>(data)); break;
                            case Type::Tag::UINT32: print_primitive(*reinterpret_cast<const uint32_t *>(data)); break;
                            case Type::Tag::INT64: print_primitive(*reinterpret_cast<const int64_t *>(data)); break;
                            case Type::Tag::UINT64: print_primitive(*reinterpret_cast<const uint64_t *>(data)); break;
                            case Type::Tag::FLOAT16: print_primitive(*reinterpret_cast<const half *>(data)); break;
                            case Type::Tag::FLOAT32: print_primitive(*reinterpret_cast<const float *>(data)); break;
                            case Type::Tag::FLOAT64: print_primitive(*reinterpret_cast<const double *>(data)); break;
                            default: LUISA_ERROR_WITH_LOCATION("Unsupported type for shader printer.");
                        }
                    } else {
                        static_assert(std::is_same_v<T, luisa::string>);
                        scratch.append(p);
                    }
                },
                _primitives[i]);
        }
        return true;
    }
};

luisa::unique_ptr<CUDAShaderPrinter> CUDAShaderPrinter::create(luisa::span<const std::pair<luisa::string, const Type *>> arg_types) noexcept {
    if (arg_types.empty()) { return nullptr; }
    luisa::vector<luisa::unique_ptr<Formatter>> formatters;
    formatters.reserve(arg_types.size());
    for (auto &&[name, type] : arg_types) {
        formatters.emplace_back(luisa::make_unique<Formatter>(name, type));
    }
    return luisa::make_unique<CUDAShaderPrinter>(std::move(formatters));// TODO
}

luisa::unique_ptr<CUDAShaderPrinter> CUDAShaderPrinter::create(luisa::span<const std::pair<luisa::string, luisa::string>> arg_types) noexcept {
    luisa::vector<std::pair<luisa::string, const Type *>> types;
    types.reserve(arg_types.size());
    for (auto &&[name, type] : arg_types) {
        types.emplace_back(name, Type::from(type));
    }
    return create(types);
}

CUDAShaderPrinter::Binding CUDAShaderPrinter::encode(CUDACommandEncoder &encoder) const noexcept {
    Binding b{
        .capacity = print_buffer_content_capacity,
        .content = 0ull,
    };
    encoder.with_download_pool_no_fallback(
        print_buffer_capacity,
        [&b, &encoder, this](CUDAHostBufferPool::View *temp) noexcept {
            if (temp == nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to allocate temporary buffer for shader "
                    "printer. Printing is disabled this time.");
                return;
            }
            *reinterpret_cast<size_t *>(temp->address()) = 0ul;
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&b.content, temp->address(), 0u));
            encoder.add_callback(Callback::create(this, temp->address()));
        });
    return b;
}

void CUDAShaderPrinter::_do_print(const void *data) const noexcept {
    struct Head {
        size_t size;
        const std::byte content[];
    };
    auto *head = reinterpret_cast<const Head *>(data);
    if (head->size == 0u) { return; }
    LUISA_INFO("[DEVICE] Printing {} byte(s)...", head->size);
    auto offset = static_cast<size_t>(0u);
    luisa::string scratch;
    scratch.reserve(128_k);
    while (offset < head->size && offset < print_buffer_content_capacity) {
        struct Item {
            uint size;
            uint fmt;
            const std::byte data[];
        };
        static_assert(sizeof(Item) == 8u);
        auto content = head->content + offset;
        auto *item = reinterpret_cast<const Item *>(content);
        if (auto item_end = offset + item->size;
            item_end > head->size ||
            item_end > print_buffer_content_capacity) { break; }
        if (item->fmt < _formatters.size()) {
            scratch.clear();
            luisa::span payload{content, item->size};
            if ((*_formatters[item->fmt])(scratch, payload)) {
                LUISA_INFO("[DEVICE] {}", scratch);// TODO: use a standalone sink?
            } else {
                break;
            }
        } else {
            LUISA_WARNING("Unknown print format: {}", item->fmt);
        }
        offset += item->size;
    }
    if (head->size > print_buffer_content_capacity) {
        LUISA_WARNING("Device print overflow. {} byte(s) truncated.",
                      head->size - print_buffer_content_capacity);
    }
}

CUDAShaderPrinter::CUDAShaderPrinter(vector<unique_ptr<Formatter>> &&formatters) noexcept
    : _formatters{std::move(formatters)} {}

CUDAShaderPrinter::~CUDAShaderPrinter() noexcept = default;

}// namespace luisa::compute::cuda
