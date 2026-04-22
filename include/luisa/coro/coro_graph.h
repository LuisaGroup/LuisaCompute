//
// Created by Mike on 2024/5/8.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/function.h>
#include <luisa/coro/coro_token.h>

namespace luisa::compute::coroutine {

class CoroFrameDesc;

class LUISA_CORO_API CoroGraph {

public:
    using CC = luisa::shared_ptr<const compute::detail::FunctionBuilder>;// current continuation function

public:
    class LUISA_CORO_API Node {

    private:
        luisa::vector<uint> _input_fields;
        luisa::vector<uint> _output_fields;
        luisa::vector<CoroToken> _targets;
        CC _cc;

    public:
        Node(luisa::vector<uint> input_fields,
             luisa::vector<uint> output_fields,
             luisa::vector<CoroToken> targets,
             CC current_continuation) noexcept;
        ~Node() noexcept;

    public:
        [[nodiscard]] auto input_fields() const noexcept { return luisa::span{_input_fields}; }
        [[nodiscard]] auto output_fields() const noexcept { return luisa::span{_output_fields}; }
        [[nodiscard]] auto targets() const noexcept { return luisa::span{_targets}; }
        [[nodiscard]] Function cc() const noexcept;
        [[nodiscard]] luisa::string dump() const noexcept;
    };

private:
    luisa::shared_ptr<const CoroFrameDesc> _frame;
    luisa::unordered_map<CoroToken, Node> _nodes;
    luisa::unordered_map<luisa::string, CoroToken> _named_tokens;

public:
    CoroGraph(luisa::shared_ptr<const CoroFrameDesc> frame_desc,
              luisa::unordered_map<CoroToken, Node> nodes,
              luisa::unordered_map<luisa::string, CoroToken> named_tokens) noexcept;
    ~CoroGraph() noexcept;

public:
    // create a coroutine graph from a coroutine function definition
    [[nodiscard]] static luisa::shared_ptr<const CoroGraph> create(Function coroutine) noexcept;

public:
    [[nodiscard]] auto frame() const noexcept { return _frame.get(); }
    [[nodiscard]] auto &shared_frame() const noexcept { return _frame; }
    [[nodiscard]] auto &nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto &named_tokens() const noexcept { return _named_tokens; }
    [[nodiscard]] const Node &entry() const noexcept;
    [[nodiscard]] const Node &node(CoroToken index) const noexcept;
    [[nodiscard]] const Node &node(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::coroutine
