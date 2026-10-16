#pragma once
#include <cstddef>

#include <cstdint>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {

// Static call/return relations, independent of scheduler queue transitions.
// Function zero is the coroutine root. A return follows only the edge whose
// return_site was stored by the active invocation, never an arbitrary edge.
struct CoroCallGraph {
    size_t analysis_state_count{0u};
    struct Function {
        uint32_t id{0u};
        luisa::vector<uint32_t> resume_tokens;
    };
    struct Edge {
        uint32_t caller{0u};
        uint32_t callee{0u};
        uint32_t return_site{0u};
    };
    luisa::vector<Function> functions;
    luisa::vector<Edge> edges;
};

}// namespace luisa::compute
