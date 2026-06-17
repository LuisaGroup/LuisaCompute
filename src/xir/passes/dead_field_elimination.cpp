#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/dead_field_elimination.h>

namespace luisa::compute::xir {

namespace {

static constexpr size_t FRAME_RESERVED_FIELD_COUNT = 3u;

}// namespace

DeadFieldEliminationInfo dead_field_elimination_pass_run(
    CoroMaterializeInfo &info,
    CoroFrameDesc &desc) noexcept {

    DeadFieldEliminationInfo result;
    result.original_field_count = info.frame_field_count;
    result.original_frame_size = desc.total_size();

    luisa::unordered_set<size_t> used_fields;
    for (size_t i = 0u; i < FRAME_RESERVED_FIELD_COUNT; ++i) {
        used_fields.emplace(i);
    }

    for (const auto &edge : info.edges) {
        for (auto idx : edge.load_fields) {
            used_fields.emplace(idx);
        }
    }

    luisa::unordered_set<size_t> dead_fields;
    for (size_t i = FRAME_RESERVED_FIELD_COUNT; i < info.frame_field_count; ++i) {
        if (!used_fields.contains(i)) {
            dead_fields.emplace(i);
        }
    }

    result.eliminated_field_count = dead_fields.size();
    result.remaining_field_count = info.frame_field_count - dead_fields.size();
    result.eliminated_field_indices = dead_fields;

    if (dead_fields.empty()) {
        result.new_frame_size = result.original_frame_size;
        return result;
    }

    luisa::unordered_map<size_t, size_t> old_to_new;
    size_t new_idx = 0u;
    for (size_t i = 0u; i < info.frame_field_count; ++i) {
        if (!dead_fields.contains(i)) {
            old_to_new[i] = new_idx++;
        }
    }

    luisa::vector<luisa::string> dead_names;
    for (auto &[name, idx] : info.name_to_field) {
        if (dead_fields.contains(idx)) {
            dead_names.push_back(name);
        } else {
            idx = old_to_new.at(idx);
        }
    }
    for (const auto &name : dead_names) {
        info.name_to_field.erase(name);
    }

    for (auto &edge : info.edges) {
        auto filter_remap = [&](luisa::vector<size_t> &fields) {
            size_t write = 0u;
            for (size_t read = 0u; read < fields.size(); ++read) {
                auto idx = fields[read];
                if (!dead_fields.contains(idx)) {
                    fields[write++] = old_to_new.at(idx);
                }
            }
            fields.resize(write);
        };
        filter_remap(edge.load_fields);
        filter_remap(edge.store_fields);
    }

    info.frame_field_count = result.remaining_field_count;

    desc.from_materialize_info(info);
    result.new_frame_size = desc.total_size();

    return result;
}

}// namespace luisa::compute::xir
