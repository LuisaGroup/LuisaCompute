#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/dead_field_elimination.h>

namespace luisa::compute::xir {

namespace {

static constexpr size_t FRAME_RESERVED_FIELD_COUNT = CoroFrameDesc::reserved_field_count;

[[nodiscard]] bool validate_materialize_info(const CoroMaterializeInfo &info,
                                             const CoroFrameDesc &desc) noexcept {
    if (!info.succeeded()) { return false; }
    if (info.frame_field_count == 0u) {
        if (!info.frame_fields.empty() || !info.name_to_field.empty() ||
            !info.name_to_type.empty() || desc.field_count() != 0u) {
            return false;
        }
        for (auto &edge : info.edges) {
            if (!edge.load_fields.empty() || !edge.store_fields.empty()) { return false; }
        }
        return true;
    }
    if (info.frame_field_count < FRAME_RESERVED_FIELD_COUNT) { return false; }
    auto user_field_count = info.frame_field_count - FRAME_RESERVED_FIELD_COUNT;
    luisa::vector<const Type *> indexed_types(user_field_count, nullptr);
    luisa::vector<luisa::string> indexed_names(user_field_count);
    luisa::unordered_set<size_t> seen_indices;
    if (info.name_to_field.size() != info.name_to_type.size()) {
        return false;
    }
    if (!info.frame_fields.empty()) {
        if (info.frame_fields.size() != user_field_count) { return false; }
        for (auto &field : info.frame_fields) {
            if (field.index < FRAME_RESERVED_FIELD_COUNT ||
                field.index >= info.frame_field_count || field.type == nullptr ||
                !seen_indices.emplace(field.index).second) {
                return false;
            }
            auto local_index = field.index - FRAME_RESERVED_FIELD_COUNT;
            indexed_names[local_index] = field.name;
            indexed_types[local_index] = field.type;
            auto field_iter = info.name_to_field.find(field.name);
            auto type_iter = info.name_to_type.find(field.name);
            if (field_iter == info.name_to_field.end() || field_iter->second != field.index ||
                type_iter == info.name_to_type.end() || type_iter->second != field.type) {
                return false;
            }
        }
        // Logical frame names may alias an interference-colored physical
        // field, but every alias must name that field's exact type.
        for (auto &[name, index] : info.name_to_field) {
            auto type_iter = info.name_to_type.find(name);
            if (index < FRAME_RESERVED_FIELD_COUNT ||
                index >= info.frame_field_count ||
                type_iter == info.name_to_type.end() ||
                type_iter->second !=
                    indexed_types[index - FRAME_RESERVED_FIELD_COUNT]) {
                return false;
            }
        }
    } else {
        // Legacy metadata without an explicit physical-field table is
        // necessarily one-name-per-field; aliases would make its layout
        // ambiguous.
        if (info.name_to_field.size() != user_field_count) {
            return false;
        }
        for (auto &[name, index] : info.name_to_field) {
            auto type_iter = info.name_to_type.find(name);
            if (index < FRAME_RESERVED_FIELD_COUNT || index >= info.frame_field_count ||
                type_iter == info.name_to_type.end() || type_iter->second == nullptr ||
                !seen_indices.emplace(index).second) {
                return false;
            }
            auto local_index = index - FRAME_RESERVED_FIELD_COUNT;
            indexed_names[local_index] = name;
            indexed_types[local_index] = type_iter->second;
        }
    }
    if (seen_indices.size() != user_field_count || desc.field_count() != user_field_count) { return false; }
    for (size_t i = 0u; i < user_field_count; ++i) {
        if (indexed_types[i] == nullptr || desc.field(i).name != indexed_names[i] ||
            desc.field(i).type != indexed_types[i]) {
            return false;
        }
    }
    for (auto &edge : info.edges) {
        for (auto index : edge.load_fields) {
            if (index >= info.frame_field_count) { return false; }
        }
        for (auto index : edge.store_fields) {
            if (index >= info.frame_field_count) { return false; }
        }
    }
    return true;
}

}// namespace

DeadFieldEliminationInfo dead_field_elimination_pass_run(
    CoroMaterializeInfo &info,
    CoroFrameDesc &desc) noexcept {

    DeadFieldEliminationInfo result;
    result.original_field_count = info.frame_field_count;
    result.original_frame_size = desc.total_size();
    if (!validate_materialize_info(info, desc)) {
        result.invalid_input_error_count = 1u;
        result.remaining_field_count = info.frame_field_count;
        result.new_frame_size = result.original_frame_size;
        return result;
    }

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
        info.name_to_type.erase(name);
    }

    if (!info.frame_fields.empty()) {
        size_t write = 0u;
        for (size_t read = 0u; read < info.frame_fields.size(); ++read) {
            auto field = std::move(info.frame_fields[read]);
            if (dead_fields.contains(field.index)) { continue; }
            field.index = old_to_new.at(field.index);
            info.frame_fields[write++] = std::move(field);
        }
        info.frame_fields.resize(write);
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
