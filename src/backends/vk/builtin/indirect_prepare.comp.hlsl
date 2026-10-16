#define LC_INDIRECT_LAYOUT(shader_name, cpp_name, value) \
    static const uint LC_INDIRECT_##shader_name = value;
#include "indirect_dispatch_layout.def"
#undef LC_INDIRECT_LAYOUT

struct IndirectPrepareConstants {
    uint command_count;
    uint source_record_offset;
    uint target_block_size_x;
    uint target_block_size_y;
    uint target_block_size_z;
    uint max_group_count_x;
    uint max_group_count_y;
    uint max_group_count_z;
    uint command_base;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

[[vk::push_constant]] ConstantBuffer<IndirectPrepareConstants> pc;
StructuredBuffer<uint> source_records : register(t0);
RWStructuredBuffer<uint> commands : register(u1);

uint ceil_div(uint value, uint divisor) {
    return value / divisor + (value % divisor != 0u);
}

[numthreads(LC_INDIRECT_PREPARE_BLOCK_SIZE, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint remaining = pc.command_count - pc.command_base;
    if (dispatch_id.x >= remaining) { return; }
    uint command_index = pc.command_base + dispatch_id.x;
    uint3 group_count = uint3(0, 0, 0);
    uint source_index = pc.source_record_offset + command_index;
    if (command_index < source_records[0]) {
        uint logical_word = LC_INDIRECT_HEADER_WORDS +
                            source_index * LC_INDIRECT_RECORD_WORDS +
                            LC_INDIRECT_LOGICAL_WORD;
        uint3 logical_size = uint3(
            source_records[logical_word],
            source_records[logical_word + 1],
            source_records[logical_word + 2]);
        uint3 block_size = uint3(
            pc.target_block_size_x,
            pc.target_block_size_y,
            pc.target_block_size_z);
        uint group_word = LC_INDIRECT_HEADER_WORDS +
                          source_index * LC_INDIRECT_RECORD_WORDS +
                          LC_INDIRECT_GROUP_WORD;
        uint3 authored_group_count = uint3(
            source_records[group_word],
            source_records[group_word + 1],
            source_records[group_word + 2]);
        if (all(authored_group_count != uint3(0, 0, 0))) {
            group_count = uint3(
                ceil_div(logical_size.x, block_size.x),
                ceil_div(logical_size.y, block_size.y),
                ceil_div(logical_size.z, block_size.z));
        }
        if (group_count.x > pc.max_group_count_x ||
            group_count.y > pc.max_group_count_y ||
            group_count.z > pc.max_group_count_z) {
            group_count = uint3(0, 0, 0);
        }
    }
    uint command_word = command_index * LC_INDIRECT_COMMAND_WORDS;
    commands[command_word] = group_count.x;
    commands[command_word + 1] = group_count.y;
    commands[command_word + 2] = group_count.z;
}
