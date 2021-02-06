//
// Created by Mike Smith on 2021/2/6.
//

#include <sstream>
#include <fmt/format.h>
#include <core/hash.h>
#include "type.h"

namespace luisa {

std::unordered_map<std::string, std::unique_ptr<TypeInfo>> TypeInfo::_description_to_info;

uint32_t TypeInfo::_compute_hash() const noexcept {
    auto desc = _compute_description_string();
    return xxh32_hash32(desc.c_str(), desc.size());
}

std::string TypeInfo::_compute_description_string() const noexcept {
    if (_tag == TypeTag::BOOL) { return "bool"; }
    if (_tag == TypeTag::FLOAT) { return "float"; }
    if (_tag == TypeTag::INT8) { return "char"; }
    if (_tag == TypeTag::UINT8) { return "uchar"; }
    if (_tag == TypeTag::INT16) { return "short"; }
    if (_tag == TypeTag::UINT16) { return "ushort"; }
    if (_tag == TypeTag::INT32) { return "int"; }
    if (_tag == TypeTag::UINT32) { return "uint"; }
    if (_tag == TypeTag::VECTOR) { return fmt::format(FMT_STRING("vector<{},{}>"), element()->_compute_description_string(), element_count()); }
    if (_tag == TypeTag::MATRIX) { return fmt::format(FMT_STRING("matrix<{},{}>"), element()->_compute_description_string(), element_count()); }
    if (_tag == TypeTag::ARRAY) { return fmt::format(FMT_STRING("array<{},{}>"), element()->_compute_description_string(), element_count()); }
    if (_tag == TypeTag::ATOMIC) { return fmt::format(FMT_STRING("atomic<{}>"), element()->_compute_description_string()); }
    if (_tag == TypeTag::STRUCTURE) {
        std::ostringstream os;
        os << "struct<" << _alignment;
        for (auto i = 0u; i < element_count(); i++) { os << "," << _members[i]->_compute_description_string(); }
        os << ">";
        return os.str();
    }
    return "unknown";
}

}
