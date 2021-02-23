#include "LCTypeProcessor.h"
#include <ast/type.h>
namespace luisa::compute {

vengine::string const& SerializedStruct::GetName() const {
	return name;
}

uint SerializedStruct::GetVarCount() const {
	return decl.size();
}

SerializedStruct::Declare const& SerializedStruct::GetVar(uint count) const {
	return decl[count];
}

SerializedStruct::~SerializedStruct() {
}
SerializedStruct const& TypeProcessor::GetSerializedStruct(Type const& type) {
	auto ite = map.Find(&type);
	if (ite)
		return ite.Value();
	ite = map.Insert(&type);

	return ite.Value();
}
TypeProcessor::TypeProcessor() {
}
TypeProcessor::~TypeProcessor() {
}
}// namespace luisa::compute