#pragma once
#include "../Common/Common.h"
namespace luisa::compute {
class Type;
class SerializedStruct {
public:
	class Declare {
	public:
		vengine::string typeName;
		vengine::string varName;
	};
	vengine::string const& GetName() const;
	uint GetVarCount() const;
	Declare const& GetVar(uint count) const;
	~SerializedStruct();

private:
	vengine::string name;
	vengine::vector<Declare> decl;
};
class TypeProcessor {
public:
	SerializedStruct const& GetSerializedStruct(Type const& type);
	TypeProcessor();
	~TypeProcessor();

private:
	HashMap<Type const*, SerializedStruct> map;
};
}// namespace luisa::compute