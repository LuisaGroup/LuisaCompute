#include <Utility/BinaryReader.h>
BinaryReader::BinaryReader(vstd::string const& path) {
	currentPos = 0;
	ifs = fopen(path.c_str(), "rb");
	isAvaliable = ifs;
	if (isAvaliable) {
		fseek(ifs, 0, SEEK_END);
		length = ftell(ifs);
		fseek(ifs, 0, SEEK_SET);
	} else {
		length = 0;
	}
}
void BinaryReader::Read(char* ptr, uint64 len) {
	if (!isAvaliable) return;
	uint64 targetEnd = currentPos + len;
	if (targetEnd > length) {
		targetEnd = length;
		len = targetEnd - currentPos;
	}
	uint64 lastPos = currentPos;
	currentPos = targetEnd;
	if (len == 0) return;
	fseek(ifs, lastPos, SEEK_SET);
	fread(ptr, len, 1, ifs);
}
void BinaryReader::SetPos(uint64 pos) {
	if (!isAvaliable) return;
	if (pos > length) pos = length;
	currentPos = pos;
}
BinaryReader::~BinaryReader() {
	if (!isAvaliable) return;
	fclose(ifs);
}
