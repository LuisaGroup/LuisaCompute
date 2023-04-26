function string_split(str, chr)
	local map = {}
	for part in string.gmatch(str, "([^" .. chr .. "]+)") do
		table.insert(map, part)
	end
	return map
end
function string_replace(str, from, to)
	local s, _ = str:gsub(from, to)
	return s
end

function string_contains(str, sub_str)
	return str:match(sub_str) ~= nil
end
local libc = import("core/base/libc")
local bytes = import("core/base/bytes")
local _string_builder = {}
function _string_builder:to_string()
	return libc.strndup(self._ptr + 1, self._size)
end
local function _add_capacity(self, s)
	local size = s + self._size
	local capa = self._capacity
	if capa >= size then
		return
	end
	while capa < size do
		capa = capa * 2
	end
	local old_ptr = self._ptr + 1
	local new_ptr = libc.malloc(capa)
	libc.memcpy(new_ptr, old_ptr, self._size)
	libc.free(old_ptr)
	self._ptr = new_ptr - 1
	self._capacity = capa
end
function _string_builder:reserve(s)
	local capa = self._capacity
	if capa >= s then
		return
	end
	local old_ptr = self._ptr + 1
	local new_ptr = libc.malloc(s)
	libc.memcpy(new_ptr, old_ptr, self._size)
	libc.free(old_ptr)
	self._ptr = new_ptr - 1
	self._capacity = s
end
function _string_builder:equal(str)
	local str_ptr
	local str_size
	if type(str) == "string" then
		str_ptr = libc.dataptr(str)
		str_size = #str
	else
		str_ptr = str:caddr()
		str_size = str:size()
	end
	if str_size ~= self.size() then
		return false
	end
	local ptr = self._ptr + self._size + 1
	return libc.memcmp(ptr, str_ptr, str_size) == 0
end
function _string_builder:add(str)
	local str_ptr
	local str_size
	if type(str) == "string" then
		str_ptr = libc.dataptr(str)
		str_size = #str
	else
		str_ptr = str:caddr()
		str_size = str:size()
	end
	if str_size == 0 then
		return
	end
	_add_capacity(self, str_size)
	local ptr = self._ptr + self._size + 1
	libc.memcpy(ptr, str_ptr, str_size)
	self._size = self._size + str_size
	return self
end
function _string_builder:subview(offset, size)
	local sf = self
	return {
		_size = math.min(sf._size - offset + 1, size),
		_ptr = sf._ptr + offset,
		size = function(self)
			return self._size
		end,
		caddr = function(self)
			return self._ptr
		end
	}
end
function _string_builder:add_char(c)
	_add_capacity(self, 1)
	self._size = self._size + 1
	libc.setbyte(self._ptr, self._size, c)
	return self
end
function _string_builder:dispose()
	if self._ptr ~= -1 then
		libc.free(self._ptr + 1)
		self._ptr = -1
	end
end
function _string_builder:write_to(path)
	local f = io.open(path, "wb")
	f:write(self)
	f:close()
end
function _string_builder:get(i)
	return libc.byteof(self._ptr, i)
end
function _string_builder:set(i, v)
	return libc.setbyte(self._ptr, i, v)
end
function _string_builder:erase(i)
	self._size = math.max(self._size - i, 1)
end
function _string_builder:size()
	return self._size
end
function _string_builder:capacity()
	return self._capacity
end
function _string_builder:caddr()
	return self._ptr + 1
end
function _string_builder:clear()
	self._size = 0
end
function StringBuilder(str)
	local inst = table.inherit(_string_builder)
	if str then
		local str_ptr
		local str_size
		if type(str) == "string" then
			str_ptr = libc.dataptr(str)
			str_size = #str
		else
			str_ptr = str:caddr()
			str_size = str:size()
		end
		local capa = math.max(32, str_size)
		local addr = libc.malloc(capa)
		inst._size = str_size
		inst._capacity = capa
		inst._ptr = addr - 1
		libc.memcpy(addr, str_ptr, str_size)
	else
		inst._size = 0
		inst._capacity = 32
		inst._ptr = libc.malloc(32) - 1
	end
	return inst
end
function char(str)
	return libc.byteof(libc.dataptr(str), 0)
end
function to_hex_array(input, out)
	local cut = char(',')
	local hex_table = {char('0'), char('1'), char('2'), char('3'), char('4'), char('5'), char('6'), char('7'), char('8'),
                    char('9'), char('a'), char('b'), char('c'), char('d'), char('e'), char('f')}

	local str_ptr
	local str_size
	if type(input) == "string" then
		str_ptr = libc.dataptr(input)
		str_size = #input
	else
		str_ptr = input:caddr()
		str_size = input:size()
	end
	local byte_count = 0
	for idx = 0, (str_size - 1) do
		if byte_count == 0 then
			out:add("0x")
		end
		local i = libc.byteof(str_ptr, idx)
		local low_bit = i >> 4
		local high_bit = (i & 15)
		out:add_char(hex_table[low_bit + 1])
		out:add_char(hex_table[high_bit + 1])
		byte_count = byte_count + 1
		if byte_count >= 4 then
			byte_count = 0
			out:add_char(cut)
		end
	end
	if byte_count > 0 then
		for i = byte_count, 3 do
			out:add("00")
		end
	end
	if out:get(out:size()) == cut then
		out:erase(1)
	end
end
