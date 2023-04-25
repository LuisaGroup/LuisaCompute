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
function _string_builder:add_capacity(s)
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
local function max(a, b)
	if a > b then
		return a
	end
	return b
end
function _string_builder:add(str)
	if #str == 0 then
		return
	end
	self:add_capacity(#str)
	local ptr = self._ptr + self._size + 1
	libc.memcpy(ptr, libc.dataptr(str), #str)
	self._size = self._size + #str
	return self
end
function _string_builder:add_char(c)
	self:add_capacity(1)
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
	self._size = max(self._size - i, 1)
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
function StringBuilder(str)
	local inst = table.inherit(_string_builder)
	if str then
		local capa = max(32, #str)
		local addr = libc.malloc(capa)
		inst._size = #str
		inst._capacity = capa
		inst._ptr = addr - 1
		libc.memcpy(addr, libc.dataptr(str), #str)
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
