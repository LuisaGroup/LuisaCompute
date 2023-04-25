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
local function _to_string(self)
	return libc.strndup(self._ptr + 1, self._size)
end
local function add_capacity(self, s)
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
local function _reserve(self, s)
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
local function _add(self, str)
	if #str == 0 then
		return
	end
	add_capacity(self, #str)
	local ptr = self._ptr + self._size + 1
	libc.memcpy(ptr, libc.dataptr(str), #str)
	self._size = self._size + #str
	return self
end
local function _add_char(self, c)
	add_capacity(self, 1)
	self._size = self._size + 1
	libc.setbyte(self._ptr, self._size, c)
	return self
end
local function _dispose(self)
	if self._ptr ~= -1 then
		libc.free(self._ptr + 1)
		self._ptr = -1
	end
end
local function _write_to(self, path)
	local f = io.open(path, "wb")
	f:write(self)
	f:close()
end
local function _get(self, i)
	return libc.byteof(self._ptr, i)
end
local function _set(self, i, v)
	return libc.setbyte(self._ptr, i, v)
end
local function _erase(self, i)
	self._size = max(self._size - i, 1)
end
function StringBuilder(str)
	local inst
	if str then
		local capa = max(32, #str)
		local addr = libc.malloc(capa)
		inst = {
			_size = #str,
			_capacity = capa,
			_ptr = addr - 1
		}
		libc.memcpy(addr, libc.dataptr(str), #str)
	else
		inst = {
			_size = 0,
			_capacity = 32,
			_ptr = libc.malloc(32) - 1
		}
	end
	inst.capacity = function(self)
		return self._capacity
	end
	inst.caddr = function(self)
		return self._ptr + 1
	end
	inst.size = function(self)
		return self._size
	end
	inst.to_string = _to_string
	inst.reserve = _reserve
	inst.add = _add
	inst.dispose = _dispose
	inst.__gc = _dispose
	inst.write_to = _write_to
	inst.get = _get
	inst.set = _set
	inst.add_char = _add_char
	inst.erase = _erase
	return inst
end
function char(str)
	return libc.byteof(libc.dataptr(str), 0)
end
