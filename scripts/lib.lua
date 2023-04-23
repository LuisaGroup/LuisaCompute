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
