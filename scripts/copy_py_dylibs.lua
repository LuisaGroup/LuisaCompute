function main()
	local my_is_host
	if type(os.is_host) == "function" then
		my_is_host = os.is_host
	elseif type(os.host) == "function" then
		my_is_host = function(p)
			return os.host() == p
		end
	else
		my_is_host = is_host
	end
	if my_is_host("windows") then
		local dst = path.join(os.scriptdir(), "../src/py/luisa/dylibs")
		os.mkdir(dst)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.dll"), dst)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.pyd"), dst)
	end
end
