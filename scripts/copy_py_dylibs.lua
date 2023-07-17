function main()
	if os.is_host("windows") then
		local dst = path.join(os.scriptdir(), "../src/py/luisa/dylibs")
		os.mkdir(dst)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.dll"), dst)
		os.cp(path.join(os.scriptdir(), "../bin/release/*.pyd"), dst)
	end
end
