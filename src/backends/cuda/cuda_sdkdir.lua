import("lib.detect.find_file")
function main(version)
	-- init the search directories
	local paths = {}
	if version then
		if is_host("macosx") then
			table.insert(paths, format("/Developer/NVIDIA/CUDA-%s/bin", version))
		elseif is_host("windows") then
			table.insert(paths, format("$(env CUDA_PATH_V%s)/bin", version:gsub("%.", "_")))
			table.insert(paths, format("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s\\bin", version))
		else
			table.insert(paths, format("/usr/local/cuda-%s/bin", version))
		end
	else
		if is_host("macosx") then
			table.insert(paths, "/Developer/NVIDIA/CUDA/bin")
			table.insert(paths, "/Developer/NVIDIA/CUDA*/bin")
		elseif is_host("windows") then
			table.insert(paths, "$(env CUDA_PATH)/bin")
			table.insert(paths, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*\\bin")
		else
			-- find from default symbol link dir
			table.insert(paths, "/usr/local/cuda/bin")
			table.insert(paths, "/usr/local/cuda*/bin")
		end
		table.insert(paths, "$(env PATH)")
	end

	-- attempt to find nvcc
	local nvcc = find_file(is_host("windows") and "nvcc.exe" or "nvcc", paths)
	if nvcc then
		return path.directory(path.directory(nvcc))
	end
end