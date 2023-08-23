-- Global config
option("_lc_enable_py")
set_showmenu(false)
set_default(false)
option_end()

option("_lc_enable_rust")
set_showmenu(false)
set_default(false)
option_end()

option("_lc_vk_path")
set_default(false)
set_showmenu(false)
option_end()

option("_lc_config_public")
set_default(false)
set_showmenu(false)
before_check(function(option)
	option:set_value(
		path.absolute(path.join(os.projectdir(), "scripts")) ~= path.absolute(os.scriptdir()))
end)
option_end()

option("_lc_bin_dir")
set_default(false)
set_showmenu(false)
add_deps("enable_mimalloc", "enable_unity_build", "enable_simd", "dx_backend", "vk_backend", "cuda_backend",
				"metal_backend", "cpu_backend", "enable_tests", "py_include", "py_linkdir", "py_libs",
				"enable_ir", "enable_api", "enable_dsl", "enable_gui", "bin_dir", "_lc_enable_py", "_lc_vk_path", "_lc_enable_rust", "_lc_config_public")
before_check(function(option)
	local v = import("options", {
		try = true,
		anonymous = true
	})
	if v then
		local opt = v.get_options
		if type(opt) == "function" then
			local map = opt()
			for k, v in pairs(map) do
				if v ~= nil then
					option:dep(k):enable(v)
				end
			end
		end
	end

	local enable_tests = option:dep("enable_tests")
	if enable_tests:enabled() then
		option:dep("enable_dsl"):enable(true, {
			force = true
		})
	end
	-- checking python
	local enable_py = option:dep("_lc_enable_py")
	local function non_empty_str(s)
		return type(s) == "string" and s:len() > 0
	end
	if non_empty_str(option:dep("py_include"):enabled()) then
		enable_py:enable(true)
	end
	local is_win = is_plat("windows")
	-- checking dx
	local dx_backend = option:dep("dx_backend")
	if dx_backend:enabled() and not is_win then
		dx_backend:enable(false, {
			force = true
		})
		if dx_backend:enabled() then
			utils.error("DX backend not supported in this platform, force disabled.")
		end
	end
	-- checking metal
	local metal_backend = option:dep("metal_backend")
	if metal_backend:enabled() and not is_plat("macosx") then
		metal_backend:enable(false, {
			force = true
		})
		if metal_backend:enabled() then
			utils.error("Metal backend not supported in this platform, force disabled.")
		end
	end
	-- checking cuda
	local cuda_backend = option:dep("cuda_backend")
	if cuda_backend:enabled() and not (is_win or is_plat("linux")) then
		cuda_backend:enable(false, {
			force = true
		})
		if cuda_backend:enabled() then
			utils.error("CUDA backend not supported in this platform, force disabled.")
		end
	end
	if enable_py:enabled() then
		option:dep("enable_gui"):enable(true, {
			force = true
		})
	end
	if path.absolute(path.join(os.projectdir(), "scripts")) == path.absolute(os.scriptdir()) then
		local bin_dir = option:dep("bin_dir"):enabled()
		if is_mode("debug") then
			bin_dir = path.join(bin_dir, "debug")
		elseif is_mode("releasedbg") then
			bin_dir = path.join(bin_dir, "releasedbg")
		else
			bin_dir = path.join(bin_dir, "release")
		end
		option:set_value(bin_dir)
	else
		option:set_value(false)
	end
	-- checking rust
	import("lib.detect.find_tool")
	local rust_cargo = find_tool("cargo") ~= nil
	if not rust_cargo then
		utils.warning("Cargo not installed, IR module and CPU backend disabled.")
	end
	local enable_ir = option:dep("enable_ir")
	local cpu_backend = option:dep("cpu_backend")
	if not enable_ir:enabled() then
		option:dep("_lc_enable_rust"):set_value(false)
		cpu_backend:enable(false, {
			force = true
		})
	else
		option:dep("_lc_enable_rust"):set_value(rust_cargo)
		if not rust_cargo then
			enable_ir:enable(false)
			cpu_backend:enable(false)
			if enable_ir:enabled() then
				utils.error("Cargo not installed, IR module force disabled.")
				enable_ir:enable(false, {
					force = true
				})
			end
			if cpu_backend:enabled() then
				utils.error("Cargo not installed, CPU backend force disabled.")
				cpu_backend:enable(false, {
					force = true
				})
			end
		end
	end

	-- checking vulkan
	local vk_path = os.getenv("VULKAN_SDK")
	if not vk_path then
		vk_path = os.getenv("VK_SDK_PATH")
		local vk_backend = option:dep("vk_backend")
		if vk_backend:enabled() then
			vk_backend:enable(false, {
				force = true
			})
			if vk_backend:enabled() then
				utils.error("VK backend not supported in this platform, force disabled.")
			end
		end
	else
		option:dep("_lc_vk_path"):set_value(vk_path)
	end
	-- TODO: cpu backend and rust config
end)
option_end()

rule("lc_vulkan")
on_load(function(target)
	local vk_path = get_config("_lc_vk_path")
	if is_plat("linux", "macosx") then
		target:add("linkdirs", path.join(vk_path, "lib"))
		target:add("links", "vulkan")
		target:add("includedirs", path.join(vk_path, "include"))
	else
		target:add("linkdirs", path.join(vk_path, "Lib"))
		target:add("links", "vulkan-1")
		target:add("includedirs", path.join(vk_path, "Include"))
	end
end)
rule_end()
rule("lc_basic_settings")
on_config(function(target)
	if is_plat("linux") then
		local _, cc = target:tool("cxx")
		-- Linux should use -stdlib=libc++
		-- https://github.com/LuisaGroup/LuisaCompute/issues/58
		if (cc == "clang" or cc == "clangxx") then
			target:add("cxflags", "-stdlib=libc++", {
				force = true
			})
			target:add("syslinks", "c++")
		end
	end
end)
on_load(function(target)
	local _get_or = function(name, default_value)
		local v = target:values(name)
		if v == nil then
			return default_value
		end
		return v
	end
	local config_is_public = get_config("_lc_config_public") or false
	local project_kind = _get_or("project_kind", "phony")
	target:set("kind", project_kind)
	if not is_plat("windows") then
		if project_kind == "static" then
			target:add("cxflags", "-fPIC", {
				tools = {"clang", "gcc"}
			})
		end
	end
	local c_standard = target:values("c_standard")
	local cxx_standard = target:values("cxx_standard")
	if type(c_standard) == "string" and type(cxx_standard) == "string" then
		target:set("languages", c_standard, cxx_standard, {
			public = config_is_public
		})
	else
		target:set("languages", "clatest", "cxx20", {
			public = config_is_public
		})
	end

	local enable_exception = _get_or("enable_exception", nil)
	if enable_exception then
		target:set("exceptions", "cxx")
	else
		target:set("exceptions", "no-cxx")
	end

	if is_mode("debug") then
		target:set("runtimes", "MDd", {
			public = config_is_public
		})
		target:set("optimize", "none", {
			public = config_is_public
		})
		target:set("warnings", "none", {
			public = config_is_public
		})
		target:add("cxflags", "/GS", "/Gd", {
			tools = {"clang_cl", "cl"},
			public = config_is_public
		})
		target:add("cxflags", "/Zc:preprocessor", {
			tools = "cl",
			public = config_is_public
		});
	else
		target:set("runtimes", "MD", {
			public = config_is_public
		})
		target:set("optimize", "aggressive", {
			public = config_is_public
		})
		target:set("warnings", "none", {
			public = config_is_public
		})
		target:add("cxflags", "/GS-", "/Gd", {
			tools = {"clang_cl", "cl"},
			public = config_is_public
		})
		target:add("cxflags", "/Zc:preprocessor", {
			tools = "cl",
			public = config_is_public
		})
	end
	if _get_or("use_simd", false) then
		if is_arch("arm64") then
			target:add("vectorexts", "neon", {
				public = config_is_public
			})
		else
			target:add("vectorexts", "avx", "avx2", {
				public = config_is_public
			})
		end
	end
	if _get_or("no_rtti", false) then
		target:add("cxflags", "/GR-", {
			tools = {"clang_cl", "cl"},
			public = config_is_public
		})
		target:add("cxflags", "-fno-rtti", "-fno-rtti-data", {
			tools = {"clang"},
			public = config_is_public
		})
		target:add("cxflags", "-fno-rtti", {
			tools = {"gcc"},
			public = config_is_public
		})
	end
end)
rule_end()

rule("lc-rename-ext")
on_load(function(target)
	target:set("basename", "lc-ext-" .. target:name())
end)
rule_end()

rule("build_cargo")
set_extensions(".toml")
on_buildcmd_file(function(target, batchcmds, sourcefile, opt)
	local lib = import("lib")
	local sb = lib.StringBuilder("cargo build -q ")
	-- if backend_off then
	sb:add("--no-default-features ")
	-- end
	sb:add("--manifest-path ")
	sb:add(sourcefile):add(' ')
	local features = target:get('features')
	if features then
		sb:add("--features ")
		sb:add(features):add(' ')
	end
	if not is_mode("debug") then
		sb:add("--release ")
	end
	local cargo_cmd = sb:to_string()
	print(cargo_cmd)
	batchcmds:vrunv(cargo_cmd)
	sb:dispose()
end)
rule_end()

-- In-case of submod, when there is override rules, do not overload
if _config_rules == nil then
	_config_rules = {"lc_basic_settings"}
end
if _disable_unity_build == nil then
	local unity_build = get_config("enable_unity_build")
	if unity_build ~= nil then
		_disable_unity_build = not unity_build
	end
end

if _configs == nil then
	_configs = {}
end
_configs["use_simd"] = get_config("enable_simd")
if not _config_project then
	function _config_project(config)
		if type(_configs) == "table" then
			for k, v in pairs(_configs) do
				set_values(k, v)
			end
		end
		if type(_config_rules) == "table" then
			add_rules(_config_rules)
		end
		if type(config) == "table" then
			for k, v in pairs(config) do
				set_values(k, v)
			end
		end
		local batch_size = config["batch_size"]
		if type(batch_size) == "number" and batch_size > 1 and (not _disable_unity_build) then
			add_rules("c.unity_build", {
				batchsize = batch_size
			})
			add_rules("c++.unity_build", {
				batchsize = batch_size
			})
		end
	end
end
