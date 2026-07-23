target("lc-spirv-llvm")
set_basename("luisa-spirv-llvm")
_config_project({
    project_kind = "static"
})
add_deps("lc-vstl", "lc-runtime", "spirv-tools")
add_files("*.cpp")

on_config(function(target)
    local llvm_root = get_config("lc_llvm_path")
    if type(llvm_root) ~= "string" or llvm_root:len() == 0 then
        raise("lc-spirv-llvm requires lc_llvm_path to name an LLVM installation.")
    end

    -- Resolve the matching llvm-config instead of guessing whether the user
    -- supplied an install prefix, a build tree, or an llvm-project checkout.
    local executable = target:is_plat("windows") and
                       "llvm-config.exe" or "llvm-config"
    local candidate_roots = {}
    if is_mode("debug") then
        table.insert(candidate_roots, path.join(llvm_root, "build", "debug"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "Debug"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "release"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "Release"))
    else
        table.insert(candidate_roots, path.join(llvm_root, "build", "release"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "Release"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "debug"))
        table.insert(candidate_roots, path.join(llvm_root, "build", "Debug"))
    end
    table.insert(candidate_roots, path.join(llvm_root, "build"))
    table.insert(candidate_roots, llvm_root)

    local llvm_config
    if os.isfile(llvm_root) then
        llvm_config = llvm_root
    else
        for _, candidate_root in ipairs(candidate_roots) do
            local candidate = path.join(candidate_root, "bin", executable)
            if os.isfile(candidate) then
                llvm_config = candidate
                break
            end
        end
    end
    if not llvm_config then
        raise("Could not find " .. executable .. " below lc_llvm_path=" ..
              llvm_root .. ". Pass an LLVM install/build prefix (or the " ..
              "llvm-config executable itself).")
    end

    local function llvm_config_output(arguments, description)
        local ok, output = pcall(os.iorunv, llvm_config, arguments)
        if not ok then
            raise("llvm-config failed while querying " .. description ..
                  " for lc-spirv-llvm: " .. tostring(output))
        end
        return output:match("^%s*(.-)%s*$")
    end

    local function llvm_config_arguments(options, components)
        local arguments = {"--quote-paths"}
        if type(options) == "table" then
            for _, option in ipairs(options) do
                table.insert(arguments, option)
            end
        else
            table.insert(arguments, options)
        end
        if components then
            for _, component in ipairs(components) do
                table.insert(arguments, component)
            end
        end
        return arguments
    end

    local function llvm_config_argv(arguments, description)
        return os.argv(llvm_config_output(arguments, description))
    end

    local targets_built = llvm_config_output({"--targets-built"}, "built targets")
    local has_spirv_target = false
    for _, built_target in ipairs(os.argv(targets_built)) do
        if built_target == "SPIRV" then
            has_spirv_target = true
            break
        end
    end
    if not has_spirv_target then
        raise("The LLVM installation at " .. llvm_root ..
              " was not built with the native SPIR-V target.")
    end

    if target:is_plat("windows") then
        local llvm_build_mode = llvm_config_output(
            {"--build-mode"}, "LLVM build mode"):lower()
        local llvm_is_debug = llvm_build_mode == "debug"
        local llvm_is_release = llvm_build_mode == "release" or
                                llvm_build_mode == "relwithdebinfo" or
                                llvm_build_mode == "minsizerel"
        if not llvm_is_debug and not llvm_is_release then
            raise("Cannot determine whether the selected LLVM build is Debug " ..
                  "or Release from llvm-config --build-mode=" .. llvm_build_mode ..
                  "; refusing a potentially incompatible MSVC STL boundary.")
        end
        if llvm_is_debug ~= is_mode("debug") then
            raise("The selected LLVM build mode (" .. llvm_build_mode ..
                  ") does not match the LuisaCompute mode (" ..
                  (is_mode("debug") and "debug" or "non-debug") ..
                  "). Mixing MSVC iterator-debug levels across the LLVM " ..
                  "facade is unsafe.")
        end

        -- llvm-config does not expose CMAKE_MSVC_RUNTIME_LIBRARY directly,
        -- but its adjacent package config does. Compare MD/MT families instead
        -- of allowing LLVMConfig's setting to leak into the rest of the graph.
        local llvm_cmake_dir = llvm_config_output(
            {"--cmakedir"}, "LLVM CMake package directory")
        local llvm_cmake_config_file = path.join(
            llvm_cmake_dir, "LLVMConfig.cmake")
        local llvm_cmake_config = io.readfile(llvm_cmake_config_file)
        if not llvm_cmake_config then
            raise("Cannot read " .. llvm_cmake_config_file ..
                  " to verify the MSVC runtime-library ABI.")
        end
        local llvm_runtime = llvm_cmake_config:match(
            "set%s*%(%s*CMAKE_MSVC_RUNTIME_LIBRARY%s+([^%)]*)%)")
        if llvm_runtime then
            llvm_runtime = llvm_runtime:match('^%s*"(.-)"%s*$') or
                           llvm_runtime:match("^%s*(.-)%s*$")
        end
        if not llvm_runtime or llvm_runtime:len() == 0 then
            raise("LLVMConfig.cmake does not report CMAKE_MSVC_RUNTIME_LIBRARY; " ..
                  "refusing an unverifiable C++ ABI boundary.")
        end
        local project_runtime = target:get("runtimes")
        if type(project_runtime) == "table" then
            project_runtime = project_runtime[#project_runtime]
        end
        if type(project_runtime) ~= "string" or
           project_runtime:len() == 0 then
            raise("LuisaCompute does not report its MSVC runtime selection; " ..
                  "refusing an unverifiable C++ ABI boundary.")
        end
        local llvm_uses_dll_crt = llvm_runtime:find("DLL", 1, true) ~= nil
        local project_uses_dll_crt =
            project_runtime:upper():match("^MD") ~= nil
        if llvm_uses_dll_crt ~= project_uses_dll_crt then
            raise("LLVM and LuisaCompute use different MSVC runtime-library " ..
                  "families: LLVM='" .. llvm_runtime .. "', project='" ..
                  project_runtime .. "'.")
        end
    end

    local llvm_include = llvm_config_output({"--includedir"}, "include directory")
    local llvm_libdir = llvm_config_output({"--libdir"}, "library directory")
    local llvm_include_dirs = {llvm_include}
    local seen_include_dirs = {[llvm_include] = true}
    local function add_llvm_include(include_dir)
        if include_dir and include_dir:len() ~= 0 and
           not seen_include_dirs[include_dir] then
            seen_include_dirs[include_dir] = true
            table.insert(llvm_include_dirs, include_dir)
        end
    end

    local llvm_glibcxx_abi
    local cppflags = llvm_config_argv(
        llvm_config_arguments("--cppflags"), "preprocessor flags")
    local cppflag_index = 1
    while cppflag_index <= #cppflags do
        local flag = cppflags[cppflag_index]
        local include_dir = flag:match("^-I(.+)$")
        local definition = flag:match("^[-/]D(.+)$")
        if flag == "-I" and cppflag_index < #cppflags then
            cppflag_index = cppflag_index + 1
            add_llvm_include(cppflags[cppflag_index])
        elseif include_dir then
            add_llvm_include(include_dir)
        elseif definition then
            local abi = definition:match("^_GLIBCXX_USE_CXX11_ABI=([01])$")
            if abi then
                if llvm_glibcxx_abi and llvm_glibcxx_abi ~= abi then
                    raise("llvm-config reported conflicting libstdc++ ABIs.")
                end
                llvm_glibcxx_abi = abi
            else
                target:add("defines", definition)
            end
        end
        cppflag_index = cppflag_index + 1
    end

    local spirv_intrinsics_found = false
    for _, include_dir in ipairs(llvm_include_dirs) do
        if os.isfile(path.join(include_dir, "llvm", "IR", "IntrinsicsSPIRV.h")) then
            spirv_intrinsics_found = true
            break
        end
    end
    if not spirv_intrinsics_found then
        raise("The LLVM installation at " .. llvm_root ..
              " does not provide llvm/IR/IntrinsicsSPIRV.h in the include " ..
              "directories reported by llvm-config --cppflags.")
    end
    for _, include_dir in ipairs(llvm_include_dirs) do
        target:add("includedirs", include_dir)
    end
    target:add("includedirs", path.join(os.scriptdir(), ".."), {public = true})
    target:add("linkdirs", llvm_libdir, {public = true})

    if llvm_glibcxx_abi then
        local snippets = {}
        snippets["llvm_glibcxx_abi_" .. llvm_glibcxx_abi] = string.format([[
            #include <string>
            #if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != %s
            #error incompatible libstdc++ ABI
            #endif
            void llvm_glibcxx_abi_probe() {}
        ]], llvm_glibcxx_abi)
        if not target:check_cxxsnippets(snippets) then
            raise("LLVM uses _GLIBCXX_USE_CXX11_ABI=" .. llvm_glibcxx_abi ..
                  ", but the LuisaCompute toolchain uses a different C++ " ..
                  "standard-library ABI.")
        end
        -- Propagate only the ABI macro across the LLVM-free facade. All other
        -- LLVM definitions and include paths remain implementation details.
        target:add("defines", "_GLIBCXX_USE_CXX11_ABI=" .. llvm_glibcxx_abi,
                   {public = true})
    end

    local components = {
        "core", "support", "bitwriter", "transformutils", "analysis",
        "codegen", "target", "mc", "spirvcodegen", "spirvdesc",
        "spirvinfo", "spirvanalysis"
    }
    local shared_mode = llvm_config_output(
        llvm_config_arguments("--shared-mode", components),
        "collective LLVM component link mode")
    if shared_mode ~= "shared" and shared_mode ~= "static" then
        raise("llvm-config --shared-mode returned unexpected value: " .. shared_mode)
    end
    -- llvm-config reports DLL filenames rather than import libraries for
    -- --link-shared on MSVC. Static linkage is the only portable mode here;
    -- it also avoids an undeclared runtime-DLL staging obligation.
    if target:is_plat("windows") then
        shared_mode = "static"
    end
    local llvm_link_kind = shared_mode == "shared" and
                           "--link-shared" or "--link-static"
    local library_names = llvm_config_argv(
        llvm_config_arguments(
            {llvm_link_kind, "--libnames"}, components),
        "LLVM SPIR-V component libraries")
    if #library_names == 0 then
        raise("llvm-config did not report linkable LLVM SPIR-V libraries.")
    end
    if llvm_link_kind == "--link-shared" and
       not target:is_plat("windows") then
        target:add("rpathdirs", llvm_libdir, {public = true})
        target:add("rpathdirs", llvm_libdir,
                   {public = true, installonly = true})
    end
    for _, filename in ipairs(library_names) do
        local link = path.filename(filename)
        link = link:gsub("^lib", "")
        link = link:gsub("%.so.*$", "")
        link = link:gsub("%.dylib$", "")
        link = link:gsub("%.dll$", "")
        link = link:gsub("%.a$", "")
        link = link:gsub("%.lib$", "")
        target:add("links", link, {public = true})
    end

    local system_libraries = llvm_config_argv(
        llvm_config_arguments(
            {llvm_link_kind, "--system-libs"}, components),
        "LLVM component system libraries")
    local system_library_index = 1
    while system_library_index <= #system_libraries do
        local flag = system_libraries[system_library_index]
        local library = flag:match("^-l(.+)$")
        if library then
            target:add("syslinks", library, {public = true})
        elseif flag == "-framework" and
               system_library_index < #system_libraries then
            system_library_index = system_library_index + 1
            target:add("frameworks", system_libraries[system_library_index],
                       {public = true})
        elseif flag:match("%.lib$") then
            local library_dir = path.directory(flag)
            if library_dir and library_dir ~= "." then
                target:add("linkdirs", library_dir, {public = true})
            end
            target:add("syslinks", path.basename(flag), {public = true})
        else
            target:add("ldflags", flag, {public = true, force = true})
        end
        system_library_index = system_library_index + 1
    end
end)

target_end()
