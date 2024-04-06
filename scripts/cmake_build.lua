local lib = import('lib')
local function find_msvc()
    import("detect.sdks.find_vstudio")
    local tool = find_vstudio()
    local max_version = 0
    local key = nil
    for version, dict in pairs(tool) do
        local ver_num = tonumber(version)
        if ver_num >= max_version then
            max_version = ver_num
            key = version
        end
        break
    end
    if not key then
        target:set_enabled(false)
        utils.error("Can not find Visual Studio. lcub disabled.")
        return
    end
    local vcvarsall = tool[key]["vcvarsall"]
    local vs_dict
    if vcvarsall then
        if os.is_arch("x64") then
            vs_dict = vcvarsall["x64"]
        elseif os.is_arch("arm64") then
            vs_dict = vcvarsall["arm64"]
        end
    end
    if not vs_dict then
        target:set_enabled(false)
        utils.error("Can not find Visual Studio. lcub disabled.")
        return
    end
    os.setenv("PATH", vs_dict["PATH"])
end

local function build(args)
    local compiler, version
    local config = args["config"] or "Release"
    if os.is_host('windows') then
        compiler = args['compiler'] or 'cl'
        find_msvc()
        if compiler == "clang" then
            os.setenv('CC', 'clang')
            os.setenv('CXX', 'clang++')
        else
            os.setenv('CC', compiler)
            os.setenv('CXX', compiler)
        end
        os.exec('cmake -S . -G Ninja -B build -D CMAKE_BUILD_TYPE=' .. config)
    elseif os.is_host('linux') then
        compiler = args['compiler'] or 'gcc'
        version = args['version'] or '13'
        if compiler == 'gcc' then
            os.setenv('LUISA_CC', 'gcc-' .. version)
            os.setenv('LUISA_CXX', 'g++-' .. version)
            os.setenv('LUISA_FLAGS', '')
        elseif compiler == "clang" then
            os.setenv('LUISA_CC', 'clang-' .. version)
            os.setenv('LUISA_CXX', 'clang++-' .. version)
            os.setenv('LUISA_FLAGS', '-stdlib=libc++')
        end
        os.exec('cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=' .. config ..
                    ' -D CMAKE_C_COMPILER=${LUISA_CC} -D CMAKE_CXX_COMPILER=${LUISA_CXX} -D CMAKE_CXX_FLAGS="${LUISA_FLAGS}"')
    elseif os.is_host('macosx') then
        compiler = args['compiler'] or 'homebrew-clang'
        if compiler == 'homebrew-clang' then
            os.setenv('PATH', '/usr/local/opt/llvm/bin:$PATH')
        end
        os.exec('cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=' .. config ..
                    ' -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D LUISA_COMPUTE_ENABLE_UNITY_BUILD=OFF')
    else
        utils.error('Unsupported platform.')
        return
    end
    os.exec('cmake --build build')
end
function main(...)
    local args = {}
    for i, v in ipairs({...}) do
        local kv = lib.string_split(v, '=')
        if table.getn(kv) == 2 then
            args[kv[1]] = kv[2]
        end
    end

    try {function()
        build(args)
    end, catch {function(errors)
        print(errors)
    end}}

end
