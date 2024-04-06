local lib = import('lib')
local function exec(sb, cmd)
    sb:add(cmd):add("\r\n")
end
local function find_msvc(sb)
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
        utils.error("Can not find Visual Studio.")
        return
    end
    local vcvarsall_bat = tool[key]["vcvarsall_bat"]
    local vcvarsall = tool[key]["vcvarsall"]
    if vcvarsall_bat then
        if os.is_arch("x64") then
            exec(sb, 'call "' .. vcvarsall_bat .. '" x64')
        elseif os.is_arch("arm64") then
            exec(sb, 'call "' .. vcvarsall_bat .. '" arm64')
        else
            utils.error("Can not find Visual Studio.")
            return
        end
    end
    local vs_dict
    if vcvarsall then
        if os.is_arch("x64") then
            vs_dict = vcvarsall["x64"]
        elseif os.is_arch("arm64") then
            vs_dict = vcvarsall["arm64"]
        end
    end
    if not vs_dict then
        utils.error("Can not find Visual Studio.")
        return
    end
    os.setenv("PATH", vs_dict["PATH"])
end

local function build(args)
    local compiler, version
    local config = args["config"] or "Release"
    if os.is_host('windows') then
        local sb = lib.StringBuilder()
        compiler = args['compiler'] or 'cl'
        find_msvc(sb)
        local cc, cxx
        if compiler == "clang" then
            cc = 'clang'
            cxx = 'clang++'
        else
            cc = compiler
            cxx = compiler
        end
        exec(sb, 'cmake -S . -G Ninja -B build -D CMAKE_BUILD_TYPE=' .. config .. ' -D CMAKE_C_COMPILER=' .. cc ..
            ' -D CMAKE_CXX_COMPILER=' .. cxx)
        exec(sb, 'cmake --build build')
        local tmp_file = "_tmp.cmd"
        sb:write_to(tmp_file)
        sb:dispose()
        os.exec(tmp_file)
        os.rm(tmp_file)
    elseif os.is_host('linux') then
        compiler = args['compiler'] or 'gcc'
        version = args['version'] or '13'
        local cc, cxx, flags
        if compiler == 'gcc' then
            cc = 'gcc-' .. version
            cxx = 'g++-' .. version
            flags = '""'
        elseif compiler == "clang" then
            cc = 'clang-' .. version
            cxx = 'clang++-' .. version
            flags = '"-stdlib=libc++"'
        end
        os.exec('cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=' .. config .. ' -D CMAKE_C_COMPILER=' .. cc ..
            ' -D CMAKE_CXX_COMPILER=' .. cxx .. ' -D CMAKE_CXX_FLAGS=' .. flags)
        os.exec('cmake --build build')
    elseif os.is_host('macosx') then
        compiler = args['compiler'] or 'homebrew-clang'
        if compiler == 'homebrew-clang' then
            os.setenv('PATH', '/usr/local/opt/llvm/bin:$PATH')
        end
        os.exec('cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=' .. config ..
            ' -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D LUISA_COMPUTE_ENABLE_UNITY_BUILD=OFF')
        os.exec('cmake --build build')
    else
        utils.error('Unsupported platform.')
        return
    end
end
function main(...)
    local args = {}
    for i, v in ipairs({...}) do
        local kv = lib.string_split(v, '=')
        if table.getn(kv) == 2 then
            local key = kv[1]
            local num = -1
            for i = 1, #key do
                local c = key:sub(i, i)
                if c ~= '-' then
                    num = i
                    break
                end
            end
            if num > 0 then
                args[key:sub(num, #key)] = kv[2]
            end
        end
    end

    try {function()
        build(args)
    end, catch {function(errors)
        print(errors)
    end}}

end
