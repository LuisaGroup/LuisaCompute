-- imports
import("core.project.config")
import("core.project.project")
import("clang.compile_commands", {rootdir = path.join(os.programdir(), "plugins", "project")})

-- config target
function _config_target(target)
    local oldenvs = os.addenvs(target:pkgenvs())
    for _, rule in ipairs(target:orderules()) do
        local on_config = rule:script("config")
        if on_config then
            on_config(target)
        end
    end
    local on_config = target:script("config")
    if on_config then
        on_config(target)
    end
    if oldenvs then
        os.setenvs(oldenvs)
    end
end

-- config targets
function _config_targets()
    for _, target in ipairs(project.ordertargets()) do
        if target:is_enabled() then
            _config_target(target)
        end
    end
end

-- main entry
function main()

    -- generate compile_commands.json
    -- @note we can only load configuration because we watched onFileChanged(xmake.conf)
    os.setenv("XMAKE_IN_PROJECT_GENERATOR", "true")
    os.setenv("XMAKE_GENERATOR_COMPDB_LSP", "clangd")
    config.load()
    _config_targets()
    compile_commands.make(".vscode")
    os.setenv("XMAKE_IN_PROJECT_GENERATOR", nil)
end
