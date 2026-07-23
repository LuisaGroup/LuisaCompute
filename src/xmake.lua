includes("ext")
includes("core", "vstl", "runtime")
if has_config("lc_enable_osl") then
    includes("osl")
end
if has_config("lc_enable_dsl") then
    includes("dsl")
end
if has_config("lc_enable_xir") then
    includes("coro")
end
if has_config("lc_enable_gui") then
    includes("gui")
end
if has_config("_lc_enable_py") then
    includes("py")
end
includes("backends")
if has_config("lc_enable_tests") then
    includes("tests")
end
if has_config("lc_enable_clangcxx") then
    includes("clangcxx")
end

-- includes("tensor")
