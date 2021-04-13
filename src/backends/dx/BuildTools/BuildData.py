PP = {
    "Debug": [
        "_DEBUG",
    ],
    "Release": [
        "NDEBUG",
    ]
}

IncludePaths = [
    ".",
    "../../",
    "../../ext/spdlog/include",
    "../../ext/mimalloc/include"
]
MSBuild = "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/MSBuild/Current/Bin/MSBuild.exe"
ConfigurePlatform = ["Release", "x64"]
dependices = [
    'kernel32.lib',
    'user32.lib',
    'gdi32.lib',
    'winspool.lib',
    'comdlg32.lib',
    'advapi32.lib',
    'shell32.lib',
    'ole32.lib',
    'oleaut32.lib',
    'uuid.lib',
    'odbc32.lib',
    'odbccp32.lib',
    "d3dcompiler.lib",
    "D3D12.lib",
    "dxgi.lib"
]
Proj = "LC_DXBackend"
SubProj = [

]
ContainedFiles = {
    'cpp': 'ClCompile',
    'h': 'ClInclude',
    'hpp': 'ClInclude',
    'lib': 'Library'
}
IgnoreFolders = {
    "x64": 1,
    "x86": 1,
    ".vs": 1,
    "Shaders": 1,
    "Doc": 1,
    "lib": 1,
    "BuildTools": 1,
    "HLSLCompiler": 1
}
IgnoreFile = {
    "LC_DXBackend.lib": 1,
    "LC_DXBackend.dll": 1
}

CopyFilePaths = [
    ["../../../out/build/x64-Release/bin/luisa-compute-ast.dll",
        "Build/luisa-compute-ast.dll"],
    ["../../../out/build/x64-Release/bin/luisa-compute-runtime.dll",
     "Build/luisa-compute-runtime.dll"],
    ["../../../out/build/x64-Release/bin/luisa-compute-core.dll",
        "Build/luisa-compute-core.dll"],
    ["../../../out/build/x64-Release/bin/spdlog.dll", "Build/spdlog.dll"],
    ["../../../out/build/x64-Release/bin/fmt.dll", "Build/fmt.dll"],
    ["../../../out/build/x64-Release/src/ast/luisa-compute-ast.lib",
        "Build/luisa-compute-ast.lib"],
    ["../../../out/build/x64-Release/src/core/luisa-compute-core.lib",
        "Build/luisa-compute-core.lib"],
        ["../../../out/build/x64-Release/src/runtime/luisa-compute-runtime.lib",
        "Build/luisa-compute-runtime.lib"],
    ["../../../out/build/x64-Release/src/ext/fmt/fmt.lib", "Build/fmt.lib"],
    ["../../../out/build/x64-Release/src/ext/spdlog/spdlog.lib", "Build/spdlog.lib"]
]
