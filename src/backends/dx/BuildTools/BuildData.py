Proj = None
SubProj = [
    {
        "Name": "VEngine_DirectX",
        "RemoveHeader": 1
    }
]

Compiler = {
    "BuildToolSubDir": "MSBuild.exe",
    "Configuration": "Release",
    "Platform": "x64"
}

ContainedFiles = {
    'cpp': 'ClCompile',
    'c': 'ClCompile',
    'cc': 'ClCompile',
    'cxx': 'ClCompile',
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
    "Project": 1,
    "Build/ShaderCompileResult": 1,
    "Build/Data": 1,
    "Build/Resource": 1,
    "MonoInclude": 1,
    "lib": 1,
    'BuildTools': 1
}
IgnoreFile = {
    "VEngine_Compute.lib": 1,
    "VEngine_Network.lib": 1,
    "program.inf": 1
}

CopyFilePaths = []
