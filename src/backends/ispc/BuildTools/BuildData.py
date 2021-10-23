Proj = None
SubProj = [
    {
        "Name": "ISPC_VSProject",
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
    'BuildTools': 1
}
IgnoreFile = {
}

CopyFilePaths = []
