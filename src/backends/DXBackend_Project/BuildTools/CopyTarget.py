from shutil import copyfile
path = ["Build/", "../../../out/build/x64-Release/bin/"]
files = [
    "mimalloc-override.dll",
    "mimalloc-redirect.dll",
    "LC_DXBackend.dll",
]
def CopyFiles():
    for i in files:
        fromPath = path[0] + i
        toPath = path[1] + i
        copyfile(fromPath, toPath)
        print("Copyed From " + fromPath + " to " + toPath)
    print("Copy Finished")