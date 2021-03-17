import Library as lb
import BuildData as bd
import os

def GetCompile(proj):
    return '\"' + bd.MSBuild + '\" \"' + proj + ".vcxproj\" -property:Configuration=" + bd.ConfigurePlatform[0] + ",Platform=" + bd.ConfigurePlatform[1] + "\n"

def CompileMain():
    f = open("__temp_TEMP_.cmd", 'w')
    for i in bd.SubProj:
        f.write(GetCompile(i))
    f.write(GetCompile(bd.Proj))
    f.close()
    os.system("__temp_TEMP_.cmd")
    os.remove("__temp_TEMP_.cmd")
    print("Compile Success!")