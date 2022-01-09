import Library as lb
import BuildData as bd
import Database as db
import ctypes
from xml.etree import ElementTree as ET
from shutil import copyfile
from shutil import rmtree
import os
import os.path


def SetCLCompile(pdbIndex: int, root: ET.Element, xmlns):
    pdb = lb.XML_GetSubElement(root, 'ProgramDataBaseFileName', xmlns)
    pdb.text = "$(IntDir)vc_" + str(pdbIndex) + ".pdb"

def GenerateCMakeFile(
    projectName: str,
    projectIncludeFiles: list,
    cmakeTemplateString: str):
    newStr = cmakeTemplateString.replace("VE_PROJECT", projectName)
    pathStr = ""
    for i in projectIncludeFiles:
        pathStr += i + '\n'
    return newStr.replace("VE_FILE", pathStr)

def SetLink(root: ET.Element, dep: str, xmlns):
    depEle = lb.XML_GetSubElement(root, 'AdditionalDependencies', xmlns)
    depEle.text = dep + "%(AdditionalDependencies);"


def GeneratePlaceHolder():
    for i in bd.SubProj:
        holder = i.get("PlaceHolder")
        if holder == None or holder == "":
            continue
        path = holder.replace("#", i["Name"])
        if os.path.isfile(path):
            continue
        f = open(path, 'w')
        f.write("0")
        f.close()


def SetItemDefinitionGroup(pdbIndex: int, root: ET.Element, data: dict, xmlns):
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, 'ItemDefinitionGroup', xmlns)
    for itemGroup in itemGroups:
        att = itemGroup.attrib.get("Condition")
        if att == None:
            continue
        clComp = lb.XML_GetSubElement(itemGroup, 'ClCompile', xmlns)
        SetCLCompile(pdbIndex, clComp, xmlns)
        link = lb.XML_GetSubElement(itemGroup, 'Link', xmlns)


def RemoveIncludes(root: ET.Element, xmlns):
    items = []
    lb.XML_GetSubElements(items, root, 'ItemGroup', xmlns)
    removeItemGroups = []
    for item in items:
        if len(item.attrib) > 0:
            continue
        removeItems = []
        for inc in item:
            if lb.XML_GetTag(inc, xmlns) == 'ClInclude':
                removeItems.append(inc)
        for i in removeItems:
            item.remove(i)
        if len(item) == 0:
            removeItemGroups.append(item)
    for i in removeItemGroups:
        root.remove(i)


def RemoveNonExistsPath(subName: str, dll, root: ET.Element, xmlns, addFile:bool, cmakeTemplateString: str):
    subName = subName.lower()
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, "ItemGroup", xmlns)
    itemGroupsRemoveList = []
    for i in itemGroups:
        if len(i.attrib) != 0:
            continue
        isCompileList = True
        for sub in i:
            if lb.XML_GetTag(sub, xmlns) != 'ClCompile':
                isCompileList = False
                break
        if isCompileList:
            itemGroupsRemoveList.append(i)
    for i in itemGroupsRemoveList:
        root.remove(i)
    if addFile:
        CompileItemGroup = ET.Element("ItemGroup", {})
        root.append(CompileItemGroup)
        dll.Py_SetPackageName(subName.encode("ascii"))
        sz = dll.Py_PathSize()
        projectFileNames = []
        for i in range(sz):
            p = str(ctypes.string_at(dll.Py_GetPath(i)), "ascii")
            projectFileNames.append(p)
            CompileItemGroup.append(
                ET.Element("ClCompile", {'Include': p}))
        return GenerateCMakeFile(
            subName,
            projectFileNames,
            cmakeTemplateString
        )


def GetVCXProj(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    xmlns = lb.XML_GetNameSpace(root)
    if xmlns != None:
        ET.register_namespace('', xmlns)
    return tree, xmlns


def ClearFilter(path):
    filterPath = path + '.filters'
    if os.path.exists(filterPath):
        os.remove(filterPath)


def OutputXML(tree, path):
    tree.write(path)

def VCXProjSettingMain(readFile:bool):
    dll = None
    cmakeTemplateString = ""
    if readFile:
        filePath = os.path.dirname(os.path.realpath(__file__))
        cmakeTempF = open(filePath + "/CMakeTemplate.txt", "r")
        cmakeTemplateString = cmakeTempF.read()
        cmakeTempF.close()
        dll = ctypes.cdll.LoadLibrary(filePath + "/VEngine_CPPBuilder.dll")
        dll.Py_InitFileSys()
        dll.Py_AddExtension("cpp".encode("ascii"))
        dll.Py_AddExtension("c".encode("ascii"))
        dll.Py_AddExtension("cxx".encode("ascii"))
        dll.Py_AddExtension("cc".encode("ascii"))
        dll.Py_AddIgnorePath(".vs".encode("ascii"))
        dll.Py_AddIgnorePath("Build".encode("ascii"))
        dll.Py_AddIgnorePath("BuildTools".encode("ascii"))
        dll.Py_AddIgnorePath("x64".encode("ascii"))
        dll.Py_AddIgnorePath("x86".encode("ascii"))
        dll.Py_ExecuteFileSys()
        dll.Py_GetPath.restype = ctypes.c_char_p

    pdbIndex = 0
    cmakeResult = ""
    for sub in bd.SubProj:
        subName = sub["Name"]
        subPath = subName + '.vcxproj'
        subTree, subXmlns = GetVCXProj(subPath)
        subRoot = subTree.getroot()
        if sub.get("RemoveHeader") == 1:
            RemoveIncludes(subRoot, subXmlns)
        cmakeData = RemoveNonExistsPath(subName, dll, subRoot, subXmlns,readFile, cmakeTemplateString)
        if cmakeData != None:
            cmakeResult += cmakeData + '\n'
        SetItemDefinitionGroup(pdbIndex, subRoot, sub, subXmlns)
        pdbIndex += 1
        lb.XML_Format(subRoot)
        OutputXML(subTree, subPath)
    if readFile:
        dll.Py_DisposeFileSys()
    cmakeTempF = open("CMakeLists.txt", "w")
    cmakeTempF.write(cmakeResult)
    cmakeTempF.close()
    print("Build Success!")

def DeleteDotVS():
    try:
        rmtree(".vs")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        return
    print("delete .vs success!");
def MakeVCXProj(inverse:bool):
    backup = ""
    vs = ""
    if inverse:
        backup = "vcxproj"
        vs = "vcxprojbackup"
    else:
        vs = "vcxproj"
        backup = "vcxprojbackup"

    ext = {backup: 1}
    fileResults = {}
    lb.File_GetRootFiles(fileResults, ".", ext)
    lst = fileResults.get(backup)
    if lst == None:
        return
    for i in lst:
        copyfile(i, i.replace("." + backup, "." + vs))


def ClearFilters():
    for sub in bd.SubProj:
        subPath = sub["Name"] + '.vcxproj'
        ClearFilter(subPath)
    print("Clear Filters Success!")


def CopyFiles():
    for i in bd.CopyFilePaths:
        copyfile(i[0], i[1])
    print("Copy Success!")


def VcxMain():
    MakeVCXProj(False)
    GeneratePlaceHolder()
    ClearFilters()
    VCXProjSettingMain(True)
    DeleteDotVS()

def VcxMain_EmptyFile():
    GeneratePlaceHolder()
    ClearFilters()
    VCXProjSettingMain(False)
    MakeVCXProj(True)

def CompileProj():
    obj = db.SerializeObject()
    rt = obj.GetRootNode()
    rt.Reset()
    rt.Add(bd.Compiler)
    projs = obj.CreateDict()
    for p in bd.SubProj:
        deps = p.get("Dependency")
        if deps == None:
            deps = []
        projs.Set(p["Name"], deps)
    rt.Set("Projects", projs)
    filePath = os.path.dirname(os.path.realpath(__file__))
    dll = ctypes.cdll.LoadLibrary(filePath + "/VEngine_CPPBuilder.dll")
    dll.compile_msbuild.restype = ctypes.c_bool
    if dll.compile_msbuild(ctypes.c_uint64(obj.data)):
        print("Finished!")
    else:
        print("Compile Failed!")