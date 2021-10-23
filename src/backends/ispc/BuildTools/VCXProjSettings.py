import Library as lb
import BuildData as bd
import Database as db
import ctypes
from xml.etree import ElementTree as ET
from shutil import copyfile
import os
import os.path


def SetCLCompile(pdbIndex: int, root: ET.Element, xmlns):
    pdb = lb.XML_GetSubElement(root, 'ProgramDataBaseFileName', xmlns)
    pdb.text = "$(IntDir)vc_" + str(pdbIndex) + ".pdb"


def SetLink(root: ET.Element, dep: str, xmlns):
    depEle = lb.XML_GetSubElement(root, 'AdditionalDependencies', xmlns)
    depEle.text = dep + "%(AdditionalDependencies);"

def GenerateCMake(root:ET.Element, xmlns):
    cmakePaths = []
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, "ItemGroup",xmlns)
    for i in itemGroups:
        if len(i.attrib) != 0:
            continue
        for sub in i:
            if lb.XML_GetTag(sub, xmlns) == 'ClCompile':
                path = sub.attrib.get('Include')
                if path != None:
                    cmakePaths.append(path.replace('\\','/'))
    example = open("CMakeLists.txt", "r")
    exampleStr = example.read()
    example.close()
    setValue = "\n    set(ISPC_BACKEND_SOURCES\n"
    for i in cmakePaths: 
        setValue += "        " + i + '\n'
    setValue += '    )\n'
    splitedStrs = exampleStr.split("##[[[]]]")
    if len(splitedStrs) == 2:
        setValue = splitedStrs[0] + "##[[[]]]" + setValue + "##[[[]]]" + splitedStrs[1]
    elif len(splitedStrs) == 3:
        setValue = splitedStrs[0] + "##[[[]]]" + setValue + "##[[[]]]" + splitedStrs[2]
    else: 
        print("CMakeLists format wrong!")
        return
    example = open("CMakeLists.txt", "w")
    example.write(setValue)
    example.close()
    print("CMake Generate Success!")
    

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


def RemoveNonExistsPath(subName: str, dll, root: ET.Element, xmlns, addFile:bool):
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
        for i in range(sz):
            p = str(ctypes.string_at(dll.Py_GetPath(i)), "ascii")
            CompileItemGroup.append(
                ET.Element("ClCompile", {'Include': p}))


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


def UpdateCPPFile(filePath: str, resultDict: dict):
    cppFile = open(filePath)
    firstLine = cppFile.readline().lower().replace(
        '\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cppFile.close()
    pragmaStr = "#pragma"
    packageStr = "vengine_package"
    pragmaID = firstLine.find(pragmaStr)
    if pragmaID == -1:
        return
    for i in range(pragmaID):
        if firstLine[i] != ' ':
            return
    packID = firstLine.find(packageStr)
    if packID == -1 or packID <= (pragmaID + len(pragmaStr)):
        return
    startRange = 0
    for i in range(packID + len(packageStr), len(firstLine)):
        if firstLine[i] != ' ':
            startRange = i
            break
    endRange = len(firstLine)
    for i in range(endRange - 1, startRange, -1):
        if (firstLine[i] == ' '):
            endRange = i
            break
    if(endRange - startRange > 0):
        packageName = firstLine[startRange: endRange]
        lst = resultDict.get(packageName)
        if lst == None:
            lst = []
            resultDict[packageName] = lst
        lst.append(filePath)


def UpdateCPPFiles(fileDict: dict):
    resultDict = {}  # Package : Path
    for v in ["cpp", "c", "cc", "cxx"]:
        lst = fileDict.get(v)
        if lst != None:
            for path in lst:
                UpdateCPPFile(path, resultDict)
    return resultDict


def VCXProjSettingMain(readFile:bool):
    dll = None
    if readFile:
        filePath = os.path.dirname(os.path.realpath(__file__))
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
    for sub in bd.SubProj:
        subName = sub["Name"]
        subPath = subName + '.vcxproj'
        subTree, subXmlns = GetVCXProj(subPath)
        subRoot = subTree.getroot()
        if sub.get("RemoveHeader") == 1:
            RemoveIncludes(subRoot, subXmlns)
        RemoveNonExistsPath(subName, dll, subRoot, subXmlns,readFile)
        SetItemDefinitionGroup(pdbIndex, subRoot, sub, subXmlns)
        GenerateCMake(subRoot, subXmlns)
        pdbIndex += 1
        lb.XML_Format(subRoot)
        OutputXML(subTree, subPath)
    if readFile:
        dll.Py_DisposeFileSys()
    print("Build Success!")


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

def VcxMain_EmptyFile():
    GeneratePlaceHolder()
    ClearFilters()
    VCXProjSettingMain(False)
    MakeVCXProj(True)