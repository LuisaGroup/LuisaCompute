import Library as lb
import BuildData as bd
from xml.etree import ElementTree as ET
from shutil import copyfile
import os
import os.path

def SetCLCompile(root:ET.Element, prep:str, includeDir:str, xmlns):
    inc = lb.XML_GetSubElement(root, 'AdditionalIncludeDirectories', xmlns)
    inc.text = includeDir + "%(AdditionalIncludeDirectories);"
    ppEle = lb.XML_GetSubElement(root, 'PreprocessorDefinitions', xmlns)
    ppEle.text = prep + "%(PreprocessorDefinitions);"

def SetLink(root:ET.Element, dep:str, xmlns):
    depEle = lb.XML_GetSubElement(root, 'AdditionalDependencies', xmlns)
    depEle.text = dep + "%(AdditionalDependencies);"

def SetItemDefinitionGroup(root : ET.Element, xmlns):
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, 'ItemDefinitionGroup', xmlns)
    for itemGroup in itemGroups:
        att = itemGroup.attrib.get("Condition")
        if att == None:
            continue
        preprocess = ''
        includes = ''
        deps = ''
        for inc in bd.IncludePaths:
            includes += inc + ';'
        for d in bd.dependices:
            deps += d + ';'
        for i in bd.PP:
            if att.find('\'' + i + '|') >= 0:
                for macro in bd.PP[i]:
                    preprocess += macro
                    preprocess += ';'
        clComp = lb.XML_GetSubElement(itemGroup, 'ClCompile', xmlns)
        SetCLCompile(clComp, preprocess, includes, xmlns)
        link = lb.XML_GetSubElement(itemGroup, 'Link', xmlns)
        SetLink(link, deps, xmlns)

def SetItemGroups(root:ET.Element, xmlns, ignoreFiles:dict, allFiles:dict):
    subEles = []
    for sub in root:
        if lb.XML_GetTag(sub, xmlns) == "ItemGroup" and len(sub.attrib) == 0:
            subEles.append(sub)
    for i in subEles:
        root.remove(i)
    for exten in allFiles:
        tag = bd.ContainedFiles[exten]
        itemGroup = ET.Element('ItemGroup')
        files = allFiles[exten]
        for i in files:
            if ignoreFiles.get(i) != None:
                continue
            son = ET.Element(tag, {'Include': i})
            itemGroup.append(son)
        root.append(itemGroup)

def RemoveIncludes(root:ET.Element, xmlns):
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

def GetClCompile(result:dict, root:ET.Element, xmlns):
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, "ItemGroup",xmlns)
    for i in itemGroups:
        if len(i.attrib) != 0:
            continue
        for sub in i:
            if lb.XML_GetTag(sub, xmlns) == 'ClCompile':
                includePath = lb.ProcessPath(sub.attrib.get('Include'))
                if includePath == None:
                    continue
                result[includePath] = 1

def RemoveNonExistsPath(root:ET.Element, xmlns, filePaths: dict):
    itemGroups = []
    lb.XML_GetSubElements(itemGroups, root, "ItemGroup", xmlns)
    itemGroupsRemoveList = []
    for i in itemGroups:
        if len(i.attrib) != 0:
            continue
        removeList = []
        for sub in i:
            includePath = lb.ProcessPath(sub.attrib.get('Include'))
            if includePath == None:
                continue
            if filePaths.get(includePath) == None:
                removeList.append(sub)
        for r in removeList:
            i.remove(r)
        if len(i) == 0:
            itemGroupsRemoveList.append(i)
    for i in itemGroupsRemoveList:
        root.remove(i)

def GetVCXProj(path:str):
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

def VCXProjSettingMain():
    xmlPath = bd.Proj + '.vcxproj'
    tree, xmlns = GetVCXProj(xmlPath)
    root = tree.getroot()
    allFiles = {}
    lb.File_GetAllFiles(allFiles, '.', bd.IgnoreFolders, {}, bd.IgnoreFile, bd.ContainedFiles)
    GenerateCMake(allFiles)
    allFileDict = {}
    for i in allFiles:
        lst = allFiles[i]
        for j in lst:
            allFileDict[j] = 1

    ############## Process Sub Model
    ignoreFiles = {}
    for sub in bd.SubProj:
        subPath = sub + '.vcxproj'
        subTree, subXmlns = GetVCXProj(subPath)
        subRoot = subTree.getroot()
        RemoveIncludes(subRoot, subXmlns)
        RemoveNonExistsPath(subRoot, subXmlns, allFileDict)
        GetClCompile(ignoreFiles, subRoot, subXmlns)
        lb.XML_Format(subRoot)
        subTree.write(subPath)
    SetItemGroups(root, xmlns, ignoreFiles, allFiles)
    SetItemDefinitionGroup(root, xmlns)
    lb.XML_Format(root)
    tree.write(xmlPath)
    print("Build Success!")

def GenerateCMake(allFiles:dict):
    example = open("CMakeExample.txt", "r")
    exampleStr = example.read()
    example.close()
    setValue = "set(DX_BACKEND_SOURCES\n"
    hList = allFiles["h"]
    cppList = allFiles["cpp"]
    for i in hList:
        setValue += i + '\n'
    for i in cppList: 
        setValue += i + '\n'
    setValue += ')\n'
    setValue += exampleStr
    f = open("CMakeLists.txt", "w")
    f.write(setValue)
    f.close()
    print("CMake Generated!")
    
def ClearFilters():
    xmlPath = bd.Proj + '.vcxproj'
    ClearFilter(xmlPath)
    for sub in bd.SubProj:
        subPath = sub + '.vcxproj'
        ClearFilter(subPath)
    print("Clear Filters Success!")

def CopyFiles():
    for i in bd.CopyFilePaths:
        copyfile(i[0], i[1])
    print("Copy Success!")


def main():
    CopyFiles()
    ClearFilters()
    VCXProjSettingMain()
