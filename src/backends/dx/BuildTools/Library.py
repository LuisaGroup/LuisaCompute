import os
from xml.etree import ElementTree as ET
################ File
def ProcessPath(oldPath:str):
    newPath = ''
    for i in oldPath:
        if i == '\\':
            newPath += '/'
        else:
            newPath += i
    return newPath

def __GetAllFiles(result:dict, rootDir:str, curDir:str, ignorePath:dict, ignoreFolders:dict, ignoreName:dict, extensions:dict):
    for lists in os.listdir(curDir):
        path = ProcessPath(os.path.join(curDir, lists))
        culledPath = path[len(rootDir): len(path)]
        if ignorePath.get(culledPath) != None:
            continue
        splitPaths = culledPath.split('/')
        if len(splitPaths) == 0:
            continue
        if os.path.isdir(path):
            if ignoreFolders.get(splitPaths[len(splitPaths) - 1]) != None:
                continue
            __GetAllFiles(result, rootDir, path, ignorePath, ignoreFolders, ignoreName, extensions)
        else:
            name = splitPaths[len(splitPaths) - 1]
            if ignoreName.get(name) != None:
                continue
            fileSplits = name.split('.')
            fileExt = fileSplits[len(fileSplits) - 1]
            if extensions.get(fileExt) != None:
                if result.get(fileExt) == None:
                    result[fileExt] = []
                result[fileExt].append (culledPath)

def __GetRootFiles(result:dict, rootDir:str, curDir:str, extensions:dict):
    for lists in os.listdir(curDir):
        path = ProcessPath(os.path.join(curDir, lists))
        culledPath = path[len(rootDir): len(path)]
        splitPaths = culledPath.split('/')
        if len(splitPaths) == 0:
            continue
        if not os.path.isdir(path):
            name = splitPaths[len(splitPaths) - 1]
            fileSplits = name.split('.')
            fileExt = fileSplits[len(fileSplits) - 1]
            if extensions.get(fileExt) != None:
                if result.get(fileExt) == None:
                    result[fileExt] = []
                result[fileExt].append (culledPath)

def File_GetAllFiles(result:dict, rootDir:str, ignorePath:dict, ignoreFolders:dict, ignoreName:dict, extensions:dict):
    rootDir = ProcessPath(rootDir)
    if rootDir[len(rootDir) - 1] != '/':
        rootDir += '/'
    __GetAllFiles(result, rootDir, rootDir, ignorePath, ignoreFolders, ignoreName, extensions)

def File_GetRootFiles(result:dict, rootDir:str, extensions:dict):
    rootDir = ProcessPath(rootDir)
    if rootDir[len(rootDir) - 1] != '/':
        rootDir += '/'
    __GetRootFiles(result, rootDir, rootDir, extensions)
########################## XML
def XML_GetNameSpace(root:ET.Element):
    startIndex = root.tag.find('{')
    if startIndex < 0:
        return None
    endIndex = root.tag.find('}')
    if endIndex < 0:
        return None
    return root.tag[startIndex + 1 : endIndex]

def XML_GetTag(root:ET.Element, xmlns):
    if xmlns == None or len(root.tag) <= len(xmlns) + 2:
        return root.tag
    return root.tag[len(xmlns) + 2:len(root.tag)]


def XML_TextEmpty(text):
    if text == None:
        return True
    for i in text:
        if i != ' ' and i != '\t' and i != '\n':
            return False
    return True
def XML_GetSubElement(root:ET.Element, subElementName:str, xmlns):
    for sub in root:
        if XML_GetTag(sub, xmlns) == subElementName:
            return sub
    newEle = ET.Element(subElementName)
    root.append(newEle)
    return newEle

def XML_Format(element, indent = '\t', newline = '\n', level=0):
    if element:
        if (element.text is None) or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:
            subelement.tail = newline + indent * level
        XML_Format(subelement, indent, newline, level=level + 1)

def XML_GetSubElements(subEles: list, root:ET.Element, subElementName:str, xmlns):
    for sub in root:
        if XML_GetTag(sub, xmlns) == subElementName:
            subEles.append(sub)

