import Library as lb
import BuildData as bd
import os
import os.path

def GetPath(currentPath, inputStr):
    currentPath = currentPath.replace('\\', '/')
    strs = currentPath.split("/")    
    strs.pop(len(strs) - 1)
    inputStr = inputStr.replace('\\', '/')
    chunks = inputStr.split("/")  
    for i in chunks:
        if i == "..":
            strs.pop(len(strs) - 1)
        else:
            strs.append(i)
    result = ""
    for i in strs:
        result += i + '/'
    result = result[0:len(result) - 1]
    while(result[len(result) - 1] == '\"'):
        result = result[0:len(result) - 1]
    return result


def GetIncludeData(currentPath, inputStr):
    originPath = inputStr
    inputStr = inputStr.strip()
    includePre = "#include \""
    idx = inputStr.find(includePre)
    if idx != 0:
        return originPath
    inputStr = inputStr[len(includePre):len(inputStr) - 1]
    return "#include <" + GetPath(currentPath, inputStr) + ">\n"

def ProcessFile(path):
    try:
        file = open(path, encoding='utf-8')
        lines = file.readlines()
        result = ""
        for line in lines:
            result += GetIncludeData(path, line.replace('\r', ' '))
        file.close()
        file = open(path, 'w')
        file.write(result)
        file.close()
    except:
        print("File: " + path + " have non utf-8 code!")

def MakeInclude():  
    allFiles = {}
    lb.File_GetAllFiles(allFiles, '.', bd.IgnoreFolders, {}, bd.IgnoreFile, bd.ContainedFiles)
    lst = [
        allFiles.get('h'),
        allFiles.get('cpp'),
        allFiles.get('c'),
        allFiles.get('cxx'),
        allFiles.get('hpp')
    ]
    for fileLst in lst:
        if fileLst == None:
            continue
        for i in fileLst:
            ProcessFile(i)
    print("finished")