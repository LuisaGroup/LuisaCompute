needCodegenFile = [
    "RHI/ICommandBuffer.h",
    "RHI/ICommandBuffer.cpp"
]

codegenStart = "//VENGINE_CODEGEN start"
codegenEnd = "//VENGINE_CODEGEN end"
codegenSign = "//VENGINE_CODEGEN"


def SeparateCode(inputStr):
    layer = 0
    inField = False
    resultWords = []
    curStr = ""
    for c in inputStr:
        if not inField:
            if c == '[':
                layer = 1
                inField = True
        else:
            if c == ']':
                layer -= 1
                if layer <= 0:
                    inField = False
                    resultWords.append(curStr)
                    curStr = ""
            else:
                curStr += c
    if len(curStr) > 0:
        resultWords.append(curStr)
    return resultWords


def GenCopyCommand(resultLines: list, commands: list):
    resultLines.append(codegenStart + '\n')
    for i in range(1, len(commands)):
        resultLines.append(commands[0].replace("##", commands[i]) + '\n')
    resultLines.append(codegenEnd + '\n')


def GenReplaceCommand(resultLines: list, commands: list):
    resultLines.append(codegenStart + '\n')
    sentence = commands[0]
    for i in range(1, len(commands)):
        strs = commands[i].split(':')
        replStr = ""
        for c in range(1, len(strs)):
            replStr += strs[c]
        sentence = sentence.replace(strs[0], replStr)
    resultLines.append(sentence + '\n')
    resultLines.append(codegenEnd + '\n')


# //VENGINE_CODEGEN [copy] [virtual void Fuck##] [0] [1] [2]
# //VENGINE_CODEGEN [replace] [virtual void FuckXXYY] [XX:1] [YY:2]
codegenDict = {
    "copy": GenCopyCommand,
    "replace": GenReplaceCommand
}


def ProcessLine(lines: list):
    resultLines = []
    add = True
    for i in lines:
        line = i.strip()
        if len(line) == 0:
            resultLines.append("\n")
            continue
        if line[len(line) - 1] != '\n':
            line += '\n'
        if line.find(codegenStart) == 0:
            add = False
            continue
        elif line.find(codegenEnd) == 0:
            add = True
            continue
        if add:
            if line.find(codegenSign) == 0:
                resultLines.append(line)
                command = line[len(codegenSign):len(line)]
                resultWords = SeparateCode(command)
                func = codegenDict.get(resultWords[0])
                if func != None:
                    resultWords.pop(0)
                    func(resultLines, resultWords)
            else:
                resultLines.append(i)

    return resultLines


def ProcessFile(fileName: str):
    f = open(fileName, "r")
    lines = f.readlines()
    f.close()
    clearedList = ProcessLine(lines)
    resultStr = ""
    for i in clearedList:
        resultStr += i
    f = open(fileName, "w")
    f.write(resultStr)
    f.close()

def ProcessMain():
    for i in needCodegenFile:
        ProcessFile(i)