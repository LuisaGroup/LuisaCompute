f = open("Include.cginc")
lines = f.readlines()
f.close()
def printScalar(line, word):
    return line.replace("Z",word)

def printVec(line, word):
    result = printScalar(line, word)
    for i in range(2, 5):
        result += printScalar(line, word + str(i))
    return result

def printVecList(words:list, currLine:str):
    result = ""
    for i in words:
        result += printVec(currLine, i)
    return result

nextIsCodegen = False
func = ""
currLine = ""
valuevv = ""
def vec(words):
    return printVecList(words, currLine)
for i in lines:
    if nextIsCodegen:
        currLine = i
        exec("valuevv += " + func)
        nextIsCodegen = False
        continue
    if i.find("////") == 0:
        func = i[4:len(i)]
        nextIsCodegen = True
print(valuevv)
print("Finish")
