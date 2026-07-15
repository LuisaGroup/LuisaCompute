import struct, sys
f = open(sys.argv[1], 'rb')
data = f.read()
f.close()
words = struct.unpack('<' + 'I' * (len(data)//4), data)
print('Magic:', hex(words[0]))
print('Version:', words[1])
print('Generator:', words[2])
print('Bound:', words[3])
print('Schema:', words[4])
idx = 5
caps = []
while idx < len(words):
    word = words[idx]
    op = word & 0xFFFF
    count = word >> 16
    if op == 17:  # OpCapability
        caps.append(words[idx+1])
    elif op == 71:  # OpTypeVoid ends the capabilities
        break
    elif op == 5:  # OpName can appear before TypeVoid
        pass
    elif op == 7:  # OpDecorate
        pass
    elif op == 71:  # OpTypeVoid
        break
    idx += count
print('Capabilities:', [hex(c) for c in caps])
