def convert_ty(ty):
    if ty.startswith('api::'):
        return ty
    m = {
        'void': '()',
        'size_t': 'usize',
        'uint32_t': 'u32',
        'uint64_t': 'u64',
        'bool': 'bool',
        'VoidPtr': '*mut c_void',
        'ConstVoidPtr': '*const c_void',
        'ConstCharPtr': '*const c_char',
        'CharPtr': '*mut c_char',
        'ConstAccelOptionPtr': '&api::AccelOption',
        'ShaderOptionPtr': '&api::ShaderOption',
        'LCDispatchCallback': 'unsafe extern "C" fn(*mut u8)',
        'LoggerCallback': 'unsafe extern "C" fn(*const c_char, *const c_char)',
        'BytePtr': '*mut u8',
    }
    return m[ty]


f = open('api.h', 'r')
out = open('../rust/luisa_compute_backend/src/binding.rs', 'w')
lines = f.readlines()
out.write('use luisa_compute_api_types as api;\n')
out.write('use std::ffi::*;\n')
out.write('use std::path::Path;\n')
out.write('#[repr(C)]\npub struct Binding {\n')
out.write('   #[allow(dead_code)]\n    pub lib: libloading::Library,\n')
funcnames = []


def parse_line(line):
    line = line.replace('LC', 'api::')
    tokens = [t.strip() for t in line.split(' ')]
    tokens = [t for t in tokens if t != ''][1:]
    print(tokens)
    ret = convert_ty(tokens[0])
    funcname = tokens[1]
    funcnames.append(funcname)
    args = tokens[3:-2]
    args = ' '.join(args)
    args = args.split(',')
    print(args)
    out.write('    pub ' + funcname + ': unsafe extern "C" fn(')
    for arg in args:
        if arg == '':
            continue
        ty_param = [x for x in arg.split(' ') if x]
        ty = convert_ty(ty_param[0])
        param = ty_param[1]
        out.write(param + ': ' + ty + ', ')
    out.write(') -> ' + ret + ',\n')


for line in lines:
    line = line.replace('(', ' ( ')
    line = line.replace(')', ' ) ')
    line = line.replace(',', ' , ')
    if line.startswith('LUISA_EXPORT_API'):
        parse_line(line)
out.write('}\n')

out.write('impl Binding {\n')
out.write('    pub unsafe fn new(lib_path:&Path) -> Result<Self, libloading::Error> {\n')
out.write('        let lib = libloading::Library::new(lib_path)?;\n')
for funcname in funcnames:
    out.write('        let ' + funcname + ' =  *lib.get(b"' + funcname + '")? ;\n')
out.write('        Ok(Self { lib,\n')
for funcname in funcnames:
    out.write('            ' + funcname + ',\n')
out.write('        })\n')
out.write('    }\n')
out.write('}\n')
