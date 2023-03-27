def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def convert_ty(ty):
    m0 = {'api::CreatedResourceInfo':'api::CreatedResourceInfoRemote',
        'api::CreatedBufferInfo':'CreatedBufferInfoRemote',
        'api::CreatedShaderInfo':'api::CreatedShaderInfoRemote',
        'api::CreatedSwapchainInfo':'api::CreatedSwapchainInfoRemote'}
    if ty in m0:
        return m0[ty]
    if ty.startswith('api::'):
        return ty
    m = {
        'void': '()',
        'size_t': 'usize',
        'uint32_t': 'u32',
        'uint64_t': 'u64',
        'bool': 'bool',
        'ConstAccelOptionPtr': 'api::AccelOption',
        'ShaderOptionPtr': 'api::ShaderOption',
        'ConstCharPtr':'String',
        'BytePtr': 'Vec<u8>',
    }
    return m[ty]

f = open('api.h', 'r')
out = open('../rust/luisa_compute_backend/src/api_message.rs', 'w')
lines = f.readlines()
out.write('use luisa_compute_api_types as api;\n')
out.write('use serde::*;\n')
exclude = set([
    'luisa_compute_free_c_string',
    'luisa_compute_device_query',
    'luisa_compute_buffer_create',
    'luisa_compute_stream_dispatch',
    'luisa_compute_shader_create'
])
funcnames = []
def parse_line(line):
    line = line.replace('LC', 'api::')
    tokens = [t.strip() for t in line.split(' ')]
    tokens = [t for t in tokens if t != ''][1:]
    print(tokens)
    ret = convert_ty(tokens[0])
    funcname = tokens[1]
    if funcname in exclude:
        return
    funcnames.append(funcname)
    args = tokens[3:-2]
    args = ' '.join(args)
    args = args.split(',')
    print(args)
    funcname = snake_to_camel(funcname[len('luisa_compute_'):])
    out.write('#[derive(Serialize, Deserialize, Debug, Clone)]\n')
    out.write('pub struct ' + funcname + ' {\n')
    for arg in args:
        if arg == '':
            continue
        ty_param = [ x for x in arg.split(' ') if x]
        ty = convert_ty(ty_param[0])
        param = ty_param[1]
        out.write('    pub ' + param + ': ' + ty + ',\n')
    if ret != '()':
        out.write('    pub ret: ' + ret + ',\n')
    out.write('    pub message_id: u64,\n')
    out.write('}\n')

for line in lines:
    line = line.replace('(', ' ( ')
    line = line.replace(')', ' ) ')
    line = line.replace(',', ' , ')
    if line.startswith('LUISA_EXPORT_API'):
        parse_line(line)
