static const std::string_view headerName = R"(void kernel(
uniform uint xc,
uniform uint yc,
uniform uint zc,
uint x,
uint y,
uint z,
uniform uint64 arg) {
const uint3 dsp_id={x,y,z};
const uint3 thd_id={x,y,z};
const uint3 blk_id={0,0,0};
uniform uint3 dsp_c={xc,yc,zc};
)"sv;
static const std::string_view exportName = R"(
export void run(
uniform uint xc,
uniform uint yc,
uniform uint zc,
uniform uint64 arg) {
xc=max(xc,1);yc=max(yc,1);zc=max(zc,1);
foreach(x = 0 ... xc, y = 0 ... yc, z = 0 ... zc){
kernel(xc,yc,zc,x,y,z,arg);
}}
)"sv;