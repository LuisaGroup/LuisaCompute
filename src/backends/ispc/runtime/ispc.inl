static const std::string_view headerName = R"(void kernel(
uniform const uint3& blk_c,
uniform const uint3& thd_c,
const uint3& blk_id,
uniform const uint3& thd_id,
uniform const uint3& dsp_c,
const uint3& dsp_id,
uniform uint64 arg){
)"sv;
static const std::string_view exportName = R"(
export void run(
uniform uint thd_cX,
uniform uint thd_cY,
uniform uint thd_cZ,
uniform uint blk_cX,
uniform uint blk_cY,
uniform uint blk_cZ,
uniform uint thd_idX,
uniform uint thd_idY,
uniform uint thd_idZ,
uniform uint64 arg) {
uniform const uint3 thd_c = {X,Y,Z};
uniform uint3 thd_id = {thd_idX, thd_idY, thd_idZ};
uniform uint3 blk_c = {blk_cX,blk_cY,blk_cZ};
uniform uint3 dsp_c = thd_c * blk_c;
uniform uint3 ldsp_id = thd_id * blk_c;
)"sv;
static const std::string_view zero_blk_id = "uint3 blk_id={0,0,0};\n";
static const std::string_view blk_id = "uint3 blk_id={x,y,z};\n";
static const std::string_view tail = R"(uint3 dsp_id = ldsp_id + blk_id;
kernel(blk_c, thd_c, blk_id, thd_id, dsp_c, dsp_id, arg);
)"sv;