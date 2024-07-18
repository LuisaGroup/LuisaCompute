#include "header.h"
inline float func2_75_d1fcdb0d841dbe25c70a1a26654d0a70(buffer_type a0, int32_t a1) {
return ((float*)(a0.ptr))[a1];
}
inline float4 func2_118_97995f447a0f1b331647db4d90941341(float a0, float a1, float a2, float a3) {
return (float4){a0, a1, a2, a3};
}
static float4 func1_0_aae398fb22ecfa8aa2683046e81c16eb(float4 a, float4 b){
return (float4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline void func2_76_6c37ee55c9d122b1bbcfb450cabc56c8(buffer_type a0, int32_t a1, float4 a2) {
((float4*)(a0.ptr))[a1] = a2;
}
static void builtin_c4434d750cf64f0eae3f73cca8650b16(uint32_t3 thd_id, uint32_t3 blk_id, uint32_t3 dsp_id, uint32_t3 dsp_size, uint32_t3 ker_id, buffer_type b0, buffer_type b1, float4 v2){
float v3;
float v4;
float v5;
float v6;
float4 v7;
float v8;
float v9;
float v10;
float v11;
float4 v12;
float v13;
float v14;
float v15;
float v16;
{
v3=((0x0.0000000000000p+0f));
v3=(func2_75_d1fcdb0d841dbe25c70a1a26654d0a70(b1, (3)));
v4=((0x0.0000000000000p+0f));
v4=(func2_75_d1fcdb0d841dbe25c70a1a26654d0a70(b1, (2)));
v5=((0x0.0000000000000p+0f));
v5=(func2_75_d1fcdb0d841dbe25c70a1a26654d0a70(b1, (1)));
v6=((0x0.0000000000000p+0f));
v6=(func2_75_d1fcdb0d841dbe25c70a1a26654d0a70(b1, (0)));
v8=((0x0.0000000000000p+0f));
v8=(GET(float, v7, 0));
v9=((0x0.0000000000000p+0f));
v9=(GET(float, v7, 1));
v10=((0x0.0000000000000p+0f));
v10=(GET(float, v7, 2));
v11=((0x0.0000000000000p+0f));
v11=(GET(float, v7, 3));
v7=(((float4){0x0.0000000000000p+0f,0x0.0000000000000p+0f,0x0.0000000000000p+0f,0x0.0000000000000p+0f}));
v7=(func2_118_97995f447a0f1b331647db4d90941341(v6, v5, v4, v3));
v13=((0x0.0000000000000p+0f));
v13=(GET(float, v12, 0));
v14=((0x0.0000000000000p+0f));
v14=(GET(float, v12, 1));
v15=((0x0.0000000000000p+0f));
v15=(GET(float, v12, 2));
v16=((0x0.0000000000000p+0f));
v16=(GET(float, v12, 3));
v12=(((float4){0x0.0000000000000p+0f,0x0.0000000000000p+0f,0x0.0000000000000p+0f,0x0.0000000000000p+0f}));
v12=(func1_0_aae398fb22ecfa8aa2683046e81c16eb(v7,v2));
v7=(v12);
func2_76_6c37ee55c9d122b1bbcfb450cabc56c8(b0, (0), v7);
}
}
__declspec(dllexport) uint32_t kernel_arg_usage_c4434d750cf64f0eae3f73cca8650b16(uint32_t idx) {
static const uint32_t usages[] = {2u, 1u, 1u};
return usages[idx];
}
__declspec(dllexport) uint32_t3 kernel_block_size_c4434d750cf64f0eae3f73cca8650b16(){
return (uint32_t3){256, 1, 1};
}
__declspec(dllexport) uint64_t2 kernel_args_md5_c4434d750cf64f0eae3f73cca8650b16(){
return (uint64_t2){5288038341929428867ull, 12961784310145466620ull};
}
typedef struct {
alignas(16) buffer_type a0;
alignas(16) buffer_type a1;
alignas(16) float4 a2;
} Args;
__declspec(dllexport)  void kernel(uint32_t3 thd_id, uint32_t3 blk_id, uint32_t3 dsp_id, uint32_t3 dsp_size, uint32_t3 ker_id, Args* args){
builtin_c4434d750cf64f0eae3f73cca8650b16(thd_id, blk_id, dsp_id, dsp_size, ker_id, args->a0, args->a1, args->a2);
}
