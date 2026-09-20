#include <luisa/vstl/md5.h>
#include <luisa/core/platform.h>
#include <luisa/vstl/string_utility.h>

/* Little-endian detection for the fast load/store paths. MD5 blocks and
   the final digest are defined to be little-endian, so on little-endian
   hosts the byte shuffling in decode/encode can be skipped entirely. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#define VSTD_MD5_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#elif defined(_M_X64) || defined(_M_IX86)
#define VSTD_MD5_LITTLE_ENDIAN 1
#else
#define VSTD_MD5_LITTLE_ENDIAN 0
#endif

namespace vstd {
namespace detail {

class MD5_Impl {
public:
    /* Construct a MD5_Impl object with a string. */
    MD5_Impl(span<uint8_t const> message, uint8_t *digest);

    /* Generate md5 digest. */
    void GetDigest();

    static constexpr uint32_t s11 = 7;
    static constexpr uint32_t s12 = 12;
    static constexpr uint32_t s13 = 17;
    static constexpr uint32_t s14 = 22;
    static constexpr uint32_t s21 = 5;
    static constexpr uint32_t s22 = 9;
    static constexpr uint32_t s23 = 14;
    static constexpr uint32_t s24 = 20;
    static constexpr uint32_t s31 = 4;
    static constexpr uint32_t s32 = 11;
    static constexpr uint32_t s33 = 16;
    static constexpr uint32_t s34 = 23;
    static constexpr uint32_t s41 = 6;
    static constexpr uint32_t s42 = 10;
    static constexpr uint32_t s43 = 15;
    static constexpr uint32_t s44 = 21;

    /* Longest message (in bytes) handled by the one-shot fast path:
       two padded blocks at most (112 = 64 + 56). */
    static constexpr size_t ONESHOT_MAX_LEN = 112;

    /* MD5_Impl basic transformation. Transforms state based on block. */
    void transform(const uint8_t block[64]);

    /* Same transformation working on a caller-provided state. */
    static void transform_state(uint32_t state[4], const uint8_t block[64]);

    /* Encodes input (usigned long) into output (uint8_t). */
    static void encode(const uint32_t *input, uint8_t *output, size_t length);

    /* Decodes input (uint8_t) into output (usigned long). */
    static void decode(const uint8_t *input, uint32_t *output, size_t length);

    /* Initialization the md5 object, processing another message block,
   * and updating the context.*/
    void init(const uint8_t *input, size_t len);

    /* state (ABCD). */
    uint32_t state[4];

    /* number of bits, low-order word first. */
    uint32_t count[2];

    /* input buffer. */
    uint8_t buffer[64];

    /* message digest. */
    uint8_t *digest;

    /* padding for calculate. */
    static const uint8_t PADDING[64];

    /* Hex numbers. */
    static const char HEX_NUMBERS[16];
};

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32 - (n))))

/**
 * @Transformations for rounds 1, 2, 3, and 4.
 */
#define FF(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += F((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }
#define GG(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += G((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }
#define HH(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += H((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }
#define II(a, b, c, d, x, s, ac)            \
    {                                       \
        (a) += I((b), (c), (d)) + (x) + ac; \
        (a) = ROTATELEFT((a), (s));         \
        (a) += (b);                         \
    }
/* Define the static member of MD5_Impl. */
const uint8_t MD5_Impl::PADDING[64] = {0x80};
const char MD5_Impl::HEX_NUMBERS[16] = {
    '0', '1', '2', '3',
    '4', '5', '6', '7',
    '8', '9', 'a', 'b',
    'c', 'd', 'e', 'f'};

/**
 * @Construct a MD5_Impl object with a string.
 *
 * @param {message} the message will be transformed.
 *
 */
MD5_Impl::MD5_Impl(span<uint8_t const> message, uint8_t *digest)
    : state{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476},
      count{0, 0},
      buffer{},
      digest(digest) {

    /* Initialization the object according to message. */
    init(message.data(), message.size());
}

/**
 * @Generate md5 digest.
 *
 * @return the message-digest.
 *
 */

void MD5_Impl::GetDigest() {

    uint8_t bits[8];
    uint32_t oldState[4];
    uint32_t oldCount[2];
    uint32_t index, padLen;

    /* Save current state and count. */
    memcpy(oldState, state, MD5_SIZE);
    memcpy(oldCount, count, 8);

    /* Save number of bits */
    encode(count, bits, 8);

    /* Pad out to 56 mod 64. */
    index = (uint32_t)((count[0] >> 3) & 0x3f);
    padLen = (index < 56) ? (56 - index) : (120 - index);
    init(PADDING, padLen);

    /* Append length (before padding) */
    init(bits, 8);

    /* Store state in digest */
    encode(state, digest, MD5_SIZE);

    /* Restore current state and count. */
    memcpy(state, oldState, MD5_SIZE);
    memcpy(count, oldCount, 8);
}

/**
 * @Initialization the md5 object, processing another message block,
 * and updating the context.
 *
 * @param {input} the input message.
 *
 * @param {len} the number btye of message.
 *
 */
void MD5_Impl::init(const uint8_t *input, size_t len) {

    uint32_t i, index, partLen;

    /* Compute number of bytes mod 64 */
    index = (uint32_t)((count[0] >> 3) & 0x3f);

    /* update number of bits */
    if ((count[0] += ((uint32_t)len << 3)) < ((uint32_t)len << 3)) {
        ++count[1];
    }
    count[1] += ((uint32_t)len >> 29);

    partLen = 64 - index;

    /* transform as many times as possible. */
    if (len >= partLen) {

        memcpy(&buffer[index], input, partLen);
        transform(buffer);

        for (i = partLen; i + 63 < len; i += 64) {
            transform(&input[i]);
        }
        index = 0;

    } else {
        i = 0;
    }

    /* Buffer remaining input */
    memcpy(&buffer[index], &input[i], len - i);
}

/**
 * @MD5_Impl basic transformation. Transforms state based on block.
 *
 * @param {block} the message block.
 */
void MD5_Impl::transform(const uint8_t block[64]) {
    transform_state(state, block);
}

/**
 * @MD5_Impl basic transformation on a caller-provided state.
 *
 * @param {state} the state (ABCD) to transform in place.
 *
 * @param {block} the message block.
 */
void MD5_Impl::transform_state(uint32_t state[4], const uint8_t block[64]) {

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];

#if VSTD_MD5_LITTLE_ENDIAN
    /* The block is already a sequence of little-endian 32-bit words. */
    memcpy(x, block, 64);
#else
    decode(block, x, 64);
#endif

    /* Round 1 */
    FF(a, b, c, d, x[0], s11, 0xd76aa478);
    FF(d, a, b, c, x[1], s12, 0xe8c7b756);
    FF(c, d, a, b, x[2], s13, 0x242070db);
    FF(b, c, d, a, x[3], s14, 0xc1bdceee);
    FF(a, b, c, d, x[4], s11, 0xf57c0faf);
    FF(d, a, b, c, x[5], s12, 0x4787c62a);
    FF(c, d, a, b, x[6], s13, 0xa8304613);
    FF(b, c, d, a, x[7], s14, 0xfd469501);
    FF(a, b, c, d, x[8], s11, 0x698098d8);
    FF(d, a, b, c, x[9], s12, 0x8b44f7af);
    FF(c, d, a, b, x[10], s13, 0xffff5bb1);
    FF(b, c, d, a, x[11], s14, 0x895cd7be);
    FF(a, b, c, d, x[12], s11, 0x6b901122);
    FF(d, a, b, c, x[13], s12, 0xfd987193);
    FF(c, d, a, b, x[14], s13, 0xa679438e);
    FF(b, c, d, a, x[15], s14, 0x49b40821);

    /* Round 2 */
    GG(a, b, c, d, x[1], s21, 0xf61e2562);
    GG(d, a, b, c, x[6], s22, 0xc040b340);
    GG(c, d, a, b, x[11], s23, 0x265e5a51);
    GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
    GG(a, b, c, d, x[5], s21, 0xd62f105d);
    GG(d, a, b, c, x[10], s22, 0x2441453);
    GG(c, d, a, b, x[15], s23, 0xd8a1e681);
    GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
    GG(a, b, c, d, x[9], s21, 0x21e1cde6);
    GG(d, a, b, c, x[14], s22, 0xc33707d6);
    GG(c, d, a, b, x[3], s23, 0xf4d50d87);
    GG(b, c, d, a, x[8], s24, 0x455a14ed);
    GG(a, b, c, d, x[13], s21, 0xa9e3e905);
    GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
    GG(c, d, a, b, x[7], s23, 0x676f02d9);
    GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

    /* Round 3 */
    HH(a, b, c, d, x[5], s31, 0xfffa3942);
    HH(d, a, b, c, x[8], s32, 0x8771f681);
    HH(c, d, a, b, x[11], s33, 0x6d9d6122);
    HH(b, c, d, a, x[14], s34, 0xfde5380c);
    HH(a, b, c, d, x[1], s31, 0xa4beea44);
    HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
    HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
    HH(b, c, d, a, x[10], s34, 0xbebfbc70);
    HH(a, b, c, d, x[13], s31, 0x289b7ec6);
    HH(d, a, b, c, x[0], s32, 0xeaa127fa);
    HH(c, d, a, b, x[3], s33, 0xd4ef3085);
    HH(b, c, d, a, x[6], s34, 0x4881d05);
    HH(a, b, c, d, x[9], s31, 0xd9d4d039);
    HH(d, a, b, c, x[12], s32, 0xe6db99e5);
    HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
    HH(b, c, d, a, x[2], s34, 0xc4ac5665);

    /* Round 4 */
    II(a, b, c, d, x[0], s41, 0xf4292244);
    II(d, a, b, c, x[7], s42, 0x432aff97);
    II(c, d, a, b, x[14], s43, 0xab9423a7);
    II(b, c, d, a, x[5], s44, 0xfc93a039);
    II(a, b, c, d, x[12], s41, 0x655b59c3);
    II(d, a, b, c, x[3], s42, 0x8f0ccc92);
    II(c, d, a, b, x[10], s43, 0xffeff47d);
    II(b, c, d, a, x[1], s44, 0x85845dd1);
    II(a, b, c, d, x[8], s41, 0x6fa87e4f);
    II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
    II(c, d, a, b, x[6], s43, 0xa3014314);
    II(b, c, d, a, x[13], s44, 0x4e0811a1);
    II(a, b, c, d, x[4], s41, 0xf7537e82);
    II(d, a, b, c, x[11], s42, 0xbd3af235);
    II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
    II(b, c, d, a, x[9], s44, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

/**
* @Encodes input (unsigned long) into output (uint8_t).
*
* @param {input} usigned long.
*
* @param {output} uint8_t.
*
* @param {length} the length of input.
*
*/
void MD5_Impl::encode(const uint32_t *input, uint8_t *output, size_t length) {

#if VSTD_MD5_LITTLE_ENDIAN
    /* Output words are already in the target byte order. */
    memcpy(output, input, length);
#else
    for (size_t i = 0, j = 0; j < length; ++i, j += 4) {
        output[j] = (uint8_t)(input[i] & 0xff);
        output[j + 1] = (uint8_t)((input[i] >> 8) & 0xff);
        output[j + 2] = (uint8_t)((input[i] >> 16) & 0xff);
        output[j + 3] = (uint8_t)((input[i] >> 24) & 0xff);
    }
#endif
}

/**
 * @Decodes input (uint8_t) into output (usigned long).
 *
 * @param {input} bytes.
 *
 * @param {output} unsigned long.
 *
 * @param {length} the length of input.
 *
 */
void MD5_Impl::decode(const uint8_t *input, uint32_t *output, size_t length) {
    for (size_t i = 0, j = 0; j < length; ++i, j += 4) {
        output[i] = ((uint32_t)input[j]) | (((uint32_t)input[j + 1]) << 8) | (((uint32_t)input[j + 2]) << 16) | (((uint32_t)input[j + 3]) << 24);
    }
}

/**
 * @One-shot md5 for short messages. Builds the padded block(s) directly
 * on the stack (message bytes + 0x80 + zeros + 64-bit little-endian bit
 * length), transforms them from the initial state and encodes the result,
 * bypassing the streaming context setup and finalize. Only valid for
 * messages of at most ONESHOT_MAX_LEN bytes.
 *
 * @param {message} the message (message.size() <= ONESHOT_MAX_LEN).
 *
 * @param {digest} the 16-byte message-digest output.
 */
static void md5_oneshot(span<uint8_t const> message, uint8_t *digest) {

    uint32_t st[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    uint8_t blocks[128] = {};
    size_t const len = message.size();

    memcpy(blocks, message.data(), len);
    blocks[len] = 0x80;

    /* Append the bit length as a 64-bit little-endian integer at byte 56
       of the last block; padding longer than 55 bytes spills over into a
       second block. */
    uint64_t const bits = (uint64_t)len << 3;
    uint8_t *const len_ptr = blocks + (len <= 55 ? 56 : 120);
    for (uint32_t i = 0; i < 8; ++i) {
        len_ptr[i] = (uint8_t)(bits >> (i << 3));
    }

    MD5_Impl::transform_state(st, blocks);
    if (len > 55) {
        MD5_Impl::transform_state(st, blocks + 64);
    }

    MD5_Impl::encode(st, digest, MD5_SIZE);
}

/**
 * @Convert digest to string value.
 *
 * @return the hex string of digest.
 *
 */

#undef F
#undef G
#undef H
#undef I
#undef FF
#undef GG
#undef HH
#undef II
#undef ROTATELEFT
#undef VSTD_MD5_LITTLE_ENDIAN

}// namespace detail
std::array<uint8_t, MD5_SIZE> GetMD5FromString(string const &str) {
    using namespace detail;
    std::array<uint8_t, MD5_SIZE> arr{};
    if (str.size() <= MD5_Impl::ONESHOT_MAX_LEN) {
        md5_oneshot({reinterpret_cast<uint8_t const *>(str.data()), str.size()}, arr.data());
    } else {
        MD5_Impl md5({reinterpret_cast<uint8_t const *>(str.data()), str.size()}, arr.data());
        md5.GetDigest();
    }
    return arr;
}
std::array<uint8_t, MD5_SIZE> GetMD5FromArray(span<uint8_t const> data) {
    using namespace detail;
    std::array<uint8_t, MD5_SIZE> arr{};
    if (data.size() <= MD5_Impl::ONESHOT_MAX_LEN) {
        md5_oneshot(data, arr.data());
    } else {
        MD5_Impl md5(data, arr.data());
        md5.GetDigest();
    }
    return arr;
}
// NOTE: do not route these through a delegating MD5(span) constructor —
// doing so made clang inline the one-shot path into one thunk but not the
// other, leaving the string_view entry ~15 ns slower for identical work.
// Keep each entry point's branch explicit (same shape as the two free
// functions above).
MD5::MD5(string const &str)
    : data{} {
    using namespace detail;
    span<uint8_t const> bin{reinterpret_cast<uint8_t const *>(str.data()), str.size()};
    if (bin.size() <= MD5_Impl::ONESHOT_MAX_LEN) {
        md5_oneshot(bin, reinterpret_cast<uint8_t *>(&data));
    } else {
        MD5_Impl md5(bin, reinterpret_cast<uint8_t *>(&data));
        md5.GetDigest();
    }
}
MD5::MD5(std::string_view str)
    : data{} {
    using namespace detail;
    span<uint8_t const> bin{reinterpret_cast<uint8_t const *>(str.data()), str.size()};
    if (bin.size() <= MD5_Impl::ONESHOT_MAX_LEN) {
        md5_oneshot(bin, reinterpret_cast<uint8_t *>(&data));
    } else {
        MD5_Impl md5(bin, reinterpret_cast<uint8_t *>(&data));
        md5.GetDigest();
    }
}
MD5::MD5(span<uint8_t const> bin)
    : data{} {
    using namespace detail;
    if (bin.size() <= MD5_Impl::ONESHOT_MAX_LEN) {
        md5_oneshot(bin, reinterpret_cast<uint8_t *>(&data));
    } else {
        MD5_Impl md5(bin, reinterpret_cast<uint8_t *>(&data));
        md5.GetDigest();
    }
}
MD5::MD5(MD5Data const &data)
    : data(data) {
}

bool MD5::operator==(MD5 const &m) const {
    return data.data0 == m.data.data0 && data.data1 == m.data.data1;
}
bool MD5::operator!=(MD5 const &m) const {
    return data.data0 != m.data.data0 || data.data1 != m.data.data1;
}
string MD5::to_string(bool upper) const {
    string str;
    vstd::StringUtil::to_hex_string(
        {reinterpret_cast<uint8_t const *>(&data),
         sizeof(data)},
        str, upper);
    return str;
}
#ifdef EXPORT_UNITY_FUNCTION
VENGINE_UNITY_EXTERN void unity_get_md5(
    uint8_t *data,
    uint64 dataLength,
    uint8_t *destData) {
    detail::MD5_Impl md5(span<uint8_t>(data, dataLength), destData);
    md5.GetDigest();
}
VENGINE_UNITY_EXTERN void md5_to_std::string(
    uint64_t const *md5,
    char *result,
    bool upper) {
    using namespace detail;
    UInt64ToHex(md5[0], result, upper);
    UInt64ToHex(md5[1], result, upper);
}
#endif
}// namespace vstd
