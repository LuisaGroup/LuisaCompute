/**
 * @file md5.h
 * @The header file of md5.
 * @author Jiewei Wei
 * @mail weijieweijerry@163.com
 * @github https://github.com/JieweiWei
 * @data Oct 19 2014
 *
 */
#pragma once
/* Parameters of MD5. */

/**
 * @Basic MD5 functions.
 *
 * @param there uint32_t.
 *
 * @return one uint32_t.
 */


#include <stdint.h>
#include <span>
#include <array>

/* Define of btye.*/
/* Define of uint8_t. */

class MD5 {
public:
	/* Construct a MD5 object with a string. */
	MD5(std::span<uint8_t> message);

	/* Generate md5 digest. */
	std::array<uint8_t, 16> const& GetDigest();

private:
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

	/* Initialization the md5 object, processing another message block,
   * and updating the context.*/
	void init(const uint8_t* input, size_t len);

	/* MD5 basic transformation. Transforms state based on block. */
	void transform(const uint8_t block[64]);

	/* Encodes input (usigned long) into output (uint8_t). */
	void encode(const uint32_t* input, uint8_t* output, size_t length);

	/* Decodes input (uint8_t) into output (usigned long). */
	void decode(const uint8_t* input, uint32_t* output, size_t length);

private:
	/* Flag for mark whether calculate finished. */
	bool finished;

	/* state (ABCD). */
	uint32_t state[4];

	/* number of bits, low-order word first. */
	uint32_t count[2];

	/* input buffer. */
	uint8_t buffer[64];

	/* message digest. */
	std::array<uint8_t, 16> digest;

	/* padding for calculate. */
	static const uint8_t PADDING[64];

	/* Hex numbers. */
	static const char HEX_NUMBERS[16];
};