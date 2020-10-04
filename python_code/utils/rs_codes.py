### encoding code from site https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders

from python_code.utils.berlekamp_massey import BerlekampMasseyHardDecoder
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UINT8 = 256

def get_generator_poly():
    gf_symbols = [1, 232, 29, 189, 50, 142, 246, 232, 15, 43, 82, 164, 238, 1, 158, 13, 119, 158, 224, 134, 227, 210,
                  163, 50, 107, 40, 27, 104, 253, 24, 239, 216, 45]
    array_symbols = np.array(gf_symbols, dtype=np.uint8)
    unpacked_bits = np.unpackbits(array_symbols.reshape(-1, 1), axis=1)
    gf = unpacked_bits.flatten()[7:]
    return gf


def gf_mult_noLUT(x, y, prim=0, field_charac_full=256, carryless=True):
    '''Galois Field integer multiplication using Russian Peasant Multiplication algorithm (faster than the standard multiplication + modular reduction).
    If prim is 0 and carryless=False, then the function produces the result for a standard integers multiplication (no carry-less arithmetics nor modular reduction).'''
    r = 0
    while y:  # while y is above 0
        if y & 1: r = r ^ x if carryless else r + x  # y is odd, then add the corresponding x to r (the sum of all x's corresponding to odd y's will give the final product). Note that since we're in GF(2), the addition is in fact an XOR (very important because in GF(2) the multiplication and additions are carry-less, thus it changes the result!).
        y = y >> 1  # equivalent to y // 2
        x = x << 1  # equivalent to x*2
        if prim > 0 and x & field_charac_full: x = x ^ prim  # GF modulo: if x >= 256 then apply modular reduction using the primitive polynomial (we just subtract, but since the primitive number can be above 256 then we directly XOR).

    return r


def init_tables(prim=0x11d):
    global gf_exp, gf_log
    gf_exp = [0] * 2 * UINT8  # Create list of 512 elements. In Python 2.6+, consider using bytearray
    gf_log = [0] * UINT8
    '''Precompute the logarithm and anti-log tables for faster computation later, using the provided primitive polynomial.'''
    # prim is the primitive (binary) polynomial. Since it's a polynomial in the binary sense,
    # it's only in fact a single galois field value between 0 and 255, and not a list of gf values.

    gf_exp = [0] * 2 * UINT8  # anti-log (exponential) table
    gf_log = [0] * UINT8  # log table
    # For each possible value in the galois field 2^8, we will pre-compute the logarithm and anti-logarithm (exponential) of this value
    x = 1
    for i in range(0, UINT8 - 1):
        gf_exp[i] = x  # compute anti-log for this value and store it in a table
        gf_log[x] = i  # compute log at the same time
        x = gf_mult_noLUT(x, 2, prim)

        # If you use only generator==2 or a power of 2, you can use the following which is faster than gf_mult_noLUT():
        # x <<= 1 # multiply by 2 (change 1 by another number y to multiply by a power of 2^y)
        # if x & 0x100: # similar to x >= 256, but a lot faster (because 0x100 == 256)
        # x ^= prim # substract the primary polynomial to the current value (instead of 255, so that we get a unique set made of coprime numbers), this is the core of the tables generation

    # Optimization: double the size of the anti-log table so that we don't need to mod 255 to
    # stay inside the bounds (because we will mainly use this table for the multiplication of two GF numbers, no more).
    for i in range(UINT8 - 1, 2 * UINT8):
        gf_exp[i] = gf_exp[i - (UINT8 - 1)]


def gf_mul(x, y):
    if x == 0 or y == 0:
        return 0
    return gf_exp[gf_log[x] + gf_log[y]]  # should be gf_exp[(gf_log[x]+gf_log[y])%255] if gf_exp wasn't oversized


def gf_poly_div(dividend, divisor):
    '''Fast polynomial division by using Extended Synthetic Division and optimized for GF(2^p) computations
    (doesn't work with standard polynomials outside of this galois field, see the Wikipedia article for generic algorithm).'''
    # CAUTION: this function expects polynomials to follow the opposite convention at decoding:
    # the terms must go from the biggest to lowest degree (while most other functions here expect
    # a list from lowest to biggest degree). eg: 1 + 2x + 5x^2 = [5, 2, 1], NOT [1, 2, 5]

    msg_out = list(dividend)  # Copy the dividend
    # normalizer = divisor[0] # precomputing for performance
    for i in range(0, len(dividend) - (len(divisor) - 1)):
        # msg_out[i] /= normalizer # for general polynomial division (when polynomials are non-monic), the usual way of using
        # synthetic division is to divide the divisor g(x) with its leading coefficient, but not needed here.
        coef = msg_out[i]  # precaching
        if coef != 0:  # log(0) is undefined, so we need to avoid that case explicitly (and it's also a good optimization).
            for j in range(1, len(
                    divisor)):  # in synthetic division, we always skip the first coefficient of the divisior,
                # because it's only used to normalize the dividend coefficient
                if divisor[j] != 0:  # log(0) is undefined
                    msg_out[i + j] ^= gf_mul(divisor[j], coef)  # equivalent to the more mathematically correct
                    # (but xoring directly is faster): msg_out[i + j] += -divisor[j] * coef

    # The resulting msg_out contains both the quotient and the remainder, the remainder being the size of the divisor
    # (the remainder has necessarily the same degree as the divisor -- not length but degree == length-1 -- since it's
    # what we couldn't divide from the dividend), so we compute the index where this separation is, and return the quotient and remainder.
    separator = -(len(divisor) - 1)
    return msg_out[:separator], msg_out[separator:]  # return quotient, remainder.


def rs_encode_msg(msg_in):
    '''Reed-Solomon main encoding function'''
    init_tables()
    gen = get_generator_poly()
    msg_in = [int(msg_in_bit) for msg_in_bit in msg_in.flatten()]
    # Pad the message, then divide it by the irreducible generator polynomial
    _, remainder = gf_poly_div(msg_in + [0] * (len(gen) - 1), gen)
    # The remainder is our RS code! Just append it to our original message to get our full codeword (this represents a polynomial of max 256 terms)
    msg_out = msg_in + remainder
    # Return the codeword
    return np.array(msg_out, dtype=int).reshape(1,-1)


if __name__ == "__main__":
    ## simple testing
    errors_num = 15
    gf = get_generator_poly()
    words = np.random.randint(0, 2, [1, 1784], dtype=int)
    dec = BerlekampMasseyHardDecoder(255, 16)

    tx = rs_encode_msg(words)
    errors = np.zeros(2040)
    errors_ind = np.sort(np.random.choice(2040, errors_num, replace=False).astype(int))
    errors[errors_ind] = 1
    print(f'generated errors at locations: {errors_ind}')

    rx = (tx + errors) % 2
    hard_decoded_x = torch.Tensor([rx]).int().to(device=device)
    print("received word: " + str(hard_decoded_x))

    corrected_word = dec.forward(hard_decoded_x.reshape(8, 255)).reshape(-1)
    print("corrected word: " + str(corrected_word))

    flips_num = torch.sum(torch.abs(corrected_word - hard_decoded_x))
    print("flips: " + str(flips_num))

    torch_target = torch.IntTensor(tx).cuda()
    errors_num = torch.sum(torch.abs(corrected_word - torch_target))
    print("errors: " + str(errors_num))
    print(f'found them at locations: {torch.where(torch.abs(corrected_word - torch_target))}')
