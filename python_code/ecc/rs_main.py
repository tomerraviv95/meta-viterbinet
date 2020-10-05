### code from site https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders

import numpy as np

from python_code.ecc.polynomials_manipulation import init_tables, convert_binary_to_field, convert_field_to_binary
from python_code.ecc.rs_decoder import rs_calc_syndromes, rs_find_error_locator, rs_find_errors, rs_correct_errata
from python_code.ecc.rs_encoder import rs_encode_msg


def encode(binary_word: np.ndarray):
    init_tables()
    symbols_word = convert_binary_to_field(binary_word)
    symbols_codeword = rs_encode_msg(symbols_word, nsym=32)
    return convert_field_to_binary(symbols_codeword)


def decode(binary_rx):
    init_tables()
    symbols_rx = convert_binary_to_field(binary_rx.astype(int))
    synd = rs_calc_syndromes(symbols_rx, nsym=32)
    err_loc = rs_find_error_locator(synd, nsym=32)
    pos = rs_find_errors(err_loc[::-1], len(symbols_rx))  # find the errors locations
    corrected_word = rs_correct_errata(symbols_rx, synd, pos)
    return convert_field_to_binary(corrected_word)[:1784]


if __name__ == "__main__":
    ## simple testing
    words = np.random.randint(0, 2, [1784])
    tx = encode(words)
    errors = np.zeros(2040).astype(int)
    errors_ind = np.array([0, 8, 16])  # np.sort(np.random.choice(2040, errors_num, replace=False))
    errors[errors_ind] = 1
    print(f'generated errors at locations: {errors_ind // 8}')
    binary_rx = (tx + errors) % 2
    corrected_word = decode(binary_rx)
    flips_num = np.sum(np.abs(words - corrected_word))
    print("flips from original word after decoding: " + str(flips_num))
