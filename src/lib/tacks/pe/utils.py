# -*- coding: utf-8 -*-
"""Utils to handle PE files.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""

import numpy as np


DWORD_VALUES = np.array([1, 2**8, 2**16, 2**24])
WORD_VALUES = np.array([1, 2**8])


def word_to_value(word):
    """Convert a word (2 bytes) or a DWORD (4 bytes) into a decimal value.

    Parameters
    ----------
    word : bytes
        Word to convert.

    Returns
    -------
    scalar
        Value of the word in the decimal basis.
    """
    return sum([byte * DWORD_VALUES[idb] for idb, byte in enumerate(word)])


def value_to_word(value, n_bytes=4):
    """Convert a value into an array bytes.

    Word can be encoded using 2 bytes (WORD) or 4 bytes (DWORD), in
    little-endian format.

    Parameters
    ----------
    scalar
        Decimal value of the word.

    Returns
    -------
    bytes
    """
    hex_value = hex(value)[2::]
    hex_str = '0' * (2 * n_bytes - len(hex_value)) + hex_value
    np_bytez = np.array([int(hex_str[i:(i + 2)], base=16)
                         for i in range(0, 2 * n_bytes, 2)][::-1], np.uint8)
    return np_bytez.tobytes()


def get_architecture(bytez):
    """Get the architecture of the PE file."""
    pe_offset = get_pe_offset(bytez)

    arch_bytes = bytez[(pe_offset + 4):(pe_offset + 6)]
    if arch_bytes == b'\x4c\x01':
        return 32
    elif arch_bytes == b'\x64\x86':
        return 64
    else:
        err_msg = 'Unknown architecture: {}'
        raise ValueError(err_msg.format(arch_bytes))


def get_pe_offset(bytez):
    """Get offset to the PE header."""
    return word_to_value(bytez[60:64])


def get_oh_offset(bytez):
    """Get offset to the optional header."""
    return get_pe_offset(bytez) + 24


def get_dd_offset(bytez):
    """Get offset to the data directories."""
    oh_offset = get_oh_offset(bytez)
    architecture = get_architecture(bytez)

    if architecture == 32:
        return oh_offset + 96
    elif architecture == 64:
        return oh_offset + 112


def get_st_offset(bytez):
    """Get offset to the section table."""
    return get_dd_offset(bytez) + 128


def get_mutable_offsets(pe):
    """Return a list of offsets that can be changed without altering the files.

    Parameters
    ----------

    Returns
    -------

    """
    list_offsets = []

    # DOS header
    list_offsets += range(4, 60)

    # DOS stub
    pe_entrypoint = pe.DOS_HEADER.e_lfanew
    list_offsets += range(64, pe_entrypoint)

    return list_offsets


def add_section(bytez, name, content, readable=False, writable=False,
                executable=False):
    """Add a new empty section.

    Parameters
    ----------
    bytez : bytes
        Sequence of bytes.
    name : str
        Name of the section.
    content : bytes
        Content of the section.
    readable : bool, optional
        Indicates if the section is readable or not.
    writable : bool, optional
        Indicates if the section is writable or not.
    executable : bool, optional
        Indicates if the section is executable or not.

    Returns
    -------
    bytes
    """
    pe_offset = get_pe_offset(bytez)
    oh_offset = get_oh_offset(bytez)
    st_offset = get_st_offset(bytez)

    np_bytez = np.frombuffer(bytez, dtype=np.uint8)

    n_sections = word_to_value(bytez[pe_offset + 6:pe_offset + 8])
    header_size = word_to_value(bytez[oh_offset + 60:oh_offset + 64])

    # copy the first section
    new_section = np_bytez[st_offset:(st_offset + 40)].copy()
    # update section name
    new_section[0:8] = np.pad(
        np.frombuffer(name.encode(), dtype=np.uint8),
        (0, 8 - len(name)))

    # virtual size
    if executable:
        new_section[8:12] = np.frombuffer(value_to_word(len(content)),
                                          dtype=np.uint8)
    else:
        new_section[8:12] = np.zeros(4, dtype=np.uint8)

    # virtual address
    new_section[12:16] = np.zeros(4, dtype=np.uint8)

    # raw size
    new_section[16:20] = np.frombuffer(value_to_word(len(content)),
                                       dtype=np.uint8)

    # pointer to raw data
    new_section[20:24] = np.zeros(4, dtype=np.uint8)

    # characteristics
    new_section[36:40] = np.frombuffer(value_to_word(
        executable * 20 + readable * 40 + writable * 80), dtype=np.uint8)

    # add the section
    new_section_offset = st_offset + n_sections * 40

    np_bytez = np.concatenate([np_bytez[0:new_section_offset],
                               new_section,
                               np_bytez[(new_section_offset)::]])

    # update the number of sections
    np_bytez[pe_offset + 6: pe_offset + 8] = np.frombuffer(
        value_to_word(n_sections + 1, n_bytes=2), dtype=np.uint8)

    # update the size of the PE header
    np_bytez[(oh_offset + 60):(oh_offset + 64)] = np.frombuffer(
        value_to_word(header_size + 40), dtype=np.uint8)

    return np_bytez.tobytes() + content


def remove_section(bytez, name):
    """Remove a section.

    """
    pe_offset = get_pe_offset(bytez)
    oh_offset = get_oh_offset(bytez)
    st_offset = get_st_offset(bytez)

    np_bytez = np.frombuffer(bytez, dtype=np.uint8).copy()

    n_sections = word_to_value(bytez[pe_offset + 6:pe_offset + 8])
    header_size = word_to_value(bytez[oh_offset + 60:oh_offset + 64])

    # copy the first section
    for no_section in range(n_sections):

        np_name = np_bytez[(st_offset + 40 * no_section):
                           (st_offset + 40 * no_section + 8)]

        if np_name == np.frombuffer(name.encode(), dtype=np.uint8):
            print('Found')
            np_bytez = np.concatenate(
                [np_bytez[0:(st_offset + 40 * no_section)],
                 np_bytez[(st_offset + 40 * (no_section + 1))::]])

    # update the number of sections
    np_bytez[pe_offset + 6: pe_offset + 8] = np.frombuffer(
        value_to_word(n_sections - 1, n_bytes=2), dtype=np.uint8)

    # update the size of the PE header
    np_bytez[(oh_offset + 60):(oh_offset + 64)] = np.frombuffer(
        value_to_word(header_size - 40), dtype=np.uint8)

    return np_bytez.tobytes()
