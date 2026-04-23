import os
from ast import Num
from asyncio.windows_events import NULL
from ctypes import util
from sre_constants import NOT_LITERAL
import string

KEY = int(71284230948672)
RADIX = int(35)

def isEncrypted(id: string):
    data = str(id)
    for ch in data:
        if ch >= 'A' and ch <'Z': 
            return True
    return False    

def encrypt(id: Num):
    for i in range(4):
        id ^= KEY
        id <<= 3
    return str_base(id, RADIX).upper()

def decrypt(encryptId: string):
    id = -1
    id = int(encryptId, RADIX)
    for i in range(4):
        id >>= 3
        id ^= KEY
    return id


def digit_to_char(digit):
    if digit < 10:
        return str(digit)
    return chr(ord('a') + digit - 10)

def str_base(number, base):
    if number < 0:
        return '-' + str_base(-number, base)
    (d, m) = divmod(number, base)
    if d > 0:
        return str_base(d, base) + digit_to_char(m)
    return digit_to_char(m)

if __name__ == "__main__":
    print(isEncrypted(int(123)))
    print(isEncrypted('3FO4K4VY163J'))
    print(encrypt(int(123)))
    print(decrypt('3FO4K4VY163J'))