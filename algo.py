import uuid
import hashlib
import secrets
import hmac
import math
import pickle
import struct
import binascii


class DRBG(object): #이거시 DBRG(HMAC-DRBG) 입네당 | Copyright (c) 2014 David Lazar <lazard@mit.edu> under the MIT license.
    def __init__(self, seed):
        self.key = b'\x00' * 64
        self.val = b'\x01' * 64
        self.reseed(seed)

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate(self, n):
        xs = b''
        while len(xs) < n:
            self.val = self.hmac(self.key, self.val)
            xs += self.val

        self.reseed()

        return xs[:n]


nonce = [[i for i in range(4)] for j in range(0,4)] #NONCE : 한번만 쓰이는 값
# https://crypto.stackexchange.com/questions/20135/what-is-salsa20-nonce-and-its-requirements

def hash_table(key): # mod 256. KEY는 바이트로 줘라.
    # WIP
    """
    with open('rn.txt') as f:
        array = [int(line) for line in f]
    print(array)
    """
    drbg = DRBG(key)
    return [int.from_bytes(drbg.generate(1), "little") for i in range(256)]

def generate_nonce():
    """Generate pseudorandom number."""
    return secrets.token_hex() # '773027c84b50528e0382d81166f2542ccf8ae3abc014cb42b8cb52605ab563a6'

def generate_key(key):
    """Generate key in hex format"""
    hashlib.scrypt(key, salt='ksdghfkuw').hex #요게 키에서 derive한 해시
    
def convert_to_stat(data):
    data = data.encode('utf-8')     # 이젠 data 꼴이라구!
    return [[data[4*i+16*j:4*i+4+16*j] for i in xrange(4)] for j in range(4)]

#data = b"old loose universal chest avoid paste height earn mill tie refer"
secret = b"OMGWFTAMIDOING"
split_size = 1#in bytes

def gen_hash_table(secret, filename):
    with open(filename, "w") as f:
        print(hash_table(generate_key(secret)), file = f)

def encryptPiece(table, cube): #table : size 64 array with 4^3 size cubes 
    result = [[[b"\x00"*split_size for i in range(4)] for j in range(4)] for k in range(4)]
    rot = [[[b"\x00"*split_size for i in range(4)] for j in range(4)] for k in range(4)]
    #cube = [[[data[(k*16+4*j)+i:(k*16+4*j)+i+split_size] for i in range(0, 4*split_size, split_size)] for j in range(0, 4*split_size, split_size)] for k in range(0, 4*split_size, split_size)]
    #z-axis rotation
    for x in range(4):
        for y in range(4):
            for z in range(4):
                
                rot[3-y][3-x][z] = cube[x][y][z]
        #print("Moving", cubes[i], "from", i, "to", table[i])
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x_p][y_p][z_p] = rot[x][y][z]
    #y-axis rotation
    for x in range(4):
        for y in range(4):
            for z in range(4):
                rot[3-z][y][3-x] = result[x][y][z]
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x_p][y_p][z_p] = rot[x][y][z]
    #x-axis rot
    for x in range(4):
        for y in range(4):
            for z in range(4):
                rot[x][3-z][3-y] = result[x][y][z]
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x_p][y_p][z_p] = rot[x][y][z]
    array = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                array.append(result[x][y][z])
    return array
'''
def decryptPiece(table, data):
    result = [[[b"\x00"*split_size for i in range(4)] for j in range(4)] for k in range(4)]
    rot = [[[b"\x00"*split_size for i in range(4)] for j in range(4)] for k in range(4)]
    cube = data.copy()
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x][y][z] = cube[x_p][y_p][z_p]
    #x-axis rot
    for x in range(4):
        for y in range(4):
            for z in range(4):
                rot[x][3-z][3-y] = result[x][y][z]
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x][y][z] = rot[x_p][y_p][z_p]
    #y-axis rotation
    for x in range(4):
        for y in range(4):
            for z in range(4):
                rot[3-z][y][3-x] = result[x][y][z]
    #twist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                destination = table[(x*16+4*y)+z]
                x_p = destination%4
                y_p = (destination//4)%4
                z_p = destination//16
                result[x][y][z] = rot[x_p][y_p][z_p]
    #z-axis rotation
    for x in range(4):
        for y in range(4):
            for z in range(4):
                rot[3-y][3-x][z] = result[x][y][z]
    for x in range(4):
        for y in range(4):
            for z in range(4):
                cube[x][y][z] = rot[x][y][z]
    array = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                array.append(cube[x][y][z])
    return array
'''
def encrypt(table, data): #Takes 128 byte * UNIT data
    l = data[:64*split_size]
    r = data[64*split_size:]
    #print(l,r)
    for i in range(6):
        l_cube = [[[l[(k*16+4*j)+i:(k*16+4*j)+i+split_size][0] for i in range(0, 4*split_size, split_size)] for j in range(0, 4*split_size, split_size)] for k in range(0, 4*split_size, split_size)]
        #r = encryptPiece(table,r_cube)
        (l,r) = ([a^b for (a,b) in zip(r,encryptPiece(table,l_cube))],l)
        #(l,r) = (r,[a^b for (a,b) in zip(l,r)])
    l.extend(r)
    return l

def decrypt(table, data): #Takes 128 byte * UNIT data
    l = data[:64*split_size]
    r = data[64*split_size:]
    #print(l,r)
    for i in range(6):
        #l_cube = [[[l[(k*16+4*j)+i:(k*16+4*j)+i+split_size][0] for i in range(0, 4*split_size, split_size)] for j in range(0, 4*split_size, split_size)] for k in range(0, 4*split_size, split_size)]
        #(l,r) = (r,[a^b for (a,b) in zip(r,l)])
        r_cube = [[[r[(k*16+4*j)+i:(k*16+4*j)+i+split_size][0] for i in range(0, 4*split_size, split_size)] for j in range(0, 4*split_size, split_size)] for k in range(0, 4*split_size, split_size)]
        #(l,r) = (l,decryptPiece(table, r_cube))
        #(r,l) = (l,decryptPiece(table,r_cube))
        (l,r) = (r,[a^b for (a,b) in zip(l,encryptPiece(table,r_cube))])
        #(r,l) = (l,[a^b for (a,b) in zip(r,l)])
    l.extend(r)
    return l

#시간 재기
def fernet():
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    cipher_text = cipher_suite.encrypt(data)

def AESTest(key):
    cipher = AES.new(key, AES.MODE_CFB, iv)
    msg = iv + cipher.encrypt(data)

def BlowfishTest():
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    plen = bs - divmod(len(data),bs)[1]
    padding = [plen]*plen
    padding = pack('b'*plen, *padding)
    msg = iv + cipher.encrypt(data + padding)


def save(data, filename):
    #우왕 피클
    with open(filename, "wb") as f:
        return

def load(filename):
    data = b"Don't look back, giving your commodities an angel - i don;t ever look down, i dont wanna leave this town, this city never sleeps tonight"
    #cube = [[[data[(k*16+4*j)+i:(k*16+4*j)+i+split_size] for i in range(0, 4*split_size, split_size)] for j in range(0, 4*split_size, split_size)] for k in range(0, 4*split_size, split_size)]
    with open(filename, "rb") as f:
        data = f.read()
    return data


#Uncomment below to encrypt string(TRUNCATED TO FIRST 128 bytes)
data = b"Don't look back, giving your commodities an angel - i don;t ever look down, i dont wanna leave this town, this city never sleeps"  

#Uncomment below to load from file
# filename = "topsecret.txt"
# data = load(filename)

#Optionally, set chuck size
# split_size = 1#in bytes

with open("key.txt","r") as f:
    table = [int(i) for i in f.read().split()]

# res = encryptPiece(table, data)
# decryptPiece(table, res)

e = encrypt(table, data)
print(bytes(decrypt(table, e)))
# gen_hash_table(secret)

iteration = 1

import timeit
from struct import pack
from Crypto.Cipher import AES
from Crypto import Random
from cryptography.fernet import Fernet
from Crypto.Cipher import Blowfish
import aespy
print(timeit.timeit('encrypt(table, data)', globals=globals(), number=iteration))
print(timeit.timeit('fernet()', globals=globals(), number=iteration))
#Prep for AES
iv = Random.new().read(AES.block_size)
key, hmac_key, iv = aespy.get_key_iv(b"Sixteen byte key", b'ksdghfkuw', 10000)
print(timeit.timeit('aespy.AES(key).encrypt_cbc(data, iv)', globals=globals(), number=iteration))
#Prep for Blowfish
bs = Blowfish.block_size
iv = Random.new().read(bs)
key = b'An arbitrarily long key'
print(timeit.timeit('BlowfishTest()', globals=globals(), number=iteration))