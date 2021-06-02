from __future__ import division
from FixedPoint import FXfamily, FXnum
import struct
import codecs

def float_to_hex(f: float):
	# Courtesy of https://stackoverflow.com/a/23624284
	return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]

# convert threshold to fixed number
def float_to_fixed_hex(f, frac_nbits, int_nbits):
    FixedPointFamily = FXfamily(n_bits=frac_nbits, n_intbits=int_nbits)
    translatedNum = FixedPointFamily(f).toBinaryString(logBase=1)
    translatedNum = translatedNum.split('.')
    bin_res = ''.join(translatedNum)
    hex_res = hex(int(bin_res, 2))[2:]
    return hex_res

def hex_to_float(x):
	# still from stackoverflow
    return struct.unpack('!f', codecs.decode(x,'hex'))[0]
