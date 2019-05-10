
# tiny.py



import gzip

from functools import partial, reduce
from operator import mul
from numpy import *
from timing import Timer
import os



SIGNED_TYPES = set([int8, int16, int32])



def pack_int(A, datatype, low=None, high=None):

	itemsize = dtype(datatype).itemsize
	high = high if high is not None else max(A)
	low = low if low is not None else min(A)
	bits = int(ceil(log2(high + 1 - low)))

	# handle signed data types

	if datatype in SIGNED_TYPES: A = A + 2 ** (bits - 1)

	# cast A to numpy array

	A = array(A, dtype='u{0}'.format(itemsize))

	# add dimension to store unpacked bits and convert to ordered bytes

	A = expand_dims(A, axis=-1).byteswap().view(uint8)

	# convert to bitarray

	A = unpackbits(A, axis=-1)

	# take only last bits and flatten

	A = A[..., -bits:].flatten()

	# repack bits and return bytes

	return packbits(A).tobytes()



def packbitarray(A, datatype, low=None, high=None):

	itemsize = dtype(datatype).itemsize
	high = high if high is not None else max(A)
	low = low if low is not None else min(A)
	bits = int(ceil(log2(high + 1 - low)))

	# handle signed data types

	if datatype in SIGNED_TYPES: A = A + 2 ** (bits - 1)

	# cast A to numpy array

	A = array(A, dtype='u{0}'.format(itemsize))

	# add dimension to store unpacked bits and convert to ordered bytes

	A = expand_dims(A, axis=-1).byteswap().view(uint8)

	# convert to bitarray

	A = unpackbits(A, axis=-1)

	# take only last bits and flatten

	A = A[..., -bits:].flatten()

	# repack bits and return bytes

	return packbits(A).tobytes()


def unpack_int(A, datatype, low, high, shape=(-1,)):

	itemsize = dtype(datatype).itemsize
	high = high if high is not None else max(A)
	low = low if low is not None else min(A)
	bits = int(ceil(log2(high + 1 - low)))
	shape = (*shape, bits)

	# get uint8's from bytes

	A = frombuffer(A, dtype=uint8)

	# unpack bit arrays

	A = unpackbits(A)

	# remove extra bits created when unpacking

	A = A[:-(remainder(len(A), bits)) or None]

	# reshape into original shape

	A = reshape(A, shape)

	# pad last axis with extra leading 0's

	A = pad(A, [(0, 0)] * (len(shape) - 1) + [(8 * itemsize - bits, 0)], 'constant', constant_values=0)

	# pack bits back into uint8'ss

	A = packbits(A, axis=-1)

	# consolidate uint8's to larger datatype (requires byte swap)

	A = A.view(dtype='u{0}'.format(itemsize)).byteswap()

	# squeeze off extra index

	A = A.squeeze(axis=-1)

	# correct data type

	A = array(A, dtype=datatype)

	# calculate offset for signed data types

	if datatype in SIGNED_TYPES: A = A - 2 ** (bits - 1)

	# return corrected array

	return A



if __name__ == '__main__':

	high = int(2)
	datatype = uint8

	a = array(random.randint(0, high, size=(1000000, 1000)), dtype=datatype)
	print(a)

	with Timer(callback="Time to save tiny gzip: {time:.3f}s") as t:

		x = pack_int(a, datatype=datatype, low=0, high=high)

		with gzip.open('tiny_gzip.gzip', 'wb') as f:

			f.write(x)

	print(os.path.getsize('tiny_gzip.gzip'))

	with Timer(callback="Time to save gzip only: {time:.3f}s") as t:

		with gzip.open('gzip_only.gzip', 'wb') as f:

			f.write(a)

	print(os.path.getsize('gzip_only.gzip'))

	with Timer(callback="Time to read tiny gzip: {time:.3f}s") as t:

		with gzip.open('tiny_gzip.gzip', 'rb') as f:

			a = unpack_int(f.read(), datatype=datatype, low=0, high=high, shape=(-1, 5))
			print(a)

	with Timer(callback="Time to read gzip only: {time:.3f}s") as t:

		with gzip.open('gzip_only.gzip', 'rb') as f:

			f.read()
