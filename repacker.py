
# zip.py



import math
import numpy as np

from functools import reduce
from operator import mul



def packbits(A, bits=8):

	# takes an array-like of uint8 and number of bits to encode with

	A = np.array(A, dtype=np.uint8)
	if bits == 8: return A.flatten()

	# get shape and axis for array manipulation

	shape = A.shape
	axis = len(shape)

	# add dimension to store unpacked bits

	A = np.expand_dims(A, axis=axis)

	# convert to bitarray, take only last bits, and flatten

	A = np.unpackbits(A, axis=axis)[..., -bits:].flatten()

	# return re-packed bits

	return np.packbits(A)



def unpackbits(A, bits=8, shape=(-1,)):

	# takes a flat array-like of uint8 and number of bits to decode per entry

	A = np.array(A, dtype=np.uint8)
	if bits == 8: return A.reshape(shape)

	# get shape and axis for array manipulation

	axis = len(shape)
	shape = (*shape, bits)

	# unback bit arrays

	A = np.unpackbits(A)

	# remove extra bits creted when unpacking and reshape

	remainder = np.remainder(reduce(mul, shape), bits)
	if remainder: A = A[:-remainder]

	# reshape bit array into entries

	A = A.reshape(shape)

	# pad with extra zeros

	B = np.zeros((*A.shape[:-1], 8), dtype=np.uint8)
	B[..., -bits:] = A

	# re-pack bits

	A = np.packbits(B, axis=axis)

	# squeeze added index and return

	return np.squeeze(A, axis=axis)



if __name__ == '__main__':

	x = packbits([[12, 1, 56, 34, 0, 63, 13, 5, 0, 2, 45, 9]], bits=8)
	print(x.nbytes)
	print(unpackbits(x, shape=(4, 3), bits=8))
