
# zip.py



import math
import numpy as np

from functools import reduce
from operator import mul



def zip_uint8(A, bits=8):

	# takes an array-like of integers and number of bits to encode with

	A = np.array(A, dtype=np.uint8)
	if bits == 8: return A

	# get shape and axis for array manipulation

	shape = A.shape
	axis = len(shape)

	# add dimension to store unpacked bits

	A = np.expand_dims(A, axis=axis)

	# convert to bitarray, take only last bits, and flatten

	A = np.unpackbits(A, axis=axis)[..., -bits:].flatten()

	# return re-packed bits

	return np.packbits(A)



def unzip_uint8(A, shape=(-1,), bits=8):

	# calculate shape for array manipulation

	shape = (*shape, bits)
	axis = len(shape) - 1

	# unback bit arrays

	A = np.array(A, dtype=np.uint8)
	A = np.unpackbits(A)

	# remove extra bits creted when unpacking and reshape

	remainder = np.remainder(reduce(mul, shape), bits)
	if remainder: A = A[:-remainder]

	# reshape bit array into entries

	A = A.reshape(shape)

	# pad with extra zeros if necessary

	if bits == 8: B = A
	else: 

		B = np.zeros((*A.shape[:-1], 8), dtype=np.uint8)
		B[..., -bits:] = A

	# re-pack bits

	A = np.packbits(B, axis=axis)

	# squeeze added index and return

	return np.squeeze(A, axis=axis)



if __name__ == '__main__':

	x = zip_uint8([[12, 1, 56, 34, 0, 63, 13, 5, 0, 2, 45, 9]], bits=6)
	print(unzip_uint8(x, shape=(2, 3, 2), bits=6))
