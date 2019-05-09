
# zip.py



import math
import numpy as np

from functools import partial, reduce
from operator import mul
from types import MethodType



class ByteArray:


	def __init__(self, obj, *args, **kwargs):

		if isinstance(obj, ByteArray): obj = obj._array
		self._array = np.array(obj, *args, dtype=np.uint8, **kwargs)


	def unpackbits(self, *args, **kwargs):

		return ByteArray(np.unpackbits(self._array, *args, **kwargs))


	def packbits(self, *args, **kwargs):

		return ByteArray(np.packbits(self._array, *args, **kwargs))


	@property
	def shape(self):

		return self._array.shape


	def expand_dims(self, *args, **kwargs):

		return ByteArray(np.expand_dims(self._array, *args, **kwargs))


	def flatten(self, *args, **kwargs):

		return ByteArray(self._array.flatten(*args, **kwargs))


	def reshape(self, *args, **kwargs):

		return ByteArray(self._array.reshape(*args, **kwargs))


	def squeeze(self, *args, **kwargs):

		return ByteArray(self._array.squeeze(*args, **kwargs))


	@property
	def nbytes(self, *args, **kwargs):

		return self._array.nbytes


	def __getitem__(self, key):

		return ByteArray(self._array[key])


	def __setitem__(self, key, value):

		if isinstance(value, ByteArray): value = value._array
		self._array[key] = value


	def __str__(self):

		return str(self._array)



def packbits(A, bits=8):

	# takes an array-like of uint8 and number of bits to encode with

	A = ByteArray(A)
	if bits == 8: return A.flatten()

	# get shape and axis for array manipulation

	shape = A.shape
	axis = len(shape)

	# add dimension to store unpacked bits, convert to bitarray, take only last bits, flatten, repack bits and return

	return A.expand_dims(axis=axis).unpackbits(axis=axis)[..., -bits:].flatten().packbits()._array



def unpackbits(A, bits=8, shape=(-1,)):

	# takes a flat array-like of uint8 and number of bits to decode per entry

	A = ByteArray(A)
	if bits == 8: return A.reshape(shape)

	# get shape and axis for array manipulation

	axis = len(shape)
	shape = (*shape, bits)

	# unback bit arrays

	A = A.unpackbits()

	# remove extra bits creted when unpacking and reshape

	remainder = np.remainder(reduce(mul, shape), bits)
	if remainder: A = A[:-remainder]

	# reshape bit array into entries

	A = A.reshape(shape)

	# pad with extra zeros

	B = ByteArray(np.zeros((*A.shape[:-1], 8)))
	B[..., -bits:] = A

	# repack bits, squeeze added index, and return

	return B.packbits(axis=axis).squeeze(axis=axis)



if __name__ == '__main__':

	x = np.array([[[True, False], [False, True]], [[True, True], [False, False]]])
	print(x)
	print(x.nbytes)

	x = packbits(x, bits=1)
	print(x)
	print(x.nbytes)

	x = unpackbits(x, shape=(-1, 2, 2), bits=1)
	print(x)
	print(x.nbytes)

