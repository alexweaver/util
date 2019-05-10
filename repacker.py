
# zip.py



import math
import numpy as np

from functools import partial, reduce
from operator import mul
from types import MethodType
from numpy import packbits, uint8


class Array:


	# wrapper class for numpy arrays to make function calls
	# more consistent, also enable method chaining
	# ie. arr.func(...) vs np.func(arr, ...)


	def __init__(self, obj, *args, **kwargs):

		if isinstance(obj, self.__class__): obj = obj._array
		self._array = np.array(obj, *args, **kwargs)


	# array properties


	@property
	def shape(self):

		return self._array.shape

		
	@property
	def nbytes(self, *args, **kwargs):

		return self._array.nbytes


	# methods of the form np.func(arr, ...)


	def unpackbits(self, *args, **kwargs):

		return self.__class__(np.unpackbits(self._array, *args, **kwargs))


	def packbits(self, *args, **kwargs):

		return self.__class__(packbits(self._array, *args, **kwargs))


	def expand_dims(self, *args, **kwargs):

		return self.__class__(np.expand_dims(self._array, *args, **kwargs))

	def pad(self, *args, **kwargs):

		return self.__class__(np.pad(self._array, *args, **kwargs))


	# methods of the form arr.func(...)


	def flatten(self, *args, **kwargs):

		return self.__class__(self._array.flatten(*args, **kwargs))


	def reshape(self, *args, **kwargs):

		return self.__class__(self._array.reshape(*args, **kwargs))


	def squeeze(self, *args, **kwargs):

		return self.__class__(self._array.squeeze(*args, **kwargs))


	def tobytes(self, *args, **kwargs):

		return self._array.tobytes()


	def view(self, dtype, *args, **kwargs):

		return self.__class__(self._array.view(dtype, *args, **kwargs), dtype=dtype)


	def byteswap(self, *args, **kwargs):

		return self.__class__(self._array.byteswap())


	# magic methods


	def __getitem__(self, key):

		return self.__class__(self._array[key])


	def __setitem__(self, key, value):

		if isinstance(value, self.__class__): value = value._array
		self._array[key] = value


	def __str__(self):

		return str(self._array)


	def __add__(self, other):

		if isinstance(other, self.__class__): other = other._array
		return self.__class__(self._array + other)



def repackbits(A, dtype, bits=None):

	bits = bits if bits else 8 * np.dtype(dtype).itemsize

	# takes an array-like, A

	A = Array(A, dtype=dtype)
	if bits == 8 * np.dtype(dtype).itemsize: return A.flatten()

	# add dimension to store unpacked bits
	# swap byte order
	# convert to bitarray
	# take only last bits
	# flatten
	# repack bits
	# return bytes

	return A.expand_dims(axis=-1) \
			.byteswap() \
			.view(uint8) \
			.unpackbits(axis=-1)[..., -bits:] \
			.flatten() \
			.packbits() \
			.tobytes()



def unpackbits(A, dtype, bits=None, shape=(-1,)):

	itemsize = np.dtype(dtype).itemsize
	bits = bits if bits else 8 * itemsize
	shape = (*shape, bits)
	remainder = np.remainder(reduce(mul, shape), bits)

	# takes a flat array-like of uint8 and number of bits to decode per entry

	A = Array(np.frombuffer(A, dtype=uint8), dtype=uint8)
	if bits == 8 * itemsize: return A.reshape(shape)

	# unpack bit arrays and remove extra bits created when unpacking
	# reshape bit array into entries
	# pad last axis with extra 0's
	# pack bits
	# convert to original data type
	# swap bytes
	# squeeze off extra index
	# return

	return A.unpackbits()[:-remainder or None] \
		.reshape(shape) \
		.pad([(0, 0)] * (len(shape) - 1) + [(8 * itemsize - bits, 0)], 'constant', constant_values=0) \
		.packbits(axis=-1) \
		.view(dtype=dtype) \
		.byteswap() \
		.squeeze(axis=-1)



if __name__ == '__main__':

	x = np.array([[[1, 16], [30, 3]], [[134, 2], [300, 40]]], dtype=np.uint16)

	print(x)
	print(x.nbytes)

	x = repackbits(x, dtype=np.uint16, bits=11)
	print(x)
	print(len(x))

	x = unpackbits(x, dtype=np.uint16, shape=(-1, 2, 2), bits=11)
	print(x)
	print(x.nbytes)

