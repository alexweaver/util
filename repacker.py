
# zip.py



import math
import numpy as np
import gzip

from functools import partial, reduce
from operator import mul
from types import MethodType
from numpy import packbits, uint8, int8, int16, int32
from math import pow



SIGNED_TYPES = set([int8, int16, int32])



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


	def __sub__(self, other):

		if isinstance(other, self.__class__): other = other._array
		return self.__class__(self._array - other)



def repackbits(A, dtype, bits=None):

	itemsize = np.dtype(dtype).itemsize
	bits = bits if bits else 8 * itemsize

	# handle signed data types

	if dtype in SIGNED_TYPES: A = A + 2 ** (bits - 1)

	# add dimension to store unpacked bits
	# swap byte order
	# convert to bitarray
	# take only last bits
	# flatten
	# repack bits
	# return bytes

	return Array(A, dtype='u{0}'.format(itemsize)) \
			.expand_dims(axis=-1) \
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

	# unpack bit arrays and remove extra bits created when unpacking
	# reshape bit array into entries
	# pad last axis with extra 0's
	# pack bits
	# convert to original data type
	# swap bytes
	# squeeze off extra index

	A = Array(np.frombuffer(A, dtype=uint8), dtype=uint8) \
		.unpackbits()[:-(np.remainder(reduce(mul, shape), bits)) or None] \
		.reshape(shape) \
		.pad([(0, 0)] * (len(shape) - 1) + [(8 * itemsize - bits, 0)], 'constant', constant_values=0) \
		.packbits(axis=-1) \
		.view(dtype='u{0}'.format(itemsize)) \
		.byteswap() \
		.squeeze(axis=-1)

	# calculate offset for signed data types

	if dtype in SIGNED_TYPES: A = A - 2 ** (bits - 1)

	# return

	return Array(A, dtype=dtype)



if __name__ == '__main__':

	a = np.array(np.random.randint(-1, 2, size=(1000000, 5)), dtype=np.int8)

	print(a)
	print(a.nbytes)

	x = repackbits(a, dtype=np.int8, bits=2)
	print(len(x))

	with gzip.open('a.gzip', 'wb') as f:

		f.write(a)	

	with gzip.open('x.gzip', 'wb') as f:

		f.write(x[0:(len(x) - 1)])

	with open('f', 'wb') as f:

		f.write(a)

	x = unpackbits(x, dtype=np.int8, shape=(-1, 5), bits=2)
	print(x)
	print(x.nbytes)

