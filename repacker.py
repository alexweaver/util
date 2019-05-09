
# zip.py



import math
import numpy as np

from functools import partial, reduce
from operator import mul
from types import MethodType



class Array:


	def __init__(self, obj, *args, **kwargs):

		if isinstance(obj, self.__class__): obj = obj._array
		self._array = np.array(obj, *args, **kwargs)


	@property
	def shape(self):

		return self._array.shape

		
	@property
	def nbytes(self, *args, **kwargs):

		return self._array.nbytes


	# methods which have to be called from the numpy module


	def unpackbits(self, *args, **kwargs):

		return self.__class__(np.unpackbits(self._array, *args, **kwargs))


	def packbits(self, *args, **kwargs):

		return self.__class__(np.packbits(self._array, *args, **kwargs))


	def expand_dims(self, *args, **kwargs):

		return self.__class__(np.expand_dims(self._array, *args, **kwargs))


	# methods which allow the method to be called by the array object


	def flatten(self, *args, **kwargs):

		return self.__class__(self._array.flatten(*args, **kwargs))


	def reshape(self, *args, **kwargs):

		return self.__class__(self._array.reshape(*args, **kwargs))


	def squeeze(self, *args, **kwargs):

		return self.__class__(self._array.squeeze(*args, **kwargs))


	def __getitem__(self, key):

		return self.__class__(self._array[key])


	def __setitem__(self, key, value):

		if isinstance(value, self.__class__): value = value._array
		self._array[key] = value


	def __str__(self):

		return str(self._array)



def packbits(A, dtype=np.uint8, bits=None):

	bits = bits if bits else 8 * np.dtype(dtype).itemsize

	# takes an array-like of uint8 and number of bits to encode with

	A = Array(A, dtype=dtype)
	if bits == 8 * np.dtype(dtype).itemsize: return A.flatten()

	# get shape and axis for array manipulation

	shape = A.shape
	axis = len(shape)

	# add dimension to store unpacked bits, convert to bitarray, take only last bits, flatten, repack bits and return

	return A.expand_dims(axis=axis).unpackbits(axis=axis)[..., -bits:].flatten().packbits()._array



def unpackbits(A, dtype=np.uint8, bits=None, shape=(-1,)):

	bits = bits if bits else 8 * np.dtype(dtype).itemsize

	# takes a flat array-like of uint8 and number of bits to decode per entry

	A = Array(A, dtype=dtype)
	if bits == 8 * np.dtype(dtype).itemsize: return A.reshape(shape)

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

	B = Array(np.zeros((*A.shape[:-1], 8)), dtype=np.uint8)
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

