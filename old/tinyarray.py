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