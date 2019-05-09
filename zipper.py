
# zip.py



import math
import numpy as np



def zip_uint8(A, bits=8):

	# takes an array-like of integers and number of bits to encode with

	A = np.array(A, dtype=np.uint8)
	if bits == 8: return A

	# convert to bitarray, reshape to arrays of length 8 and take only last bits, flatten

	A = np.unpackbits(A).reshape(-1, 8)[:, -bits:].flatten()

	return np.packbits(A)



def unzip_uint8(A, bits=8):

	# unback bit arrays

	A = np.array(A, dtype=np.uint8)
	A = np.unpackbits(A)

	# remove extra bits creted when unpacking and reshape

	if bits < 8: A = A[:-np.remainder(len(A), bits)]

	# reshape into entries of A

	A = A.reshape(-1, bits)

	# pad with extra zeros if necessary

	if bits == 8: B = A
	else: 

		B = np.zeros((len(A), 8), dtype=np.uint8)
		B[:, -bits:] = A

	return np.packbits(B)
