'''
Nama	: Fatih Darielma Gaizta
NIM 	: 18220009
FILE	: dctPython18220009.py
	      Melakukan kompresi gambar JPG dengan DCT (Discrete Cosine Transform)
'''

import cv2
import numpy as np

from scipy import fft

# Konstanta TABEL_KUANTISASI berisi matrix of float
# Catatan: Nilai pada tabel kuantisasi telah ditetapkan secara konvensional
TABEL_KUANTISASI = np.array(np.matrix([
	[16, 11, 10, 16, 24, 40, 51, 61],
	[12, 12, 14, 19, 26, 58, 60, 55],
	[14, 15, 16, 24, 40, 57, 69, 56],
	[14, 17, 22, 29, 51, 87, 80, 62],
	[18, 22, 37, 56, 68, 109, 103, 77],
	[24, 35, 55, 64, 81, 104, 113, 92],
	[49, 64, 78, 87, 103, 121, 120, 101],
	[72, 92, 95, 98, 112, 100, 103, 99],
], dtype='float').tolist())


def encodeDCT(macroblock):
	global TABEL_KUANTISASI

	# DCT (Discrete Cosine Transform) hanya bekerja di rentang [-128,127]
	macroblock -= 128

	# Transform dengan DCT
	dct_data = fft.dct(macroblock)

	# Setiap pixel dibagi dengan nilai yang ada di tabel kuantisasi
	for i in range(8):
		for j in range(8):
			dct_data[i][j] = np.round(dct_data[i][j]/TABEL_KUANTISASI[i][j])

	return dct_data

def decodeDCT(dct_data):
	global TABEL_KUANTISASI

	# Setiap pixel dari gambar yang sudah di-encode dikalikan dengan nilai dari tabel kuantisasi
	for i in range(8):
		for j in range(8):
			dct_data[i][j] = np.round(dct_data[i][j]*TABEL_KUANTISASI[i][j])

	# Invers DCT
	dct_data = fft.idct(dct_data)
	for i in range(8):
		for j in range(8):
			dct_data[i][j] = np.round(dct_data[i][j])

	# Pixel kembali ditambahkan 128
	dct_data += 128

	return dct_data

if __name__ == '__main__':
	# Variabel image berisi matrix of pixels berukuran 1280x1280
	# Catatan: Ukuran foto image.jpg adalah 1280x1280
	image = np.array(cv2.imread('image.jpg', 0), dtype='float')

	# Macroblock merupakan pixel 8x8 yang diambil dari bagian mata
	macroblock = image[580:588, 640:648]
	print('--- SEBELUM KOMPRESI ---')
	print(macroblock, '\n')

	# Proses encoding
	dct_data = encodeDCT(macroblock)
	print('Nilai koefisien DC:', dct_data[0][0])
	print('Nilai koefisien AC:\n', dct_data[1:])

	# Proses decoding
	compressed_img = decodeDCT(dct_data)
	print('\n--- SETELAH KOMPRESI ---')
	print(compressed_img, '\n')