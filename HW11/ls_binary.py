import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_orig = Image.open("sample.png")
img = np.asarray(img_orig)
row, col = img.shape

mat_a = np.zeros((row*col, 6))
mat_z = img.flatten()
mat_a[:,5] = 1

for i in range(row):
    for j in range(col):
        mat_a[col*i+j,3] = j
        mat_a[col*i+j,4] = i
        mat_a[col*i+j,0] = j*j
        mat_a[col*i+j,1] = i*i
        mat_a[col*i+j,2] = i*j

mat_x = np.linalg.inv(mat_a.T @ mat_a) @ mat_a.T @ mat_z
a, b, c, d, e, f = mat_x

mat_i = np.zeros((row,col))
for y in range(row):
    for x in range(col):
        mat_i[y,x] = a * x**2 + b * y**2 + c*x*y + d*x + e*y + f

mat_i = np.clip(mat_i, a_min=0, a_max=255).astype(np.uint8)
img_i = Image.fromarray(mat_i)

img_i.save("intensity_change.png", "PNG", bits=8)

diff = img.astype(int) - mat_i.astype(int)
diff = (diff - diff.min())/(diff.max() - diff.min())
diff *= 255
diff = np.clip(diff, 0, 255).astype(np.uint8)

img_diff = Image.fromarray(diff)
img_diff.save("Image Difference.png", "PNG", bits=8)

thr_list = [50,100,150,200]
for thr in thr_list:
    diff_copy = np.copy(diff)
    diff_copy[diff_copy<thr] = 0
    diff_copy[diff_copy>=thr] = 255
    img_diff_binary = Image.fromarray(diff_copy)
    img_diff_binary.save(f"Image Difference Binary (threshold={thr}).png", "PNG", bits=8)


for thr in thr_list:
    img_copy = np.copy(img)
    img_copy[img_copy<thr] = 0
    img_copy[img_copy>=thr] = 255
    img_diff_binary = Image.fromarray(img_copy)
    img_diff_binary.save(f"Original Image Binary (threshold={thr}).png", "PNG", bits=8)
