import PIL.Image as pilimg
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

face_path = Path('./att_faces')
face_image_list = list(face_path.rglob('*.png'))
face_image_list = sorted(face_image_list)
#print(face_image_list)

face_dataset = []
face_testset = []
face_dataset_fname = []
face_testset_fname = []
for i, fname in enumerate(face_image_list):
    image = pilimg.open(str(fname))
    if i % 10 == 0:
        face_testset.append(np.array(image))
        face_testset_fname.append(fname)
    else:
        face_dataset.append(np.array(image))
        face_dataset_fname.append(fname)


face_dataset = np.array(face_dataset)
face_width = face_dataset.shape[2]
face_height = face_dataset.shape[1]
num_pixel = face_dataset.shape[1] * face_dataset.shape[2]
face_dataset = face_dataset.reshape(-1, num_pixel)

face_testset = np.array(face_testset)
face_testset = face_testset.reshape(-1, num_pixel)

face_dataset_cov_np = np.cov(face_dataset.T)
eig_value, eig_vec = np.linalg.eig(face_dataset_cov_np)
eig_vec = eig_vec.real.T

pca_index = np.flip(np.argsort(eig_value))
eig_vec_top_10_index = pca_index[:10]

for i, idx in enumerate(eig_vec_top_10_index):
    eigen_vector = eig_vec[idx]
    eigen_face = eigen_vector.reshape(face_height, face_width)
    plt.imsave(f'eigen_face_{i+1}.png', eigen_face, cmap='gray')

test_face = face_testset[9]
coeff = np.matmul(eig_vec, test_face)
plt.imsave(f'test_face_original.png', test_face.reshape(face_height, face_width), cmap='gray')

for i, k in enumerate([1, 10, 100, 200, 500, 1000, num_pixel]):
    top_k_index = pca_index[:k]
    test_face_rec = np.zeros(test_face.shape)
    for idx in top_k_index:
        test_face_rec += coeff[idx] * eig_vec[idx]
    
    plt.imsave(f'test_face_with_k_{k}.png', test_face_rec.reshape(face_height, face_width), cmap='gray')

subspace_s = np.array([eig_vec[idx] for idx in pca_index[:40]])
face_dataset_proj = np.matmul(subspace_s, face_dataset.T)
face_testset_proj = np.matmul(subspace_s, face_testset.T)

correct_count = 0
for i in range(40):
    input_face = face_testset[i].reshape(face_height, face_width)
    input_face_proj = face_testset_proj[:,i]
    input_face_proj = np.repeat(np.expand_dims(input_face_proj, axis=-1), face_dataset_proj.shape[1], axis=1)
    distance_mat = np.sqrt(np.sum(np.power(np.abs(input_face_proj - face_dataset_proj), 2), axis=0))
    closest_face = face_dataset[np.argmin(distance_mat)].reshape(face_height, face_width)

    closest_face_id = face_dataset_fname[np.argmin(distance_mat)].stem.split('_')[0]
    test_face_id = face_testset_fname[i].stem.split('_')[0]

    if closest_face_id == test_face_id:
        correct_count += 1
    else:
        print(f"Test Face ID : {test_face_id}, Closest Face ID : {closest_face_id}")
        plt.figure()
        plt.suptitle(f"Face ID : {closest_face_id}\nTest Face Filename : {test_face_id}")
        plt.subplot(1,2,1)
        plt.imshow(input_face, interpolation='lanczos', aspect='auto', cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(closest_face, interpolation='lanczos', aspect='auto', cmap='gray')
        plt.show()

acc = correct_count / 40 * 100
print(f"Classification Accuracy is {acc:.4f} %")

my_face = pilimg.open('my_face.png').convert('L')
my_face = np.array(my_face, 'uint8')
my_face_vector = my_face.reshape(my_face.shape[0]*my_face.shape[1])
my_face_proj = np.matmul(subspace_s, my_face_vector)
my_face_proj = np.repeat(np.expand_dims(my_face_proj, axis=-1), face_dataset_proj.shape[1], axis=1)
distance_mat = np.sqrt(np.sum(np.power(np.abs(my_face_proj - face_dataset_proj), 2), axis=0))
closest_face = face_dataset[np.argmin(distance_mat)].reshape(face_height, face_width)

closest_face_id = face_dataset_fname[np.argmin(distance_mat)].stem.split('_')[0]
plt.figure()
plt.suptitle(f"Closest Face ID : {closest_face_id}")
plt.subplot(1,2,1)
plt.imshow(my_face, interpolation='lanczos', aspect='auto', cmap='gray')
plt.subplot(1,2,2)
plt.imshow(closest_face, interpolation='lanczos', aspect='auto', cmap='gray')
plt.show()

