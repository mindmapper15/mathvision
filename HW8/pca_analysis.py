import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind':lambda x:"{0:0.6f}".format(x)})

data_a = np.loadtxt('data_a.txt', delimiter=',').T
data_b = np.loadtxt('data_b.txt', delimiter=',').T

data_ab = np.append(data_a, data_b, axis=1)

cov_ab = np.cov(data_ab)
eigenvalue, eigenvector = np.linalg.eig(cov_ab)
print(f"Eigen-value : {eigenvalue}")
print(f"Eigen-vector : {eigenvector}")

subspace_s = np.array([eigenvector[0], eigenvector[3]])
print(f"Projection matrix to subspace S")
print(f'{subspace_s}')
data_a_proj = np.matmul(subspace_s, data_a)
data_b_proj = np.matmul(subspace_s, data_b)

plt.figure()
plt.scatter(data_a_proj[0,:], data_a_proj[1,:])
plt.scatter(data_b_proj[0,:], data_b_proj[1,:])
plt.legend(["Apple A", "Apple B"])
plt.show()
plt.close()

data_a_mean = np.mean(data_a_proj, axis=1)
data_b_mean = np.mean(data_b_proj, axis=1)

data_a_cov = np.cov(data_a_proj)
data_b_cov = np.cov(data_b_proj)

gauss_a_x, gauss_a_y = np.random.multivariate_normal(data_a_mean, data_a_cov, 10000).T
hist_a, a_xedges, a_yedges = np.histogram2d(gauss_a_x, gauss_a_y, bins=20)

gauss_b_x, gauss_b_y = np.random.multivariate_normal(data_b_mean, data_b_cov, 10000).T
hist_b, b_xedges, b_yedges = np.histogram2d(gauss_b_x, gauss_b_y, bins=20)

plt.figure()
plt.subplot(1,2,1)
plt.title("2D Gaussian Distribution of Projection of Apple A")
plt.contourf(a_xedges[:-1], a_yedges[:-1], hist_a)
plt.subplot(1,2,2)
plt.title("2D Gaussian Distribution of Projection of Apple B")
plt.contourf(b_xedges[:-1], b_yedges[:-1], hist_b)
plt.colorbar()
plt.show()

test_data_1 = np.array([0.6322,-3.2764,1.6776,-0.2263])
test_data_2 = np.array([-4.1113,10.9427,-1.1570,-1.7366])

test_data_1_proj = np.matmul(subspace_s, test_data_1)
test_data_2_proj = np.matmul(subspace_s, test_data_2)

m_distance_1a = np.sqrt(np.matmul(test_data_1_proj - data_a_mean, np.matmul(np.linalg.inv(data_a_cov), test_data_1_proj - data_a_mean).T))
m_distance_1b = np.sqrt(np.matmul(test_data_1_proj - data_b_mean, np.matmul(np.linalg.inv(data_b_cov), test_data_1_proj - data_b_mean).T))
m_distance_2a = np.sqrt(np.matmul(test_data_2_proj - data_a_mean, np.matmul(np.linalg.inv(data_a_cov), test_data_2_proj - data_a_mean).T))
m_distance_2b = np.sqrt(np.matmul(test_data_2_proj - data_b_mean, np.matmul(np.linalg.inv(data_b_cov), test_data_2_proj - data_b_mean).T))

print("Mahalanobis Distance")
print(f"Test Data #1 - Apple A : {m_distance_1a}")
print(f"Test Data #1 - Apple B : {m_distance_1b}")
print(f"Test Data #2 - Apple A : {m_distance_2a}")
print(f"Test Data #2 - Apple B : {m_distance_2b}")
