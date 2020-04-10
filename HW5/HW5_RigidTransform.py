import numpy as np

poly_a = [
    [-0.5, 0.0, 2.121320],
    [0.5, 0.0, 2.121320],
    [0.5, -0.707107, 2.828427],
    [0.5, 0.707107, 2.828427],
    [1, 1, 1]
    ]

poly_a_prime = [
    [1.363005, -0.427130, 2.339082],
    [1.748084, 0.437983, 2.017688],
    [2.636461, 0.184843, 2.400710],
    [1.4981, 0.8710, 2.8837],
    [0, 0, 0]                       # We don't know the coordinate of p5 yet.
    ]

poly_a = np.array(poly_a, dtype=np.float64)
poly_a_prime = np.array(poly_a_prime, dtype=np.float64)

poly_a_original = np.copy(poly_a)
p1_original = poly_a_original[0]

# Moving the coordinates of A that makes p1 to O(0,0)
for i in range(poly_a.shape[0]):
    for j in range(poly_a.shape[1]):
        poly_a[i][j] -= p1_original[j]

# p1 moved to O(0,0), so now vector p1p2 equals to p2 and p1p3 to p3
p2 = poly_a[1]
p3 = poly_a[2]

vector_h = np.cross(p2, p3)

p2_prime = poly_a_prime[1] - poly_a_prime[0]
p3_prime = poly_a_prime[2] - poly_a_prime[0]

vector_h_prime = np.cross(p2_prime, p3_prime)

# Calculate Rotation axis vector between h and h`
# by dividing h x h` with its norm
rot_axis = np.cross(vector_h, vector_h_prime)
rot_axis = rot_axis / np.linalg.norm(rot_axis)


# Calculate Rotation angle(radian) between h and h'
# by dividing inner product of h and h` with their norm
cos_theta = np.inner(vector_h, vector_h_prime) / \
            (np.linalg.norm(vector_h) * np.linalg.norm(vector_h_prime))

sin_theta = np.sqrt(1 - cos_theta ** 2)

ux, uy, uz = rot_axis

r1_mat = np.array([
    [cos_theta + ux**2 * (1 - cos_theta), ux*uy*(1-cos_theta)-uz*sin_theta, ux*uz*(1-cos_theta)+uy*sin_theta],
    [uy*ux*(1-cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1-cos_theta), uy*uz*(1-cos_theta)-ux*sin_theta],
    [uz*ux*(1-cos_theta) - uy*sin_theta, uz*uy*(1-cos_theta) + ux*sin_theta, cos_theta+uz**2 * (1-cos_theta)]
    ])

vector_q_prime = poly_a_prime[2] - poly_a_prime[0]
vector_q = np.matmul(r1_mat, p3)

ux, uy, uz = vector_h_prime / np.linalg.norm(vector_h_prime)

cos_theta_2 = np.inner(vector_q, vector_q_prime) / \
              (np.linalg.norm(vector_q) * np.linalg.norm(vector_q_prime))
sin_theta_2 = np.sqrt(1 - cos_theta_2 ** 2)

r2_mat = np.array([
    [cos_theta_2 + ux**2 * (1 - cos_theta_2), ux*uy*(1-cos_theta_2)-uz*sin_theta_2, ux*uz*(1-cos_theta_2)+uy*sin_theta_2],
    [uy*ux*(1-cos_theta_2) + uz*sin_theta_2, cos_theta_2 + uy**2 * (1-cos_theta_2), uy*uz*(1-cos_theta_2)-ux*sin_theta_2],
    [uz*ux*(1-cos_theta_2) - uy*sin_theta_2, uz*uy*(1-cos_theta_2) + ux*sin_theta_2, cos_theta_2+uz**2 * (1-cos_theta_2)]
    ])

print(r1_mat)
print(r2_mat)

p4 = poly_a[3]
p1_original = poly_a_original[0]
p4_prime = poly_a_prime[3]
p4_prime_calculate = np.matmul(np.matmul(r1_mat, p4), r2_mat) + poly_a_prime[0]
print(p4_prime_calculate, p4_prime)
