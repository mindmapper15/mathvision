import numpy as np
import matplotlib.pyplot as plt

size = 10
x_random = np.random.randint(500, size=size)
y_random = np.random.randint(500, size=size)

mat_a = np.zeros((size,2))
mat_a[:,0] = x_random
mat_a[:,1] = 1
w = np.zeros((size, size))

plt.figure()
plt.title("Random dots and its line based on Least Square")
for i in range(size):
    x, y = x_random[i], y_random[i]
    plt.scatter(x, y, c='blue', marker='o')
mat_x = np.linalg.inv(mat_a.T @ mat_a) @ mat_a.T @ y_random
for n_iter in range(10):
    a, b = mat_x
    print(f'Robust LR iter #{n_iter} : {a}x + {b}')
    line_x = np.arange(500)
    line_y = a*line_x + b
    plt.plot(line_x, line_y, label=f'iter {n_iter}')

    y_updated = a*x_random + b
    r = y_random - mat_a @ mat_x
    for j in range(len(r)):
        w[j,j] = 1 / (np.abs(r[j])/1.3998 + 1)
    mat_x = np.linalg.inv(mat_a.T @ w @ mat_a) @ (mat_a.T @ w @ y_random)
plt.legend()
plt.show()

