import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import shutil

def sine_function(data, a, b, c, d):
    return a * np.sin(b * data + c) + d

def make_jacobian_mat(data, a, b, c, d):
    # Make Jacobian Matrix of residual functions
    j_mat = np.zeros((len(data), 4))
    j_mat[:,0] = -np.sin(b * data + c)
    j_mat[:,1] = -a * data * np.cos(b * data + c)
    j_mat[:,2] = -a * np.cos(b * data + c)
    j_mat[:,3] = -1
    return j_mat

# Generate samples from sine function
x = np.linspace(0, 10, 500)
y = np.sin(0.5 * np.pi * x)

x_sample = np.random.choice(x, 120)
y_sample = np.sin(0.5 * np.pi * x_sample)

# Add noise to sample for modeling
noise = np.random.randn(120) * 0.2
y_sample_noise = y_sample + noise
print("noise : ", noise)

plt.figure(figsize=(12,4))
plt.plot(x, y, color='red', label='Base Function')
plt.scatter(x_sample, y_sample_noise, label='Noised Data')
plt.legend()
plt.title(f"Comparison of base sine function and noised data, sin{(0.5*np.pi):.6f}x")
plt.tight_layout()
plt.savefig("original_sine_function.png")
plt.close()

if os.path.isdir('./optimization_iter'):
    shutil.rmtree('./optimization_iter')
os.makedirs('./optimization_iter', exist_ok=True)

if os.path.isfile('optimization.gif'):
    os.remove('optimization.gif')

# Initializing coefficients of sine function model
b, c = 0.9, -3
d = np.mean(y_sample_noise)
a = np.std(y_sample_noise)

max_iter = 500
y_observed = sine_function(x_sample, a, b, c, d)
p = np.array([a, b, c, d])
sigma = 0.01
threshold = 1e-3
for i in range(max_iter):
    # Plotting Sample and approximated function
    y_approx = sine_function(x, a, b, c, d)
    plt.figure(figsize=(12,4))
    plt.title(f'Iter. #{i}, a={a:.4f},b={b:.4f},c={c:.4f},d={d:.4f},')
    plt.plot(x, y_approx, color='red', label='Approx. Function')
    plt.scatter(x_sample, y_sample_noise, label='')
    plt.savefig(f'./optimization_iter/optimize_iter_#{i:03d}.png')
    plt.close()

    # Updating parameters of sine function
    residual = y_sample_noise - y_observed
    jacobian = make_jacobian_mat(x_sample, a, b, c, d)
    p_new = p - sigma * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ residual
    # Calculate values with updated sine function
    a, b, c, d = p_new
    y_observed = sine_function(x_sample, a, b, c, d)

    # If the parameter has converged, stop updating
    p_delta = np.linalg.norm(np.abs(p_new - p))
    c_str = f' + {c:.6f}' if c >= 0 else f'{ - np.abs(c):.6f}'
    d_str = f' + {d:.6f}' if d >= 0 else f'{ - np.abs(d):.6f}'
    print(f"iter #{i:02d} : {a:.6f} * sin({b:.6f}x{c_str}){d_str}, p_delta = {p_delta:.6f}")
    if p_delta < threshold:
        break
    else:
        p = p_new

img_list = sorted(glob.glob("./optimization_iter/*.png"))
with imageio.get_writer('optimization.gif', mode='I') as writer:
    for img_fname in img_list:
        image = imageio.imread(img_fname)
        writer.append_data(image)
