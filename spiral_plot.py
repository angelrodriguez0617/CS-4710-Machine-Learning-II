import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu, sigma = 0, 0.2
num_samples_per_class = 1000

# Generate spiral data for Class 0
s0 = np.linspace(0, 2 * np.pi, num_samples_per_class)
r0 = np.linspace(1, 5, num_samples_per_class)  # Increase radius for spiral effect
C0_samples = np.array([r0 * np.cos(s0) + 4, r0 * np.sin(s0) + 5])
C0_samples += np.random.normal(mu, sigma, size=C0_samples.shape)  # Add Gaussian noise
C0_samples = np.transpose(C0_samples)
print(C0_samples.shape)

# Generate spiral data for Class 1 (distinct from Class 0)
s1 = np.linspace(0, 2 * np.pi, num_samples_per_class)
r1 = np.linspace(1.5, 5.5, num_samples_per_class)  # Slightly different radius
C1_samples = np.array([r1 * np.cos(s1) + 4, r1 * np.sin(s1) + 5])
C1_samples += np.random.normal(mu, sigma, size=C1_samples.shape)  # Add Gaussian noise
C1_samples = np.transpose(C1_samples)
print(C1_samples.shape)

# Plot the data
plt.scatter(C0_samples[:, 0], C0_samples[:, 1], label="Class 0", color="yellow", alpha=0.7)
plt.scatter(C1_samples[:, 0], C1_samples[:, 1], label="Class 1", color="purple", alpha=0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Noisy Spiral Data")
plt.legend()
plt.grid(True)
plt.show()
