from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Load the dataset from the provided .npy file, center it around the origin, and return it as a numpy array of floats
    x = np.load(filename).astype(float)
    mean = np.mean(x, axis=0)
    centered_X = x - mean
    return centered_X

def get_covariance(dataset):
    # Calculate and return the covariance matrix of the dataset as a numpy matrix (d × d array)
    n = dataset.shape[0]
    S = np.dot(np.transpose(dataset),dataset) / (n - 1)
    return S

def get_eig(S, m):
    # Perform eigendecomposition on the covariance matrix S
    eigval, eigvec = eigh(S)
    idx = np.argsort(eigval)[::-1]  # Sort eigenvalues in descending order
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    Lambda = np.diag(eigval[:m])  # Top m eigenvalues
    U = eigvec[:, :m]  # Corresponding eigenvectors
    return Lambda, U

def get_eig_prop(S, prop):
    # Return eigenvalues and eigenvectors that explain more than a proportion of the variance
    eigval, eigvec = eigh(S)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]
    total_variance = np.sum(eigval)
    variance_explained = 0
    k = 0
    while variance_explained / total_variance < prop and k < len(eigval):
        variance_explained += eigval[k]
        k += 1
    # Lambda = np.diag(eigval[:k])
    Lambda = np.diag(np.diag(eigval[:k]))
    U = eigvec[:, :k]
    return Lambda, U

def project_image(image, U):
    # Project each m × 1 image into your k-dimensional subspace
    proj = np.dot(U.T, image)
    return np.dot(U, proj)

def display_image(orig, proj):
    # Display the original and projected images side-by-side
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    im1 = ax1.imshow(orig_image, aspect='equal', cmap='gray')
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(proj_image, aspect='equal', cmap='gray')
    ax2.set_title('Projection')
    plt.colorbar(im2, ax=ax2)
    plt.show()
    return fig, ax1, ax2

def perturb_image(image, U, sigma):
    # Perturb the given image using Gaussian distribution
    proj = np.dot(U.T, image)
    perturbation = np.random.normal(0, sigma, proj.shape)
    perturbed_proj = proj + perturbation
    return np.dot(U, perturbed_proj)

def combine_image(image1, image2, U, lam):
    # Combine two image features into one using convex combination
    proj1 = np.dot(U.T, image1)
    proj2 = np.dot(U.T, image2)
    combined_proj = lam * proj1 + (1 - lam) * proj2
    return np.dot(U, combined_proj)

