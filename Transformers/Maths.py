from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


def jacobian(s, C = 1):
    """ Evaluates softmax Jacobian for vector `s`. Adds scaling factor `C` if 
    provided, else defaults to scaling of 1.
    """
    softmax_s = softmax(s / C)
    return 1./C * ( np.diag(softmax_s) - np.outer(softmax_s, softmax_s) )


def plot_jacobian(matrix, size, ax, norm=None):
    im = ax.imshow(matrix, norm=norm, cmap="RdBu")

    ax.set_title(
        "Length: {} \n Mean: {:.3} \n SD: {:.3} \n Range: {:.3}"
        .format(size, np.mean(matrix), matrix.std(), matrix.ptp())
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)


def generate_jacobian_comparison(sizes):
    fig_scaled, axes_scaled = plt.subplots(1, len(sizes), figsize=(23, 4))
    fig, axes = plt.subplots(1, len(sizes), figsize=(23, 4))
        
    fig.suptitle("Unscaled Softmax Jacobian Evaluated at Random Vector", y=1.1)
    fig_scaled.suptitle("Scaled Softmax Jacobian Evaluated at Random Vector", y=1.1)

    for n, size in enumerate(sizes):    
        scale = size ** 0.5     

        # random normal vector with mean 0 and variance `size`.
        # this represents the vector "s" in our math
        s = np.random.randn(size) * scale
                    
        jac_s = jacobian(s, C=1)
        jac_s_scaled = jacobian(s, C=scale)
        
        # normalize to extrema of scaled Jacobian to better visualize
        # unscaled, which is often dominated by a handful of large values
        norm = Normalize(jac_s_scaled.min(), jac_s_scaled.max())

        plot_jacobian(jac_s_scaled, size, axes_scaled[n], norm=norm)
        plot_jacobian(jac_s, size, axes[n], norm=norm)

    plt.show()

sizes = [4, 8, 16, 32, 64] # some sample lengths for our inputs
generate_jacobian_comparison(sizes)