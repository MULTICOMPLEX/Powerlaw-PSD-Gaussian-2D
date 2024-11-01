from matplotlib import pylab as plt
import numpy as np
from numpy.fft import irfft, rfftfreq, irfft2
from matplotlib.ticker import ScalarFormatter
import phimagic_prng32
import phimagic_prng64
import time


steps = 2**12 # number of steps to generate
return_to_beginning = 0
beta = 1.9 # the exponent
fmin = 0.0

#Number of frequencies for approximation
Nfreq = 30


def powerlaw_psd_gaussian_2D(beta, size, rng, fmin=0):

    # Validate fmin
    if not 0 <= fmin <= 0.5:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Dimensions (assuming square for simplicity)
    N = size
    # Generate 2D frequencies
    fx = np.fft.fftfreq(N)  # Note: fftfreq, not rfftfreq
    fy = np.fft.fftfreq(N)
    fx, fy = np.meshgrid(fx, fy) # 2D frequency grid
    f = np.sqrt(fx**2 + fy**2)   # Radial frequencies


    # Ensure fmin is not too small
    fmin = max(fmin, 1./N)  # Lower bound on fmin

    # Scaling factors
    s_scale = f.copy() #copy so we don't modify the original f
    s_scale[f < fmin] = fmin  # Avoid division by zero
    s_scale = s_scale**(-beta/2.)
    
    v = np.sqrt(np.pi)
    # Generate random Gaussian field in Fourier space
    real_part = rng.uniform(-v, v, size=(N, N))
    imag_part = rng.uniform(-v, v, size=(N, N))

    # Ensure conjugate symmetry for real signal
    sr = real_part * s_scale
    si = imag_part * s_scale
    #Enforce conjugate symmetry

    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / steps
    
    # Combine to create Fourier components
    s = sr + 1j * si

    # Inverse FFT
    y = np.fft.ifft2(s).real / sigma # Take the real part 

    return y


def func_approx_2D(x, n):
    """
    Performs 2D Fourier approximation.

    Args:
        x: 2D NumPy array (e.g., an image).
        n: Number of frequency components to keep in each dimension.

    Returns:
        The 2D approximated array (real part).
    """
    # 2D Fourier Transform
    yf = np.fft.fft2(x)

    # Truncate higher frequencies
    num_components = int(n)
    h, w = yf.shape
    yf_truncated = yf.copy() #Important to copy so you don't modify original yf
    yf_truncated[num_components:h-num_components,:] = 0 #rows, all cols
    yf_truncated[:,num_components:w-num_components] = 0 #all rows, cols



    # 2D Inverse Fourier Transform
    y_approx = np.fft.ifft2(yf_truncated)
    return y_approx.real

#Time seed 
current_time_seconds = int(time.time())
rng = np.random.default_rng(current_time_seconds)       #numpy PRNG


# Example usage:
noise = powerlaw_psd_gaussian_2D(beta, steps, rng, fmin)
noise = func_approx_2D(noise, Nfreq)


def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(which = 'major', colors='white')
    ax.tick_params(which = 'minor', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 

fig = plt.figure(figsize=(8, 8), facecolor='#002b36')  
ax = fig.gca()
set_axis_color(ax)
plt.title("Power Spectral Density", color = 'white')
plt.imshow(noise, cmap='gray')


fig = plt.figure(figsize=(8, 8), facecolor='#002b36')  
ax = fig.gca()
set_axis_color(ax)

# Create X and Y coordinates
x = np.arange(noise.shape[1])  # x-coordinates (columns)
y = np.arange(noise.shape[0])  # y-coordinates (rows)
X, Y = np.meshgrid(x, y)

# Create the contour plot
contour = ax.contour(X, Y, noise, levels=32, cmap='Pastel1') #Or contourf

#plt.grid()

plt.show()











