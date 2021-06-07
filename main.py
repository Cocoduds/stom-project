import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np

vals = STOM_higgs_tools.generate_data()

# bin_heights, bin_edges, patches = plt.hist(vals, range=[104, 155], bins=30)

bin_heights, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)

plt.xlabel("$m_{\gamma\gamma} (MeV)$")
plt.ylabel("Number of entries")
# plt.show()


# %% Background parametrisation
import scipy.integrate as integrate
import scipy
from scipy.optimize import curve_fit

background_data = [j for j in vals if j < 120]  # to avoid taking the signal bump, upper limit of 120 MeV set
# print(background_data)
N_background = len(background_data)
print('the number of data points in the background is', N_background)
sigma_background_data = sum(background_data)
print('the sum of the data points in the background is', sigma_background_data)
lamb = sigma_background_data / (N_background)  # maximum likelihood estimator for lambda
print('lambda estimate is', lamb)

background_bin_edges = [k for k in bin_edges if
                        103 < k < 120]  # takes the histogram values only within the background data region
background_bin_heights = bin_heights[0:9]  # As above, only takes the bin_heights within the background data region
area_hist = sum(np.diff(
    background_bin_edges) * background_bin_heights)  # calculates the area under the histogram between 104MeV and 120MeV
print('The area under the histogram is', area_hist)

A = area_hist / (lamb * (np.exp(-104 / lamb) - np.exp(
    -119.3 / lamb)))  # Finds a value for A, after having equated area_hist with the integral of B(x)
print('A is', A)


# data_area= integrate.cumtrapz(background_data, x=None, dx=1.0, initial=0.0)
# print(data_area)

# popt, pcov = curve_fit(get_B_expectation, bin_centres, bin_edges)

B_x = STOM_higgs_tools.get_B_expectation(bin_edges, A, lamb)
print(B_x)
plt.plot(bin_edges, B_x, label='B(x)')
plt.legend()
plt.show()
