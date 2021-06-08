import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits import mplot3d

vals = STOM_higgs_tools.generate_data()

# bin_heights, bin_edges, patches = plt.hist(vals, range=[104, 155], bins=30)

bin_heights, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)

plt.xlabel("$m_{\gamma\gamma} (MeV)$")  # Latex syntax
plt.ylabel("Number of entries")
# plt.show()


# %% Background parametrisation
import scipy.integrate as integrate
import scipy
import scipy.optimize

background_data = [j for j in vals if j < 120]  # to avoid taking the signal bump, upper limit of 120 MeV set
# print(background_data)
N_background = len(background_data)
print('the number of data points in the background is', N_background)

sum_background_data = sum(background_data)
print('the sum of the data points in the background is', sum_background_data)

lamb = sum_background_data / N_background  # maximum likelihood estimator for lambda
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

# data_area = integrate.cumtrapz(background_data, x=None, dx=1.0, initial=0.0)
# print(data_area)

# popt, pcov = curve_fit(get_B_expectation, bin_centres, bin_edges)

B_x = STOM_higgs_tools.get_B_expectation(bin_edges, A, lamb)
print(B_x)
plt.plot(bin_edges, B_x, label='B(x)')
plt.grid()
plt.legend()
plt.show()
# %%
a_values = []
lamb_values = []
chi_squared = []

for l in np.arange(26, 29, 0.1):
    for a in range(70000, 80000, 1000):
        result = STOM_higgs_tools.get_B_chi(vals, [104, 119.3], 9, a, l)
        chi_squared.append(result)
        lamb_values.append(l)
        a_values.append(a)

plt.plot(a_values[0:30], chi_squared[0:30])
fig = plt.figure()

sns.set(style="darkgrid")
ax = plt.axes(projection="3d")
ax.scatter3D(a_values, lamb_values, chi_squared)
plt.show()


#%%

print(STOM_higgs_tools.get_B_chi(vals, [104, 119.3], 9, A, lamb))
