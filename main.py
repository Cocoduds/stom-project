import STOM_higgs_tools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import chi2
import scipy
#%%
vals = STOM_higgs_tools.generate_data()
#%%
"""Add a cell break above to stop regeneration of data"""
sns.set_style("ticks")
# bin_heights, bin_edges, patches = plt.hist(vals, range=[104, 155], bins=30)

bin_heights, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
# sns.scatterplot(x=bin_centres, y=bin_heights, s=25, marker='x', color='blue')
# plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)

# plt.xlabel("$m_{\gamma\gamma} (MeV)$")
# plt.ylabel("Frequency")
# plt.show()

import scipy.integrate as integrate

from scipy.optimize import curve_fit

background_data = [j for j in vals if j < 120]  # to avoid taking the signal bump, upper limit of 120 MeV set
# print(background_data)
N_background = len(background_data)
print('the number of data points in the background is', N_background)
sigma_background_data = sum(background_data)
print('the sum of the data points in the background is', sigma_background_data)
lamb = (sigma_background_data) / (N_background)  # maximum likelihood estimator for lambda
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

B_x = STOM_higgs_tools.get_B_expectation(bin_centres, A, lamb)
# C_x = STOM_higgs_tools.get_B_expectation(bin_edges, 62500, 29.2)

print(B_x)

# sns.lineplot(x=bin_edges, y=C_x, label='C(x)', color='Purple') # These parameters are taken from the 2D search

# plt.legend()
# sns.despine()
#
# plt.show()
#%%
"""Takes around a minute to run"""
a_values = []
lamb_values = []
chi_squared = []

for l in np.arange(27, 28, 0.02):
    for a in range(77000, 80000, 100):
        result = STOM_higgs_tools.get_B_chi(vals, [104, 119.3], 9, a, l)
        chi_squared.append(result)
        lamb_values.append(l)
        a_values.append(a)

sns.set(style="darkgrid")

ax = plt.axes(projection="3d")
ax.scatter3D(a_values, lamb_values, chi_squared)
plt.xticks(np.arange(77000, 80000, 500))

ax.set_zlabel('Reduced $\chi^{2}$ value')
# ax.set_zlim(0, 10)

plt.xlabel('A')
plt.ylabel('$\lambda$')


plt.show()
#%%
"""Run this after previous cell"""
i_chi2min = np.argmin(chi_squared)  # gives index where chi_squared is min.
print(f"A = {a_values[i_chi2min]}, lamb = {lamb_values[i_chi2min]}, chi2 = {chi_squared[i_chi2min]}")
print(STOM_higgs_tools.get_B_chi(vals, [104, 119.3], 9, A, lamb))

print(np.sort(chi_squared)[:10])  # This is the 10 smallest chi_squared values for different a and lamb combinations
#%%
chi_value_background_only = STOM_higgs_tools.get_B_chi(background_data, (104, 119.3), 9, A,
                                                       lamb)  # it is 9 since we consider only before the bump
print(chi_value_background_only, 'wtaf')

chi_value_with_signal = STOM_higgs_tools.get_B_chi(vals, (104, 155), 30, A, lamb)
print(chi_value_with_signal,
      'seeing as this is much higher than for the signal only hypothesis, that implies signla-only is not such a good idea')

#%%

p_value_background_only = chi2.sf(chi_valus, 1)
print(p_value_background_only, 'implies we can reject the background only hypotehsis at 10% sig level')

p_value_with_signal = chi2.sf(chi_value, 1)
print(p_value_with_signal, 'implies can be accepted at 97% confidence level')

#%%
chi_vals = []

for i in range(10000):
    vals = STOM_higgs_tools.generate_data()
    background_data = [j for j in vals if j < 120]  # to avoid taking the signal bump, upper limit of 120 MeV set
    lamb = sigma_background_data / N_background  # maximum likelihood estimator for lambda
    bin_h, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
    area_hist = sum(np.diff(background_bin_edges) * background_bin_heights)
    A = area_hist / (lamb * (np.exp(-104 / lamb) - np.exp(-119.3 / lamb)))
    B_x = STOM_higgs_tools.get_B_expectation(bin_edges, A, lamb)
    chi_valus = STOM_higgs_tools.get_B_chi(background_data, (104, 119.3), 9, A, lamb)
    chi_vals.append(chi_valus)

print(chi_vals)

bin_h, bin_e = np.histogram(chi_vals, range=[0, 7], bins=40)
bin_c = (bin_e[:-1] + bin_e[1:]) / 2.
sns.scatterplot(x=bin_c, y=bin_h, s=25, marker='x', color='blue')
plt.errorbar(bin_c, bin_h, np.sqrt(bin_h), fmt=',', capsize=2)

sns.set_style("ticks")
sns.despine()

plt.xlabel('Reduced $\chi^{2}$ value')
plt.ylabel('Frequency')
plt.show()


#%%

def exponential(x, A, lamb):
    return A * np.exp(-x / lamb)

chi_v = []

for i in range(10000):
    vals = STOM_higgs_tools.generate_data()
    background_data = [j for j in vals if j < 120]  # to avoid taking the signal bump, upper limit of 120 MeV set
    popt = curve_fit(exponential, background_data)
    bin_h, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
    area_hist = sum(np.diff(background_bin_edges) * background_bin_heights)
    A = area_hist / (lamb * (np.exp(-104 / lamb) - np.exp(-119.3 / lamb)))
    B_x = get_B_expectation(bin_edges, A, lamb)
    chi_valus = STOM_higgs_tools.get_B_chi(background_data, (104, 119.3), 9, A, lamb)
    chi_v.append(chi_valus)

print(chi_v)
#%%

bin_heights, bin_edges = np.histogram(chi_vals, range=[0, 7], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)

plt.show()


#%%

def fit_func(x):
    gaus = 700 * np.exp(-(x - 125) ** 2 / (2 * 1.5 ** 2))
    expo = A * np.exp(-x / lamb)
    return gaus + expo


# bin_heights, bin_edges, patches = plt.hist(vals, range=[104, 155], bins=30)

bin_heights_boi, bin_edges_boi = np.histogram(vals, range=[104, 155], bins=30)
bin_centres_boi = (bin_edges_boi[:-1] + bin_edges_boi[1:]) / 2.
plt.errorbar(bin_centres_boi, bin_heights_boi, np.sqrt(bin_heights_boi), fmt=',', capsize=2)

x = np.linspace(100, 155, 200)
y = fit_func(x)

sns.set_style("ticks")
sns.lineplot(x=x, y=y, label='B+S')


def get_B_chi_gaussian(vals, mass_range, nbins):
    """
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analytic form,
    using the hard coded scale of the exp. That depends on the binning, so pass
    in as argument. The mass range must also be set - otherwise, its ignored.
    """
    bin_heights, bin_edges = np.histogram(vals, range=mass_range, bins=nbins)
    half_bin_width = 0.5 * (bin_edges[1] - bin_edges[0])
    ys_expected = fit_func(bin_edges + half_bin_width)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i]) ** 2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi / float(nbins - 3)  # B has 2 parameters.


chi_value_gaussfit = get_B_chi_gaussian(vals, (104, 155), 30)
print(chi_value_gaussfit,
      'seeing as this is much higher than for the signal only hypothesis, that implies signla-only is not such a good idea')


def fit_func_optimise(x, a, mu, sig, A, lamba):
    gaus = a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    expo = A * np.exp(-x / lamba)
    return gaus + expo


fit, cov = curve_fit(fit_func_optimise, bin_centres_boi, bin_heights_boi, p0=[300, 125, 1.5,A,lamb])

sns.lineplot(x=x, y=fit_func_optimise(x, *fit), label='B+S optimised fit')

# =============================================================================
# print(fit)
# print(cov)
# print('frac uncertainty in amplitude of gaussian',np.sqrt(cov[0,0])/fit[0],np.sqrt(cov[1,1]))
# =============================================================================

def get_B_chi_gaussian_optimal(vals, mass_range, nbins):
    """
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analytic form,
    using the hard coded scale of the exp. That depends on the binning, so pass
    in as argument. The mass range must also be set - otherwise, its ignored.
    """
    bin_heights, bin_edges = np.histogram(vals, range=mass_range, bins=nbins)
    half_bin_width = 0.5 * (bin_edges[1] - bin_edges[0])
    ys_expected = fit_func_optimise(bin_edges + half_bin_width, *fit)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i]) ** 2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi / float(nbins - 3)  # B has 2 parameters.


chi_value_gaussfit_optimal = get_B_chi_gaussian_optimal(vals, (104, 155), 30)
print(chi_value_gaussfit_optimal)
p_value_gaussfit_optimal = chi2.sf(chi_value_gaussfit_optimal,1)

print(p_value_gaussfit_optimal)
sns.lineplot(x=bin_centres, y=B_x, label='B', color='Black', linestyle='--')
sns.scatterplot(x=bin_centres, y=bin_heights, s=25, marker='x', color='blue')
# sns.lineplot(x=bin_centres, y=STOM_higgs_tools.get_B_expectation(bin_centres, 77900, 27.62), label='2D search', color="Red")
sns.despine()

plt.xlabel("$m_{\gamma\gamma}$ (GeV)")
plt.ylabel("Frequency")
plt.legend()
plt.xlim((104, 155))
plt.savefig("boyo", bbox_inches='tight')
plt.show()
#%%


def for_looping_chis_gauss(x, mu):
    gaus = fit[0] * np.exp(-(x - mu) ** 2 / (2 * fit[2] ** 2))
    expo = fit[3] * np.exp(-x / fit[4])
    return gaus + expo


masses = np.linspace(110, 150, 80)


def get_chi_varying_mass(vals, mass_range, nbins, mu):
    """
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analytic form,
    using the hard coded scale of the exp. That depends on the binning, so pass
    in as argument. The mass range must also be set - otherwise, its ignored.
    """
    bin_heights, bin_edges = np.histogram(vals, range=mass_range, bins=nbins)
    half_bin_width = 0.5 * (bin_edges[1] - bin_edges[0])
    ys_expected = for_looping_chis_gauss(bin_edges + half_bin_width, mu)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i]) ** 2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi / float(nbins - 3)  # B has 2 parameters.

chi_masses = []

for mass in masses:
    chi = get_chi_varying_mass(vals, (104, 155), 30, mass)
    chi_masses.append(chi)

sns.set_style('ticks')

plt.ylabel('Reduced $\chi^{2}$ Value')
plt.xlabel("$m_{\gamma\gamma}$ (GeV)")

sns.lineplot(x=masses, y=chi_masses)
sns.despine()
plt.savefig("mass_against_chi", bbox_inches='tight')

plt.show()
