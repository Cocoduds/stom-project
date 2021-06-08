import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np

vals = STOM_higgs_tools.generate_data()

# bin_heights, bin_edges, patches = plt.hist(vals, range=[104, 155], bins=30)

bin_heights, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)


plt.xlabel("$m_{\gamma\gamma} (GeV)$")
plt.ylabel("Number of entries")
plt.xlabel("$m_{\gamma\gamma} (MeV)$")
plt.ylabel("Number of entries")
#plt.show()


#%% Background parameterisation
import scipy.integrate as integrate
import scipy
from scipy.optimize import curve_fit 
background_data=[j for j in vals if j < 120] #to avoid taking the signal bump, upper limit of 120 MeV set
#print(background_data)
N_background=len(background_data)
print('the number of data points in the background is', N_background)
sigma_background_data=sum(background_data)
print('the sum of the data points in the background is', sigma_background_data)
lamb = (sigma_background_data)/(N_background) #maximum likelihood estimator for lambda
print('lambda estimate is', lamb)

background_bin_edges = [k for k in bin_edges if 103 < k < 120 ] #takes the histogram values only within the background data region
background_bin_heights = bin_heights[0:9] #As above, only takes the bin_heights within the background data region
area_hist = sum(np.diff(background_bin_edges)*background_bin_heights) #calculates the area under the histogram between 104MeV and 120MeV
print('The area under the histogram is', area_hist)


A = area_hist/(lamb*(np.exp(-104/lamb)-np.exp(-119.3/lamb))) #Finds a value for A, after having equated area_hist with the integral of B(x)
print('A is', A)
#data_area= integrate.cumtrapz(background_data, x=None, dx=1.0, initial=0.0)
#print(data_area)

#popt, pcov = curve_fit(get_B_expectation, bin_centres, bin_edges)

def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A*np.exp(-x/lamb) for x in xs]

B_x = get_B_expectation(bin_edges, A, lamb)
print(B_x)
plt.plot(bin_edges, B_x, label='B(x)')
plt.legend()

plt.show()

#%%

chi_value_background_only = STOM_higgs_tools.get_B_chi(background_data,(104,119.3),9,A,lamb) #it is 9 since we consider only before the bump
print(chi_value_background_only , 'wtaf')

chi_value_with_signal = STOM_higgs_tools.get_B_chi(vals,(104,155),30,A,lamb)
print(chi_value_with_signal, 'seeing as this is much higher than for the signal only hypothesis, that implies signla-only is not such a good idea')


#%%

chi_valus = STOM_higgs_tools.get_B_chi(background_data,(104,119.3),9,A,lamb) #it is 9 since we consider only before the bump
print(chi_valus , 'wtaf')

chi_value = STOM_higgs_tools.get_B_chi(vals,(104,155),30,A,lamb)
print(chi_value, 'seeing as this is much higher than for the signal only hypothesis, that implies signla-only is not such a good idea')


#%%


from scipy.stats import chi2
p_value_background_only = chi2.sf(chi_valus, 1)
print(p_value_background_only, 'implies we can reject the background only hypotehsis at 10% sig level')

p_value_with_signal = chi2.sf(chi_value, 1)
print(p_value_with_signal,'implies can be accepted at 97% confidence level')

#%%

chi_vals = []

for i in range(10000):
    vals = STOM_higgs_tools.generate_data()
    background_data=[j for j in vals if j < 120] #to avoid taking the signal bump, upper limit of 120 MeV set
    lamb = (sigma_background_data)/(N_background) #maximum likelihood estimator for lambda
    bin_h, bin_edges = np.histogram(vals, range=[104, 155], bins=30)
    area_hist = sum(np.diff(background_bin_edges)*background_bin_heights)
    A = area_hist/(lamb*(np.exp(-104/lamb)-np.exp(-119.3/lamb)))
    B_x = get_B_expectation(bin_edges, A, lamb)
    chi_valus = STOM_higgs_tools.get_B_chi(background_data,(104,119.3),9,A,lamb)
    chi_vals.append(chi_valus)
    next

print(chi_vals)

bin_h, bin_e = np.histogram(chi_vals, range=[0, 7], bins=40)
bin_c = (bin_e[:-1] + bin_e[1:])/2.
plt.errorbar(bin_c, bin_h, np.sqrt(bin_h), fmt=',', capsize=2)

plt.grid()
plt.xlabel('$\chi^{2}$ Val')
plt.ylabel('Frequency')
plt.show()
#%%


bin_heights, bin_edges = np.histogram(chi_vals, range=[0, 7], bins=30)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(bin_centres, bin_heights, np.sqrt(bin_heights), fmt=',', capsize=2)

plt.show()

