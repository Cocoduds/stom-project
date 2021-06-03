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
plt.show()