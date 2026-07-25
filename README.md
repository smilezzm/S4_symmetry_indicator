# Introduction
This is the data repository for the article “Equivalence between the Axion Invariant and the S4 Symmetry Indicator”.

# Explanation
[ebr_raw_data8133.txt](./ebr_raw_data8133.txt) is the symmetry data for elementary band representation, 
extracted from open-source site 
[https://cryst.ehu.es/cgi-bin/cryst/programs/mbandrep.pl](https://cryst.ehu.es/cgi-bin/cryst/programs/mbandrep.pl). 

[band_decomposition.ipynb](./band_decomposition.ipynb) is the script to testify the availability to separate the U(N) sewing matrix into U(2) blocks.
[find_basis.py](./find_basis.py) and [gu_cover_solver.py](./gu_cover_solver.py) are the auxiliary files.

[S4_su2_stabilized_decomposition.ipynb](./S4_su2_stabilized_decomposition.ipynb) proves the availability to reduce U(2) blocks to SU(2) blocks.

[tight_binding_S4.ipynb](./tight_binding_S4.ipynb) solves the tight-binding model. 
