# Introduction
This is the data repository for the article [“Equivalence between the Axion Invariant and the S4 Symmetry Indicator”](https://arxiv.org/abs/2607.05719).

# Explanation

[tight_binding_S4.ipynb](./tight_binding_S4.ipynb) contains numerical calculation of the toy model, [s4_model.py](./s4_model.py) being its tool package.

[ebr_raw_data8133.txt](./ebr_raw_data8133.txt) is the symmetry data for elementary band representation, 
extracted from open-source site 
[https://cryst.ehu.es/cgi-bin/cryst/programs/mbandrep.pl](https://cryst.ehu.es/cgi-bin/cryst/programs/mbandrep.pl). 

[band_decomposition.ipynb](./band_decomposition.ipynb) is the script to testify the availability to separate the U(N) sewing matrix into U(2) blocks.
[find_basis.py](./find_basis.py) and [gu_cover_solver.py](./gu_cover_solver.py) are the auxiliary files. [find_basis.py](./find_basis.py) gives [hilbert_basis.npy](./hilbert_basis.npy) containing the basis of the symmetry-data vector of all the system in our consideration.

[S4_su2_stabilized_decomposition.ipynb](./S4_su2_stabilized_decomposition.ipynb) proves the availability to reduce U(2) blocks to SU(2) blocks.


# Citation
```
@misc{zhang2026equivalenceaxioninvariants4,
      title={Equivalence between the Axion Invariant and the $S_4$ Symmetry Indicator}, 
      author={Mengyao Zhang},
      year={2026},
      eprint={2607.05719},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mes-hall},
      url={https://arxiv.org/abs/2607.05719}, 
}
```
