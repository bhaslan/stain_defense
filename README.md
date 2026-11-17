# stain_defense

Code for the article submission "Genomic DNA repeats act as an antitoxin for a widespread Schlafen immune effector"

It includes: sequence preprocessing, domain annotation, embeddings, repeat detection in gene flanks, repeat enrichment models, the protein tree and the figures. Run it with Python 3.12 from the repository root as `python -m src.<package>.pipeline`.

## Layout

- `src/` holds the code, one package per stage (`stage01_preprocessing` to `stage07_figures`), plus `genomes`, `protein_tree`, `repeat_architecture_inference` and `gui`.
- `data/` holds the inputs the code reads and the supplementary tables: `X4Y60Y89016-Alignment.xml` and `X4Y60Y89016-Alignment.csv` (BLASTP results), `mapped_0.csv`, `mapped_1.csv`, `addendum.csv`, `table_S4_homologs/` and `table_S5_flanks/`.
- Every path is built in `src/config_main.py` and the `config_*.py` file of each stage. Stage outputs are written under `data/`.

## Python packages

`biopython`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `h5py`, `matplotlib`, `seaborn`, `python-dotenv`, `tqdm`, `requests`, `edlib`, `Levenshtein`, and `esm` (the ESMC client used by stages 03 and 04). The GUI additionally needs `PyQt5` and `pyqtgraph`.

## External tools and databases

The tools are looked up on `PATH`.

- `mmseqs` (stage 02 clustering)
- `mkvtree`, `vmatch`, `vmatchselect` (Vmatch 2.3.1) and `trf409.linux64` (Tandem Repeats Finder 4.09) for stage 05
- `TreeViewerCommandLine` for the tree figures
- R with CmdStan and glmmTMB, in the conda environments named by `BAYES_ENV` and `GLMM_ENV` in `src/stage06_repeat_enrichment/config_enrichment.py`

The reference databases are expected under `data/db/` (`p_data_db` in `src/config_main.py`): InterProScan 5.77-108.0 data, the Pfam clans table (`Pfam-A.clans.tsv`, v38.0), the RefSeq bacteria and archaea `assembly_summary.txt` files, and the Kraken2 standard taxonomy.

Stage 01 downloads identical protein groups from NCBI and reads `NCBI_EMAIL` and `NCBI_API_KEY` from a `.env` file; the ESMC stages read their token from the environment variable named in their config.
