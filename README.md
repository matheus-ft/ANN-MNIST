# Projeto 1 MS/MT-571

### Libraries

- numpy
- scipy
- matplotlib
- pandas

### Notebooks

Saved with .py using `# %%` to separate cells and converted to .ipynb using `jupytext`.

Instructions:

```sh
jupytext <file.ipynb> --to py:percent
jupytext <file.py> --to notebook
```

To run using JupyterLab and the virtual environment do

```sh
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --name=.env
jupyter-lab
```

The second line should be replaced by `.env\Scripts\activate` for (ugh) Windows.

### Formatting

Done via `black` and `docformatter`.

