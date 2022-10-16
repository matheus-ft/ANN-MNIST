# Projeto 1 MS/MT-571

## Para rodar o projeto

### Requisitos

- `python3`

  - `pip`

  - `venv`

### Instruções

Clone o repositório e faça:

```sh
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --name=.env
jupytext main.py --to notebook
jupyter lab main.ipynb
```

Em seguida execute as células do notebook.

Se forem feitas mudanças no notebook que devem ser salvas, faça:

```sh
jupytext main.ipynb --to py:percent
```

Caso contrário basta reabrir o notebook numa próxima execução.

### Créditos

A função `theta_meaning` é pesadamente inspirada num _snippet_ encontrado ao pesquisar modos de fazer a visualização dos
tethas mais interessante
(https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib/66961099#66961099).

