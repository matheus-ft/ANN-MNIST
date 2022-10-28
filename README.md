# Regularized neural network for identification of handwritten single digits

## Notes

This was originally a repo for a uni project, but the master branch might contain changes made after such venture,
including, but not limited to, translation and contribution from more or less people.

## Dependencies

- `python3`, version >= 3.10

  - `pip`

  - `venv`

## Instructions

Clone the repo and do:

```sh
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --name=.env
jupytext main.py --to notebook
jupyter lab main.ipynb
```

Then execute the notebook cells.

If any changes that should be kept are made, save the file and do:

```sh
jupytext main.ipynb --to py:percent
```

Otherwise, simply open the notebook again in another ocasion.

## Credits

The function `theta_meaning` is heavily inspired in
https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib/66961099#66961099.

