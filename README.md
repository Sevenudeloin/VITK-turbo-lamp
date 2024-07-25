# VITK-turbo-lamp
VTK/ITK Project (IMAGE S8)

# AUTHORS
- Ernest Bardon
- Ewan Lemonnier
- Axelle Mandy

## Installation et setup :

`pip install virtualenv`

`python3 -m virtualenv VITK-venv`

## Pour interagir avec le venv :

`source VITK-venv/bin/activate` pour entrer dans le venv

`deactivate` pour quitter le venv

## Ensuite installer les packages dans le venv avec : 

`pip install -r requirements.txt`

### Puis, toujours dans le venv faire la commande suivante pour avoir le kernel jupyter associé :

`python3 -m ipykernel install --user --name VITK-venv --display-name "(VITK-venv)"`

### ensuite, lancer le notebook depuis la racine du projet via :

`jupyter notebook` ou `jupyter lab` 

### Dans jupyter, aller dans 'Kernel', puis 'Change Kernel', et choisir "(VITK-venv)"

