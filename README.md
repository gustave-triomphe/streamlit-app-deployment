
# Penguins Sex Classifier (Random Forest)

This project trains a **Random Forest classifier** to predict **penguin sex (MALE/FEMALE)** based on body measurements.

## Dataset
A small sample of the Palmer Penguins dataset (`data/penguins_sample.csv`):
- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g
- species, island
- sex (target)

## Code
- `src/model.py` → functions to load data, train a Random Forest, plot confusion matrix & feature importance.
- `notebooks/penguins_rf.ipynb` → Jupyter notebook for exploration, training and plotting results.
- `models/` → stores trained models.

## Usage
```bash
pip install -r requirements.txt

# Train directly
python src/model.py

# Or explore interactively
jupyter notebook notebooks/penguins_rf.ipynb
```

Outputs:
- Classification report (precision, recall, f1)
- Confusion matrix heatmap
- Feature importance plot
- Saved model (`models/rf_penguins.joblib`)
