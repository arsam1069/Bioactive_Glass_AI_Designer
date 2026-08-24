# Bioactive Glass AI Designer V1

Python Streamlit research dashboard for **AI-assisted chemical design of bioactive glasses**.

This project uses the user's `MvsG (1).xlsx` dataset as the default base data and supports future CSV/Excel uploads.

## What it does

- Editable chemical composition table
- wt% and mol% default datasets
- Row normalization to 100%
- PCA score plot
- PCA biplot
- Explained variance plot
- Feature contribution plot
- PCA scores and loadings tables
- Chemical descriptors:
  - network formers = SiO2 + P2O5
  - network modifiers = CaO + Na2O + K2O + MgO
  - intermediates = Al2O3 + TiO2 + FeO(T) + MnO
  - CaO/P2O5 ratio
  - modifier/former ratio
- Min/max design constraints
- Constraint issue table
- Rule-based AI-assisted design ranking
- Candidate composition generator
- CSV, Excel and text report exports

## Important scientific note

The current design score is **not a trained ML prediction model**. It is a transparent, rule-based screening system for exploratory glass design.

For real ML prediction, add measured target columns such as:

- bioactivity result
- apatite formation
- dissolution rate
- pH change
- hardness
- density
- compressive strength
- antibacterial activity

Then the app can be upgraded to train predictive models.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Upload this folder to a GitHub repository.
2. Open Streamlit Cloud.
3. Create a new app from the repository.
4. Main file path: `app.py`
5. Deploy.

## Folder structure

```text
Bioactive_Glass_AI_Designer_V1/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── default_wt_percent.csv
│   ├── default_mol_percent.csv
│   └── MvsG_original_dataset.xlsx
└── modules/
    ├── preprocessing.py
    ├── descriptors.py
    ├── pca_analysis.py
    ├── scoring.py
    └── plotting.py
```
