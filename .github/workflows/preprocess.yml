name: Preprocess Titanic Dataset

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  preprocess:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy scikit-learn

      - name: Run preprocessing script
        run: |
          python preprocessing/automate_Muhammad-Rafi-Insani.py

      - name: Upload processed dataset as artifact
        uses: actions/upload-artifact@v4
        with:
          name: titanic_preprocessed
          path: preprocessing/titanic_preprocessing/

      - name: Commit processed output back to repo
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add preprocessing/titanic_preprocessing/titanic_processed.csv preprocessing/titanic_preprocessing/feature_columns.txt
          git commit -m "Auto-update preprocessed dataset" || echo "No changes to commit"
          git push
