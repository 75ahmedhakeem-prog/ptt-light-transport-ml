# Running the XGBoost part in Google Colab

1. Open https://colab.research.google.com and create a new notebook.
2. In the left panel, click the folder icon and upload `design_dataset.csv.gz` and `ml_design.py`.
3. Run these two cells:

```
!pip -q install xgboost
```

```
!gunzip -f design_dataset.csv.gz
!python ml_design.py --data design_dataset.csv --out results --n-train 60000
```

4. When the second cell prints `done`, download the folder `results` (right-click -> Download)
   or zip it first:

```
!zip -r results.zip results
```

5. Send `results.zip` back.

The run takes about 30-60 minutes on the free Colab CPU.
