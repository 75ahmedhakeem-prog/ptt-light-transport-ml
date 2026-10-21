# XGBoost tuning run (Google Colab)
1. Runtime -> Change runtime type -> T4 GPU (optional, makes it ~5x faster; CPU also works).
2. Upload `ptt_colab_xgb_tuning.zip` with the folder icon on the left.
3. Run this single cell:

```
!pip -q install xgboost
!unzip -o -q ptt_colab_xgb_tuning.zip
!gunzip -f design_dataset.csv.gz
!python xgb_tune.py --data design_dataset.csv --out results_tuned
!zip -q -r results_tuned.zip results_tuned
from google.colab import files
files.download("results_tuned.zip")
```
4. Send `results_tuned.zip` back. Runtime: about 15 min on GPU, 60-90 min on CPU.
