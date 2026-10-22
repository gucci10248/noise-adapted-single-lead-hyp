# Data access and layout

No data are included in this repository. The scripts expect the following
directory layout under the repository root:

```
data/
  ptb-xl/
    ptbxl_database.csv
    scp_statements.csv
    records100/
      00000/00001_lr.hea ...
  chapman/
    WFDBRecords/
      01/010/JS00001.hea ...
  georgia/
    extracted/Georgia/
      E00001.hea
      E00001.mat
```

## PTB-XL v1.0.3

Open access (CC BY 4.0). The download script uses the AWS Open Data mirror:

```bash
python scripts/download_ptbxl.py
```

This downloads the metadata and `records100` (100 Hz). The analysis uses 21,799
records: folds 1–8 training, fold 9 validation, fold 10 test. The 500 Hz
`records500` set is not required.

## Chapman-Shaoxing (`ecg-arrhythmia` 1.0.0)

Open access (CC BY 4.0). First download all headers, then the waveform subset
used in the paper:

```bash
python scripts/download_chapman.py --hea-only --all --workers 32
python scripts/download_chapman.py --from-list data_lists/chapman_records.txt --workers 32
```

`data_lists/chapman_records.txt` contains the 3,234 public record identifiers
used for the original case-control analysis. Of these, 3,227 are readable
(769 HYP-positive and 2,458 HYP-negative); the analysis samples 2,000 negatives
with seed 42. The list contains only public record IDs, not signals or clinical
data.

## Georgia 12-Lead ECG Challenge Database

CC0. The Kaggle mirror is public and does not require an account:

```bash
python scripts/download_georgia.py
```

This downloads a ~506 MB zip and extracts 10,344 records to
`data/georgia/extracted/Georgia/`. The cohort has 1,305 HYP-positive records
(12.62%).

## Licenses and use

- PTB-XL: CC BY 4.0.
- Chapman-Shaoxing: CC BY 4.0.
- Georgia: CC0.

Please cite the original dataset papers and PhysioNet resources when using the
data. The code in this repository is released under the MIT License; the data
remain under their original licenses.
