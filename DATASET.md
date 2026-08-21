# Dataset used by the final experiments

- Dataset: IBM Transactions for Anti-Money Laundering (AML)
- Variant: `Small-HI`
- Required file: `HI-Small_Trans.csv`
- Official project page: https://github.com/IBM/AML-Data
- Kaggle distribution:
  https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Kaggle input: attach **IBM Transactions for Anti Money Laundering** to the
  notebook before running.
- Runtime destination: `/kaggle/working/TH-FraudGT/data/AML/HI-Small_Trans.csv`

The raw CSV is intentionally not committed because it is large. The final
notebook searches `/kaggle/input`, copies the exact file into the expected
location, and writes `dataset_manifest.json` containing its byte size, row
count and SHA-256 digest. This makes the input used for a run auditable without
duplicating the dataset in Git.

IBM describes these records as synthetic transactions generated from a
multi-agent virtual world, not anonymized records of real individuals. Check
the upstream dataset page for the current CDLA-Sharing-1.0 data license.

The split is chronological as implemented by `AMLDataset`: earlier days form
training, followed by validation, then test. Historical features for an edge
at time `t` use only edges with timestamp strictly smaller than `t`.
