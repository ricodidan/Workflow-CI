# Workflow CI - MLflow Project & Docker Hub

Submission Kriteria 3: Membuat Workflow CI (Tingkat Advance).

## Struktur Repositori
```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   ├── modelling.py
│   ├── telco_churn_preprocessing/
│   │   └── customer_churn_clean.csv
│   └── Tautan ke Docker Hub.txt
└── README.md
```

## Deskripsi Singkat
- **MLProject/**: Folder project MLflow yang memuat konfigurasi project (`MLproject`), environment dependencies (`conda.yaml`), script training (`modelling.py`), dan dataset hasil preprocessing.
- **Workflow CI (`.github/workflows/ci.yml`)**:
  1. Trigger otomatis saat ada `push` atau `workflow_dispatch`.
  2. Melatih model machine learning.
  3. Mengunggah artefak model ke GitHub Actions Artifacts (Kategori Skilled).
  4. Membangun Docker Image model via `mlflow models build-docker` dan mem-push image ke Docker Hub (Kategori Advance).
