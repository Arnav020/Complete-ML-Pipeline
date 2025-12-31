# MLOps Pipeline Using DVC & AWS

A comprehensive MLOps project implementing a complete machine learning pipeline for spam/ham text classification using DVC (Data Version Control) for experiment tracking and AWS S3 for remote storage.

## 📋 Project Overview

This project demonstrates core MLOps practices including:
- Version control for data and models using DVC
- Pipeline automation with `dvc.yaml`
- Experiment tracking and comparison
- Remote storage integration with AWS S3
- Structured logging throughout the pipeline
- Parameter management with YAML files

The pipeline performs spam/ham classification on text data through 5 distinct stages: data ingestion, preprocessing, feature engineering, model building, and evaluation.

## 🎯 What I Learned

### Core Concepts
- **DVC Fundamentals**: Data versioning, pipeline orchestration, and experiment tracking
- **Remote Storage**: Configuring and using AWS S3 buckets for data/model storage
- **Pipeline Automation**: Creating reproducible ML pipelines with stage dependencies
- **Parameter Management**: Centralized configuration using `params.yaml`
- **Experiment Tracking**: Using DVCLive for metrics and experiment comparison
- **Logging**: Implementing structured logging for debugging and monitoring

### Technical Skills
- Git and DVC integration for ML projects
- YAML file structure and usage
- AWS S3 bucket configuration and access
- Python logging module
- ML pipeline design patterns
- Version control best practices for ML

## 📁 Project Structure
```
MLOPS-PIPELINE-USING-DVC-AWS/
│
├── .dvc/                      # DVC configuration and cache
├── cache/                     # Local cache directory
├── tmp/                       # Temporary files
├── .gitignore                 # Git ignore rules
├── config                     # Configuration files
│
├── .venv/                     # Virtual environment
│
├── data/                      # Data directory (gitignored)
│   ├── interim/               # Intermediate processed data
│   │   ├── test_processed.csv
│   │   └── train_processed.csv
│   ├── processed/             # Final processed data
│   │   ├── test_tfidf.csv
│   │   └── train_tfidf.csv
│   └── raw/                   # Raw data
│       ├── test.csv
│       └── train.csv
│
├── dvclive/                   # DVCLive experiment tracking
├── plots/                     # Generated plots
│
├── experiments/               # Experiment directory
│
├── logs/                      # Log files (gitignored)
│   ├── data_ingestion.log
│   ├── data_preprocessing.log
│   ├── feature_engineering.log
│   ├── model_building.log
│   └── model_evaluation.log
│
├── models/                    # Trained models (gitignored)
│   └── model.pkl
│
├── reports/                   # Reports directory (gitignored)
│   └── metrics.json
│
├── src/                       # Source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── .dvcignore                 # DVC ignore rules
├── .gitignore                 # Git ignore rules
├── dvc.lock                   # DVC lock file (auto-generated)
├── dvc.yaml                   # DVC pipeline configuration
├── LICENSE                    # License file
├── params.yaml                # Pipeline parameters
├── projectflow.txt            # Project workflow notes
└── README.md                  # This file
```

## 🚀 Step-by-Step Guide

### Phase 1: Initial Setup

#### 1. Create GitHub Repository and Clone Locally
```bash
# Create repo on GitHub, then:
git clone 
cd 

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install dvc dvclive pandas scikit-learn numpy
```

#### 2. Create Project Structure
```bash
# Create source directory
mkdir src

# Create your pipeline components in src/:
# - data_ingestion.py
# - data_preprocessing.py
# - feature_engineering.py
# - model_building.py
# - model_evaluation.py
```

**Test each component individually before proceeding!**

#### 3. Configure Git Ignore
```bash
# Add to .gitignore:
data/
models/
reports/
logs/
.venv/
__pycache__/
*.pyc
```
```bash
# Commit initial setup
git add .
git commit -m "Initial project setup"
git push origin main
```

---

### Phase 2: DVC Pipeline Setup (Without Parameters)

#### 4. Initialize DVC
```bash
dvc init
```

#### 5. Create `dvc.yaml` Pipeline
Create `dvc.yaml` with your pipeline stages:
```yaml
stages:
  data_ingestion:
    cmd: python src/data_ingestion.py
    deps:
      - src/data_ingestion.py
    outs:
      - data/raw/train.csv
      - data/raw/test.csv

  data_preprocessing:
    cmd: python src/data_preprocessing.py
    deps:
      - src/data_preprocessing.py
      - data/raw/train.csv
      - data/raw/test.csv
    outs:
      - data/interim/train_processed.csv
      - data/interim/test_processed.csv

  feature_engineering:
    cmd: python src/feature_engineering.py
    deps:
      - src/feature_engineering.py
      - data/interim/train_processed.csv
      - data/interim/test_processed.csv
    outs:
      - data/processed/train_tfidf.csv
      - data/processed/test_tfidf.csv

  model_building:
    cmd: python src/model_building.py
    deps:
      - src/model_building.py
      - data/processed/train_tfidf.csv
    outs:
      - models/model.pkl

  model_evaluation:
    cmd: python src/model_evaluation.py
    deps:
      - src/model_evaluation.py
      - models/model.pkl
      - data/processed/test_tfidf.csv
    metrics:
      - reports/metrics.json:
          cache: false
```

#### 6. Run and Test Pipeline
```bash
# Run the entire pipeline
dvc repro

# Visualize pipeline DAG (Directed Acyclic Graph)
dvc dag

# Commit changes
git add .
git commit -m "Add DVC pipeline configuration"
git push origin main
```

---

### Phase 3: Adding Parameters

#### 7. Create `params.yaml`
```yaml
# params.yaml
data_ingestion:
  test_size: 0.2

preprocessing:
  lowercase: true
  remove_punctuation: true

feature_engineering:
  max_features: 5000
  ngram_range: [1, 2]

model_building:
  algorithm: "LogisticRegression"
  C: 1.0
  max_iter: 1000

model_evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
```

#### 8. Update Pipeline Components
Update your Python scripts to read from `params.yaml`:
```python
import yaml

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Access parameters
max_features = params['feature_engineering']['max_features']
```

#### 9. Update `dvc.yaml` with Parameters
```yaml
stages:
  feature_engineering:
    cmd: python src/feature_engineering.py
    deps:
      - src/feature_engineering.py
      - data/interim/train_processed.csv
      - data/interim/test_processed.csv
    params:
      - feature_engineering.max_features
      - feature_engineering.ngram_range
    outs:
      - data/processed/train_tfidf.csv
      - data/processed/test_tfidf.csv
```

#### 10. Test Pipeline with Parameters
```bash
# Run pipeline
dvc repro

# Commit changes
git add .
git commit -m "Add parameter management"
git push origin main
```

---

### Phase 4: Experiment Tracking

#### 11. Install DVCLive
```bash
pip install dvclive
```

#### 12. Add DVCLive to Evaluation Script
Update `src/model_evaluation.py`:
```python
from dvclive import Live
import json

# Your evaluation code...
accuracy = ...
precision = ...
recall = ...
f1 = ...

# DVCLive tracking
with Live() as live:
    live.log_metric("accuracy", accuracy)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1_score", f1)
    
    # Log parameters
    live.log_param("model", "LogisticRegression")
    live.log_param("max_features", 5000)
```

#### 13. Run Experiments
```bash
# Run as experiment (creates experiment tracking)
dvc exp run

# View all experiments
dvc exp show

# View experiments in table format
dvc exp show --no-pager
```

#### 14. Experiment Management
```bash
# Remove a specific experiment
dvc exp remove 

# Apply a previous experiment
dvc exp apply 

# Compare experiments
dvc exp diff  
```

**VSCode Extension**: Install "DVC" extension for visual experiment tracking.

#### 15. Create Multiple Experiments
```bash
# Modify params.yaml (e.g., change max_features)
# Then run new experiment
dvc exp run

# Compare results
dvc exp show
```

#### 16. Commit Experiments
```bash
git add .
git commit -m "Add experiment tracking with DVCLive"
git push origin main
```

---

### Phase 5: AWS S3 Remote Storage

#### 17. Create AWS S3 Bucket
1. Log into AWS Console
2. Navigate to S3
3. Create new bucket (e.g., `mlops-dvc-remote-storage`)
4. Note the bucket name and region

#### 18. Configure AWS Credentials
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Output format (json)
```

#### 19. Add Remote Storage to DVC
```bash
# Add S3 as remote storage
dvc remote add -d myremote s3://mlops-dvc-remote-storage/dvcstore

# Configure remote
dvc remote modify myremote region us-east-1
```

This creates/updates `.dvc/config`:
```ini
[core]
    remote = myremote
['remote "myremote"']
    url = s3://mlops-dvc-remote-storage/dvcstore
    region = us-east-1
```

#### 20. Push Data to Remote
```bash
# Push data and models to S3
dvc push

# Commit DVC configuration
git add .dvc/config
git commit -m "Add AWS S3 remote storage"
git push origin main
```

#### 21. Pull Data from Remote (On New Machine)
```bash
# Clone repository
git clone 
cd 

# Pull data from S3
dvc pull

# Run pipeline
dvc repro
```

#### 22. Verify Remote Storage
```bash
# Check remote status
dvc remote list

# Check what's tracked
dvc status -r myremote
```

---

## 🔧 Common DVC Commands

### Pipeline Management
```bash
dvc repro                    # Run entire pipeline
dvc repro        # Run specific stage
dvc dag                      # Visualize pipeline
dvc status                   # Check pipeline status
```

### Experiment Tracking
```bash
dvc exp run                  # Run as experiment
dvc exp show                 # Show all experiments
dvc exp diff                 # Compare experiments
dvc exp apply          # Apply experiment
dvc exp remove         # Remove experiment
```

### Remote Storage
```bash
dvc push                     # Push to remote
dvc pull                     # Pull from remote
dvc fetch                    # Fetch from remote (no checkout)
dvc status -r        # Check remote status
```

### Data Management
```bash
dvc add                # Track file with DVC
dvc checkout                 # Checkout tracked files
dvc gc                       # Garbage collect old versions
```

---

## 📊 Logging Implementation

Each pipeline stage includes structured logging:
```python
import logging
import os

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_ingestion.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.info("Starting data ingestion")
logger.error("Error occurred: %s", error_message)
```

---

## 🔑 Key Takeaways

1. **DVC vs Git**: Git tracks code, DVC tracks data and models
2. **Pipeline Automation**: `dvc repro` automatically runs only changed stages
3. **Experiment Tracking**: DVCLive makes experiment comparison easy
4. **Remote Storage**: S3 enables team collaboration and data backup
5. **Parameters**: Centralized params make experimentation systematic
6. **Reproducibility**: Anyone can clone and reproduce your results

---

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVCLive Documentation](https://dvc.org/doc/dvclive)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [MLOps Best Practices](https://ml-ops.org/)

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements!

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

Your Name - [Your GitHub Profile](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- DVC team for excellent documentation
- MLOps community for best practices
- AWS for reliable cloud storage

---

**Happy Learning! 🚀**