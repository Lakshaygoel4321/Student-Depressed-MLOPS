# 🧠 Student Depression Prediction - MLOps Project

An end-to-end MLOps pipeline for predicting student depression using structured data and modern MLOps practices. The pipeline integrates data ingestion, validation, transformation, model training, evaluation, and containerization — with CI/CD automation via GitHub Actions. MongoDB Atlas is used as the primary data source.

---

## 🌟 Key Features

* **Complete MLOps Workflow:** From data ingestion to model training and evaluation.
* **MongoDB Integration:** Fetches live data from MongoDB Atlas.
* **Model Evaluation:** Automatically evaluates model performance using predefined metrics.
* **CI/CD Automation:** GitHub Actions automate testing, packaging, and container build.
* **Containerized Solution:** Docker ensures reproducible builds and portability.
* **Modular Codebase:** Separated concerns using a clean MLOps architecture.

---

## 🛠️ Tech Stack and Tools

* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Flask, PyMongo
* **Database:** MongoDB Atlas
* **Environment:** Conda
* **CI/CD:** GitHub Actions
* **Containerization:** Docker
* **Cloud Hosting:** AWS EC2 (for optional deployment)

---

## ⚙️ Architecture Overview

This MLOps pipeline is divided into modular, manageable stages:

1. **Data Ingestion:** Extracts structured student data from MongoDB Atlas.
2. **Data Validation:** Ensures schema consistency and data quality.
3. **Data Transformation:** Handles preprocessing such as encoding, normalization, and feature selection.
4. **Model Training:** Trains a binary classification model to predict depression.
5. **Model Evaluation:** Evaluates trained models against defined metrics (e.g., accuracy, F1-score).
6. **CI/CD:** Uses GitHub Actions to automate the build, test, and Docker packaging steps.

> ❗️Note: There is no `model_pusher.py` or automated deployment module in this version.

---

## 📂 Project Structure

```plaintext
student-depression-mlops/
├── src/
│   ├── pipeline/            # Training and prediction pipelines
│   ├── components/          # Data ingestion, validation, transformation
│   ├── entity/              # Config and artifact schema definitions
│   ├── configuration/       # MongoDB, logging, etc.
│   ├── model_evaluation/    # Model metrics and evaluation logic
│   ├── utils/               # Reusable helper functions
│   └── app.py               # Main Flask app for serving predictions
├── templates/               # HTML templates for web app
├── static/                  # CSS/JS/static assets
├── .github/workflows/       # CI/CD config with GitHub Actions
│   └── deploy.yaml
├── Dockerfile               # Docker build instructions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10
* Conda
* MongoDB Atlas account
* Docker
* GitHub account

### Installation Steps

1. **Clone the repository**

```sh
git clone https://github.com/yourusername/student-depression-mlops.git
cd student-depression-mlops
```

2. **Create and activate the Conda environment**

```sh
conda create -n studentml python=3.10 -y
conda activate studentml
```

3. **Install the dependencies**

```sh
pip install -r requirements.txt
```

4. **Set your MongoDB connection string**

```sh
export MONGODB_URL="your_connection_string"
```

---

## ⚙️ CI/CD Pipeline

The CI/CD pipeline is configured using GitHub Actions (`.github/workflows/deploy.yaml`) to:

* Build the project
* Test core components
* Create Docker images for reproducible builds

> ✅ Set the following repository **secrets** in GitHub:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `EC2_HOST` (optional, if deploying)
* `EC2_USER`
* `EC2_KEY`

---

## 🐳 Docker Usage

### Build Docker Image

```sh
docker build -t student-depression-app .
```

### Run the App

```sh
docker run -d -p 5000:5000 student-depression-app
```

Visit: `http://localhost:5000`

---

## 📈 Model Evaluation

The `model_evaluation` module:

* Calculates key metrics (accuracy, precision, recall, F1-score)
* Compares trained models with previous versions (if available)
* Logs results for review and future comparison

> You can trigger evaluation by running:

```sh
python src/pipeline/train_pipeline.py
```

---

## 🔍 Future Enhancements

* Add automated monitoring and alerting
* Integrate unit testing and pre-commit hooks
* Build user authentication for prediction UI

---

## 👨‍💻 Author

**Lakshay**

* [LinkedIn](https://www.linkedin.com/in/lakshay-goel-b10878326)
* [GitHub](https://github.com/Lakshaygoel4321)

---

## 📝 License

Licensed under the MIT License. See the [LICENSE](MIT) file for details.

