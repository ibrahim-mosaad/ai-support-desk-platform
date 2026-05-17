#  AI Support Desk Platform (Production SaaS)

An AI-powered customer support intelligence system that automatically classifies support tickets, analyzes trends, and provides real-time business insights using Machine Learning and a Streamlit-based SaaS dashboard.

---

#  Overview

Modern customer support systems handle thousands of tickets daily across multiple channels.

This platform solves this problem by:

- Automatically classifying tickets using AI
- Providing real-time analytics
- Tracking customer support performance
- Simulating a real SaaS support system (like Zendesk / Freshdesk)

---

#  Key Features

###  AI Capabilities
- Smart ticket classification
- Confidence scoring
- NLP-based feature extraction

###  Analytics Dashboard
- Most common issues detection
- Category distribution analysis
- Daily ticket volume trends

###  Production Features
- Logging system (CSV-based)
- Multi-input form (subject, description, product, channel)
- Real-time prediction engine
- Modular architecture

---

#  System Architecture

```

User Input (Streamlit UI)
↓
Data Cleaning & Validation
↓
Feature Engineering (TF-IDF)
↓
Machine Learning Model
↓
Prediction Output
↓
Logging System
↓
Analytics Dashboard

```

---

#  Machine Learning Model

- Algorithm: Multinomial Naive Bayes
- Feature Extraction: TF-IDF (n-grams 1–2)
- Text Processing: NLP preprocessing pipeline
- Training Type: Supervised Learning (Classification)

---

#  Dataset

This project uses a real-world customer support dataset containing:

### Features:
- Ticket Subject
- Ticket Description
- Ticket Type
- Product Purchased
- Ticket Channel
- Ticket Priority
- Customer Metadata

Dataset Source:
https://www.kaggle.com/datasets/adityasharma01/customer-support-ticket-classification

---

# 📁 Project Structure

```

customer-ticket-ai/
│
├── app/
│   ├── streamlit_app.py
│   ├── utils.py
│
├── model/
│   ├── model.pkl
│   ├── vectorizer.pkl
│
├── training/
│   ├── train.py
│   ├── dataset.csv
│
├── logs/
│   ├── predictions.csv
│
├── requirements.txt
├── README.md

````

---

#  Installation & Setup

##  Clone Repository

```bash
git clone https://github.com/ibrahim-mosaad/ai-support-desk-platform.git
cd ai-support-desk-platform
````

---

##  Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

##  Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Train the Model

```bash
python training/train.py
```

This will generate:

* model.pkl
* vectorizer.pkl

---

##  Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

#  Application Modules

##  1. Ticket Prediction Module

* Input customer ticket details
* AI-based classification
* Confidence score output

---

##  2. Analytics Dashboard

Includes:

*  Most common issues
*  Category distribution
*  Daily ticket volume trends

---

##  3. Logging System

Each prediction is stored with:

* Timestamp
* Input data
* Predicted category
* Confidence score

---

#  Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. TF-IDF Vectorization
5. Model Training (Naive Bayes)
6. Prediction Engine
7. Logging & Monitoring
8. Dashboard Visualization

---

#  Example Prediction

### Input:

```
"My internet is not working and I need urgent help"
```

### Output:

```
Prediction: Technical Issue
Confidence: 93.4%
```

---

#  Business Impact

This system helps organizations:

* Reduce customer support workload
* Automate ticket routing
* Improve response time
* Enhance customer satisfaction
* Lower operational costs

---

#  Production Highlights

* Modular and scalable architecture
* Real-time inference
* SaaS-style dashboard
* Logging and monitoring system
* Extensible ML pipeline

---

#  Future Improvements

*  Upgrade model to XGBoost / LightGBM
*  Add FastAPI backend
*  Integrate PostgreSQL database
*  Add authentication system (Admin / Agent roles)
*  Deploy on cloud (AWS / Render / Streamlit Cloud)
*  Add CI/CD pipeline (GitHub Actions)
*  Docker containerization
*  Real-time streaming analytics

---

#  Tech Stack

* Python 
* Pandas 
* Scikit-learn 
* Streamlit 
* TF-IDF NLP 
* Matplotlib 
* Joblib 

---

#  Author

**Ibrahim Hamada Massaad**
AI Engineer | Data Science Instructor | NTI

---

#  License

This project is for educational and research purposes.

---

#  Support

If you like this project, please give it a ⭐ on GitHub.
It helps improve visibility and supports development 
FAANG Portfolio"

```
