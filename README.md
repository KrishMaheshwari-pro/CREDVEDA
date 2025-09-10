# ⚡ CredVeda – Intelligent Credit Risk Analysis Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Framework-black.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> CredVeda is an AI-powered credit risk assessment platform that leverages machine learning and predictive modeling to evaluate creditworthiness.  
> Designed to help financial institutions, fintechs, and lenders make **smarter, faster, and more reliable** lending decisions.

---

## ✨ Features

- 📊 **Credit Scoring Engine** – Predict borrower creditworthiness using ML models  
- 🤖 **Chatbot Assistant** – Interact with the platform using natural language  
- 📈 **Dashboard** – Visual insights into predictions and risk factors  
- 🗂️ **Data Ingestion Pipeline** – Automated data cleaning and preprocessing  
- 🛠️ **Model Training** – Pre-trained Random Forest & XGBoost models stored with `joblib`  
- 💾 **SQLite Database** – Lightweight, embedded storage for data and results  
- 🌐 **Flask Web App** – Simple and interactive user interface  

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **Frontend**: HTML, CSS, JS (via Flask templates)  
- **Database**: SQLite  
- **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy  
- **Model Storage**: Joblib  

---

## 📂 Project Structure

```bash
CredVeda/
├── app.py                  # Main Flask app entry point
├── chatbot.py              # Chatbot assistant logic
├── config.py               # Configuration file
├── dashboard.py            # Dashboard endpoints
├── data_ingestion.py       # Data preprocessing pipeline
├── database.py             # Database connections
├── model_training.py       # Model training script
├── scoring.py              # Scoring and prediction logic
├── credit_intelligence.db  # SQLite database
├── requirements.txt        # Project dependencies
├── static/                 # Static files (CSS, JS)
├── templates/              # HTML templates
├── output/                 # Model outputs/reports
└── README.md               # Project documentation
🚀 Getting Started
✅ Prerequisites

Python 3.9 or higher

pip (Python package manager)

📥 Installation
# Clone the repository
git clone https://github.com/your-username/CredVeda.git
cd CredVeda

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

▶️ Run the Application
python app.py


Then open http://localhost:5000
 in your browser.

📊 Usage

Upload or input borrower data

View credit score and risk level predictions

Explore the interactive dashboard for insights

Chat with the AI assistant for guidance

🔮 Future Enhancements

🌍 Deploy to cloud (Heroku, AWS, Azure)

🔑 Add authentication and role-based access

📱 Mobile-friendly UI/UX improvements

📡 Integration with real-world credit data APIs

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.
