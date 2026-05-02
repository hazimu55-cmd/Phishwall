# 🛡️ PhishWall – Intelligent Phishing URL Detection System

PhishWall is a **machine learning-based cybersecurity application** that detects malicious URLs in real time. It analyzes the structure and intent behind a URL to determine whether it is **legitimate or phishing**, helping users avoid potential scams and credential theft.

---

## 🧠 How It Works

1. A user inputs a URL
2. The system extracts key characteristics from the URL
3. These characteristics are converted into numerical features
4. A trained machine learning model analyzes the features
5. The system predicts whether the URL is:

   * ✅ Safe
   * ⚠️ Phishing

---

## 🔍 Feature Engineering (Core Intelligence)

Instead of relying on a single rule, PhishWall evaluates URLs using a **combination of 14 engineered features**, grouped into:

* **Structural patterns** (length, domain complexity, path depth)
* **Symbol and encoding analysis** (special characters, obfuscation, parameters)
* **Security indicators** (HTTPS usage, IP-based domains, spoofing patterns)
* **Behavioral signals** (presence of phishing-related keywords)

This multi-dimensional approach helps the model capture both **technical anomalies** and **social engineering cues** used in phishing attacks.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Model:** Random Forest Classifier
* **Frontend:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ System Workflow

* Input URL → Feature Extraction → ML Model → Prediction Output

The pipeline is designed to be **modular**, allowing easy upgrades to the model or feature set.

---

## 📂 Project Structure

PhishWall/
│── app.py
│── model.pkl
│── feature_extractor.py
│── train_model.py
│── requirements.txt
│── README.md

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/PhishWall.git
cd PhishWall

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Details

* **Algorithm:** Random Forest
* **Type:** Binary Classification
* **Input:** Engineered URL features
* **Output:** Safe / Phishing

---

## ⚠️ Limitations

* May struggle with highly novel (zero-day) phishing techniques
* Accuracy depends on training dataset quality
* Does not yet use real-time threat intelligence

---

## 🔮 Future Scope

* Integration with live threat intelligence APIs
* Advanced models (Deep Learning / NLP-based URL analysis)
* Browser extension for real-time protection
* Enhanced UI and analytics dashboard

---

## 💡 Impact

PhishWall demonstrates how **machine learning can be applied to cybersecurity** to build systems that are both **practical and scalable**, addressing one of the most common online threats today.

---

## 👨‍💻 Author

**Hazim Uddin**
(AI & Data Science)

---

## ⭐ Star the repo if you find it useful!
