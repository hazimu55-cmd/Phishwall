# 🛡️ PhishWall – Intelligent Phishing URL Detection System

PhishWall is a **machine learning-based cybersecurity application** that detects malicious URLs in real time. It analyzes the structure and intent behind a URL to determine whether it is **legitimate or phishing**, helping users avoid potential scams and credential theft.

🔗 **Live Demo:** [Try PhishWall](https://phishwall-m7m.streamlit.app/)
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

Instead of relying on a single rule, PhishWall evaluates URLs using a **combination of 20 engineered features** (15 original + 5 advanced), grouped into:

* **Structural patterns** (length, domain complexity, path depth)
* **Symbol and encoding analysis** (special characters, obfuscation, parameters)
* **Security indicators** (HTTPS usage, IP-based domains, spoofing patterns)
* **Behavioral signals** (presence of phishing-related keywords)
* **Advanced entropy analysis** (Shannon entropy for randomness detection)
* **Character distribution analysis** (digit and special character ratios)

This multi-dimensional approach helps the model capture both **technical anomalies** and **social engineering cues** used in phishing attacks. The recent addition of entropy-based features significantly improves detection of obfuscated and randomly-generated phishing domains.

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
│── app.py                      # Streamlit web application
│── feature_extraction.py       # Feature extraction logic (20 features)
│── train_model.py              # Model training script
│── models/
│   └── phishing_model.pkl      # Trained Random Forest model
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/PhishWall.git
cd PhishWall

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Train the model (required first run)
python train_model.py

# Run the application
streamlit run app.py
```

---

## � Recent Updates (2025)

### **Advanced Feature Integration**
- **Added 5 critical advanced features** (total now 20 features: 15 original + 5 advanced)
- **Entropy analysis:** Shannon entropy for randomness detection
- **Non-alphanumeric entropy:** Critical for obfuscation detection
- **Character distribution:** Digit and special character ratios
- **Improved accuracy:** 99.51% on 235K+ URL dataset
- **Memory optimization:** Efficient batch processing for large datasets

### **Enhanced Detection Capabilities**
- Better detection of randomly-generated phishing domains
- Improved identification of obfuscated URLs
- Enhanced character pattern analysis
- More sophisticated rule-based scoring system

---

## �📊 Model Details

* **Algorithm:** Random Forest Classifier
* **Type:** Binary Classification
* **Input:** 20 engineered URL features (15 original + 5 advanced)
* **Output:** Safe / Phishing with confidence score
* **Performance:** 99.51% accuracy on 235K+ URL dataset
* **Advanced Features:** Entropy analysis, character distribution, randomness detection

---

## ⚠️ Limitations

* May struggle with highly novel (zero-day) phishing techniques
* Accuracy depends on training dataset quality
* Does not yet use real-time threat intelligence
* Advanced obfuscation techniques may bypass current detection
* Model requires periodic retraining with new phishing patterns

---

## 🔮 Future Scope

* Integration with live threat intelligence APIs
* Advanced models (Deep Learning / NLP-based URL analysis)
* Browser extension for real-time protection
* Enhanced UI and analytics dashboard
* Real-time URL scanning and reputation checking
* Additional advanced features (visual similarity, homograph detection)

---

## 💡 Impact

PhishWall demonstrates how **machine learning can be applied to cybersecurity** to build systems that are both **practical and scalable**, addressing one of the most common online threats today.

---

## 👨‍💻 Author

**Hazim Uddin**
(AI & Data Science)

---

## ⭐ Star the repo if you find it useful!
