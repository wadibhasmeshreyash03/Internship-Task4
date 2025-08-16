Name : Shreyash Rajendra Wadibhasme 
Company : Codtech IT solutions
Id : CT08DG1782 
Domain : Python programming 
Duration : june to august 2025
Mentor : Neela Santosh Kumar

## **Project Overview: SMS Spam Detection using Machine Learning**
![taskpho1](https://github.com/user-attachments/assets/995af010-307b-4760-9f70-19a4b3db83cc)

### **Objective**

The project builds a **spam detection system** that classifies SMS messages as **spam** or **ham (not spam)** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

### **Key Components**

1. **Data Loading**
   * Dataset: **SMS Spam Collection** (TSV file from GitHub).
   * Columns:
     * `label` → ("ham" or "spam")
     * `message` → SMS text content

2. **Data Preprocessing**
   * Labels converted into numeric form:
     * `ham = 0`
     * `spam = 1`
   * Text data transformed into **TF-IDF features** using `TfidfVectorizer` (removes stopwords, extracts important terms).

3. **Model Training**
   * Train-test split: **80% training, 20% testing**.
   * Algorithm: **Multinomial Naïve Bayes (NB)** – efficient for text classification.
   * Model learns word distributions to distinguish spam from ham.

4. **Model Evaluation**
   * Metrics:
     * **Accuracy**
     * **Classification Report** (Precision, Recall, F1-score)
     * **Confusion Matrix**
   * Visualization: Heatmap of confusion matrix with Seaborn.

5. **Model Saving**
   * Trained **Naive Bayes model** saved as `spam_detector_model.pkl`.
   * **TF-IDF vectorizer** saved as `tfidf_vectorizer.pkl`.
   * Enables re-use without retraining.

### **Technologies Used**

* **Python**
* **Pandas, NumPy** → Data handling
* **Matplotlib, Seaborn** → Visualization
* **Scikit-learn** → ML (TF-IDF, Naïve Bayes, Evaluation metrics)
* **Joblib** → Model persistence

### **Applications**

* Detecting **spam SMS and emails**.
* Can be extended into:
  * Real-time spam filter for messaging apps.
  * Email filtering system.
  * Integration into chatbot or customer service platforms.
