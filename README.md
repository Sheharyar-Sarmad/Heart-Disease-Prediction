# 💓 Heart Predictor: AI-Powered Heart Disease Risk Assessment

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://heart-disease-prediction-glkjlmlhtmomn8wrr2j7sd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Groq-FF6600?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)

> **Live Demo:** [heart-disease-prediction-glkjlmlhtmomn8wrr2j7sd.streamlit.app](https://heart-disease-prediction-glkjlmlhtmomn8wrr2j7sd.streamlit.app/)

> **Linkedin Post:** [linkedin.com/in/sheharyar-sarmad-9b7736289/](https://www.linkedin.com/in/sheharyar-sarmad-9b7736289/)
## 📖 About The Project

Heart Predictor is an advanced, AI-powered web application designed to estimate the risk of heart disease based on 11 key clinical features. Built with a **Logistic Regression** machine learning model and integrated with **Groq's Llama 3.3 AI**, the app provides instant, empathetic, and actionable health insights through a stunning, futuristic glassmorphism dashboard.

## ✨ Key Features

*   **⚡ Real-time Predictions:** Instantly calculates the probability of heart disease risk using a highly accurate ML model.
*   **🤖 Groq AI Integration:** Generates compassionate, plain-language health summaries and actionable recommendations based on the prediction results.
*   **💬 AI Chat Assistant:** Ask follow-up questions about your results, lifestyle changes, or general heart health directly within the app.
*   **🎨 Premium Glassmorphism UI:** A neon-themed, fully responsive dashboard with animated gradients, glowing stat cards, and an intuitive user experience.
*   **📊 Interactive Visuals:** Displays risk scores via a beautiful Plotly gauge chart and highlights the contributing risk factors in colorful, interactive chips.
*   **📄 Downloadable Reports:** Easily export the prediction analysis and patient data as a `.txt` file for your records.

## 📊 Model Performance Metrics

The heart of this application is a robust Logistic Regression model trained on a standard cardiovascular dataset. It performs exceptionally well across all metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | `86.96%` |
| **Precision (Weighted Avg)** | `87%` |
| **Recall (Weighted Avg)** | `87%` |
| **F1-Score** | `88.46%` |

*Please note: Predictions are for educational and screening purposes only and do not constitute a medical diagnosis.*

## 🛠️ Built With

*   **Frontend Framework:** [Streamlit](https://streamlit.io/)
*   **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Logistic Regression, Standard Scaler)
*   **Data Processing:** [Pandas](https://pandas.pydata.org/)
*   **Data Visualization:** [Plotly](https://plotly.com/)
*   **Large Language Model (LLM):** [Groq Cloud](https://groq.com/) (Llama 3.3 70b)
*   **Model Serialization:** `joblib`

## 🚀 Getting Started Locally

To get a local copy up and running, follow these simple steps.

### Prerequisites
*   Python 3.8 or higher installed on your machine.
*   A **`GROQ_API_KEY`**. Get yours for free at [console.groq.com](https://console.groq.com/).

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Sheharyar-Sarmad/Heart-Disease-Prediction.git
    cd Heart-Disease-Prediction
