# 🛒 Retail Intelligence: Sales Forecasting & Performance Analytics

This is an **end-to-end** Data Science project that transforms raw supermarket sales data into strategic insights. The project ranges from data processing and Machine Learning modeling to the creation of an interactive Dashboard containerized with Docker.

---

## 📖 Project Context

In a competitive retail market, understanding customer behavior and predicting revenue is essential. This project analyzes historical sales records to:

1. **Cleanse and standardize** transactional data.
2. **Segment customers** using unsupervised learning (K-Means Clustering).
3. **Predict total revenue** using a Random Forest Regressor.

---

## 📊 Strategic KPIs (Dashboard)

The dashboard monitors three key performance indicators:

* **Total Revenue (Sales):** Sum of all sales, indicating global growth.
* **Average Rating:** Measurement of customer satisfaction by branch and category.
* **Cluster Segmentation:** Identification of sales groups based on consumption and performance profile.

---

## 🛠️ Technologies Used

* **Language:** Python 3.9
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means)
* **Preview:** Plotly Express, Streamlit
* **Containerization:** Docker (Based on Python-Slim)

---

## 📂 Folder Structure

```text
├── app.py # Streamlit Application
├── Dockerfile # Container Configuration
├── requirements.txt # Project dependencies
├── notebooks/ # Cleaning and modeling pipeline
├── data/processed/ # Data ready for BI and Training
└── models/ # Saved .pkl models

```

---

## 🚀 How to Execute

### 1. Using Docker (Recommended)

Make sure Docker Desktop is running and run:

```bash
# Build the image
docker build -t sales-app .

# Run the container
docker run -p 8501:8501 sales-app

```

Access at: `http://localhost:8501`

### 2. Local Installation

1. Create a virtual environment: `python -m venv .venv`
2. Activate the environment and install the dependencies:
```bash
pip install -r requirements.txt

```


3. Run the App:
```bash
streamlit run app.py

```



---

## 🤖 Modeling and Intelligence

The project uses **K-Means** to group sales into clusters, allowing the marketing team to identify areas of high revenue vs. low satisfaction. Additionally, the **Random Forest** model was trained to predict `Sales` based on variables such as branch, customer type and product line.

Developed by **Ricson Ramos**.
