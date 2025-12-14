# Exoplanet Predictive Analysis using Machine Learning

This project applies machine learning techniques to analyze and classify exoplanets using real astronomical data from the NASA Exoplanet Archive.

## Objective
- Classify exoplanets based on physical characteristics
- Identify hidden patterns using clustering
- Compare performance of multiple machine learning models

## Dataset
- Source: NASA Exoplanet Archive
- Records: Confirmed exoplanets
- Key Features:
  - Planet Radius (pl_rade)
  - Planet Mass (pl_bmasse)
  - Orbital Period
  - Stellar Temperature

## Machine Learning Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

## Methodology
1. Data cleaning and preprocessing
2. Feature scaling using StandardScaler
3. Binary classification target creation
4. Model training and evaluation
5. Feature distribution and correlation analysis
6. Clustering using K-Means

## Results
- Random Forest achieved the highest accuracy
- Feature correlation analysis revealed strong relationships between stellar and planetary attributes
- Ensemble models outperformed traditional classifiers

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
```bash
pip install -r requirements.txt
python src/exoplanet_predictive_analysis.py


