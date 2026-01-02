# Alzheimers-Prediction
Alzheimer’s Disease Prediction using clinical, behavioral, and lifestyle features. Includes exploratory data analysis, dimensionality reduction with PCA, classical ML models (Logistic Regression, Random Forest), and a Neural Network classifier with ROC-AUC evaluation. 
---

## Background

Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory, cognition, and behavior, ultimately impairing daily living activities. Early prediction and diagnosis are critical for clinical intervention and patient care. While neuroimaging provides valuable biomarkers, behavioral and clinical features are more accessible and can complement predictive models. Machine learning offers the ability to integrate these multidimensional features to classify individuals at risk of AD, and potentially aiding early detection.

---

## Objectives

The primary objectives of this project are:

1. To perform an exploratory analysis of clinical, behavioral, and lifestyle features associated with Alzheimer’s disease.
2. To reduce feature dimensionality using Principal Component Analysis (PCA) for visualization and improved model performance.
3. To develop predictive models using classical machine learning and neural networks.
4. To evaluate model performance with confusion matrices, classification metrics, and ROC-AUC scores.
5. To identify the most predictive features for Alzheimer’s classification.

---

## Dataset

The dataset includes anonymized patient records with 35 columns representing:

- **Demographics**: Age, Gender, Ethnicity, Education Level  
- **Lifestyle factors**: BMI, Smoking, Alcohol Consumption, Physical Activity, Diet Quality  
- **Cognitive and behavioral features**: Memory Complaints, Behavioral Problems, ADL (Activities of Daily Living), Confusion, Disorientation, Personality Changes, Difficulty Completing Tasks, Forgetfulness  
- **Target variable**: `Diagnosis` (0 = No AD, 1 = AD)

> Note: All data are anonymized and used strictly for educational purposes.  

---

## Exploratory Data Analysis (EDA)

EDA was conducted to identify patterns and relationships among features:

- **Categorical features** (e.g., Family History of AD, Gender) were analyzed using count plots  
- **Numerical features** (e.g., Age, BMI) were visualized using histograms, boxplots, and density plots  
- **Correlation analysis** revealed strong associations between cognitive deficits, memory complaints, and AD diagnosis  

Example visualization:  
![Memory Complaints Distribution](figures/memory_complaints.png)

Insights:

- Higher prevalence of memory complaints, ADL impairments, and confusion in patients diagnosed with AD  
- Some lifestyle factors, such as physical activity and diet quality, showed modest correlations with diagnosis

---

## Feature Engineering and PCA

- Features were **standardized** to have zero mean and unit variance using `StandardScaler`  
- **Principal Component Analysis (PCA)** was applied to reduce dimensionality while retaining 99% of variance  
- PCA allowed **visual exploration** of patient separation in two-dimensional space:

![PCA Plot](figures/pca_plot.png)

> Interpretation: Alzheimer’s patients cluster separately from cognitively normal patients, suggesting that the feature set effectively captures cognitive and behavioral differences.

---

## Model Development

Three types of models were trained to classify patients:

### Logistic Regression
- Interpretable baseline model
- Allows understanding of **feature coefficients** for contribution to AD risk

### Random Forest
- Tree-based ensemble model  
- Captures non-linear interactions between features  
- Provides feature importance ranking for feature interpretability  

### Neural Network
- Feedforward architecture: 8 → 4 → 2 → 1 neurons  
- ReLU activations for hidden layers, Sigmoid activation for output layer  
- Trained with early stopping to prevent overfitting  
- Captures **complex non-linear relationships** in the data

```python
model_ANN = keras.Sequential([
    keras.layers.Dense(8, input_shape=(8,), activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
