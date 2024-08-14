# final_project
Project 4

Traffic Accident Severity Prediction
##Introduction


## Overview


This project focuses on predicting the severity of traffic accidents using machine learning algorithms, with an emphasis on the Random Forest model. The process involves extensive data cleaning, feature engineering, model training, and evaluation using various metrics. Accurate predictions can assist law enforcement and emergency services in better preparation, potentially saving lives and resources. The focus of our analysis is on the impact of the traffic accident on the duration of traffic. It can also alert drivers to exercise caution or alter their routes to avoid severe accidents. 

## Project Structre 

#### Data Source:

- The dataset was sourced from Kaggle.com and contains over 7 million rows and 48 columns.
The large dataset is stored in an AWS S3 bucket.
- The large dataset is stored in an AWS S3 bucket.


## Data Cleaning:
- Incomplete, incorrect, inaccurate, or irrelevant data were identified and addressed.
- Missing data imputation approaches were applied to two datasets before merging them into a single dataset.
- Handled numerical and categorical data, followed by feature selection.

## Modeling:
- Multiple machine learning algorithms were explored, with the Random Forest model being a key component of the supervised learning approach.
- The model was evaluated using metrics like Jaccard, F1-Score, Precision, Recall, and Time.

## Files and Directories
- app.py: The main Flask application file located in the root directory.
- HTML_JS/: Contains all HTML and JS files, including:
     - assets: Static assets like CSS and images, served through custom routes.
            - CSS
            - images
     - about.html
     - data_cleaning.html
     - supervised.html
     - unsupervised.html
- template: contains htmls
       - index.html
       - model_predict.html

## Tools and Technologies
1. Python Libraries:

 - Scikit-learn: For both Random Forest and K-means clustering implementations.
 - Pandas: For data manipulation and preprocessing.
 - NumPy: For numerical operations.
 - Matplotlib/hvplot/holoviews: For data visualization.
   
2. Machine Learning Platforms:
     - AWS S3: storing large datasets and model.
     - Google Colab: For running Python code in a cloud environment as our datasets were large.

  
3. Data Storage and Management:
     - AWS S3: For storing large datasets used for training and validation.
4. Web Development and Presentation:
     - HTML
     - Bootstrap: For responsive web design.
     - Bootstrap API: api.bootstrap.com
--

## Project Structure
This project is divided into four main parts:
1.	Data Cleaning and Visualization
   - Tasks: Cleaning raw data, handling missing values, and visualizing key patterns and trends.
   - Data Cleaning Notebook: Due to large size, was not able to import the notebook in github
        - https://colab.research.google.com/drive/1gSpBv1Bet5-BKPIqa42it_991C3GtS_z
2.	Preprocessing and Exploration
    - Preparing data for modeling by scaling, encoding, and exploring relationships between features.
3.	Modeling and Optimization
    - Unsupervised Learning: Applying clustering techniques and Principal Component Analysis (PCA) to understand           data distribution and reduce dimensionality.
    - Unsupervised Learning Notebook: Due to large size, was not able to import the notebook in github
           - https://colab.research.google.com/drive/1NXYiR2uQyFVCG0RJ2h96cqmm8X00TzJ2

     - Supervised Learning: Developing and optimizing a Random Forest model to predict traffic accident severity.
     - Supervised Learning Notebook: due to the size of the notebook, was not able to import the notebook in github
             - (https://colab.research.google.com/drive/1gMwxHGdMVFF3A9OC-ijTERmujPkdWa6_)
## Model Initialization and Optimization
 - Data Preparation: Separated features from labels and split data into training and testing sets.
 - odel Training: Fitted a Random Forest model using RandomForestClassifier from Scikit-learn.
 - Prediction and Evaluation: Evaluated the model's performance using metrics such as accuracy, precision, and recall.

--
# Unsupervised Learning
     - Applying clustering techniques and Principal Component Analysis (PCA) to understand data distribution and reduce dimensionality.
     - KMeans was utilized to find an appropriate value for K, which was k = 3. 
	PCA was run for for 3 Principal Components, with PCA1, attributing almost 100% of the 	
	Variance in the dataset. 
      - Further analysis to identify top 15 features related to PCA was performed



--

# Supervised Learning (Random Forest)

## Model Optimization Process
- Parameter Tuning:
   - n_estimators: Tested values: 100, 200, 300, 500, 750, 1000
   - max_depth: Tested values: 10, 15, 20, 35
     
## Results

- Classification Report
     - Accuracy: 79%
     - Precision for Non-severe: 92%
     - Precision for Severe: 35%
     - Recall for Severe: 55%
 
Overall, the model is strong in predicting "Non-severe" cases but less effective for "Severe" cases. This indicates potential risks in misclassifying severe accidents, either by failing to predict them or overestimating their likelihood.

## Recommendations

Further efforts could include experimenting with different hyperparameters or balancing techniques to improve the model's ability to accurately predict severe cases. Continued fine-tuning will be important as the model is applied to real-world data to ensure it performs consistently well across all classes.











