# final_project
Project 4

Traffic Accident Severity Prediction
##Introduction


##Overview
This project focuses on predicting the severity of traffic accidents using machine learning algorithms, with an emphasis on the Random Forest model. The process involves extensive data cleaning, feature engineering, model training, and evaluation using various metrics. Accurate predictions can assist law enforcement and emergency services in better preparation, potentially saving lives and resources. The focus of our analysis is on the impact of the traffic accident on the duration of traffic. It can also alert drivers to exercise caution or alter their routes to avoid severe accidents. 

## Project Structre 

####Data Source:

- The dataset was sourced from Kaggle.com and contains over 7 million rows and 48 columns.
The large dataset is stored in an AWS S3 bucket.
- The large dataset is stored in an AWS S3 bucket.


##Data Cleaning:
- Incomplete, incorrect, inaccurate, or irrelevant data were identified and addressed.
- Missing data imputation approaches were applied to two datasets before merging them into a single dataset.
- Handled numerical and categorical data, followed by feature selection.

##Modeling:
- Multiple machine learning algorithms were explored, with the Random Forest model being a key component of the supervised learning approach.
- The model was evaluated using metrics like Jaccard, F1-Score, Precision, Recall, and Time.

## Files and Directories
- app.py: The main Flask application file located in the root directory.
- HTML_JS/: Contains all HTML and JS files, including:
     - about.html
     - dashboard.html
     - data-cleaning.html
     - supervised.html
- assets/: Static assets like CSS and images, served through custom routes.

##Tools and Technologies
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

## Model Optimization Process
- Parameter Tuning:
   - n_estimators: Tested values: 100, 200, 300, 500, 750, 1000
   - max_depth: Tested values: 10, 15, 20, 35

