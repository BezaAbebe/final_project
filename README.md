# final_project
Project 4

Traffic Accident Severity Prediction
##Introduction

This project aims to predict the severity of traffic accidents based on factors such as weather, road conditions, and traffic. Accurate predictions can assist law enforcement and emergency services in better preparation, potentially saving lives and resources. The focus of our analysis is on the impact of the traffic accident on the duration of traffic. It can also alert drivers to exercise caution or alter their routes to avoid severe accidents. 


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


Project Structure
This project is divided into four main parts:
1.	Data Cleaning and Visualization
o	Cleaning raw data, handling missing values, and visualizing key patterns and trends.
o	Data Cleaning Notebook
https://colab.research.google.com/drive/1gSpBv1Bet5-BKPIqa42it_991C3GtS_z
3.	Preprocessing and Exploration
o	Preparing data for modeling by scaling, encoding, and exploring relationships between features.
4.	Modeling and Optimization
o	Unsupervised Learning: Applying clustering techniques and Principal Component Analysis (PCA) to understand data distribution and reduce dimensionality.
https://colab.research.google.com/drive/1NXYiR2uQyFVCG0RJ2h96cqmm8X00TzJ2
o	Unsupervised Learning Notebook
o	Supervised Learning: Developing and optimizing a Random Forest model to predict traffic accident severity.
o	Supervised Learning Notebook
(https://colab.research.google.com/drive/1gMwxHGdMVFF3A9OC-ijTERmujPkdWa6_)
Model Initialization and Optimization
The model was initialized using the following code:

