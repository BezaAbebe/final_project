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

  
Data Storage and Management:
- AWS S3: For storing large datasets used for training and validation.



Project Structure
This project is divided into four main parts:
1.	Data Cleaning and Visualization
o	Cleaning raw data, handling missing values, and visualizing key patterns and trends.
o	Data Cleaning Notebook
2.	Preprocessing and Exploration
o	Preparing data for modeling by scaling, encoding, and exploring relationships between features.
3.	Modeling and Optimization
o	Unsupervised Learning: Applying clustering techniques and Principal Component Analysis (PCA) to understand data distribution and reduce dimensionality.
o	Unsupervised Learning Notebook
o	Supervised Learning: Developing and optimizing a Random Forest model to predict traffic accident severity.
o	Supervised Learning Notebook
Model Initialization and Optimization
The model was initialized using the following code:
![image](https://github.com/user-attachments/assets/9f64f03d-ee74-4de4-9f65-24430f7aa856)
