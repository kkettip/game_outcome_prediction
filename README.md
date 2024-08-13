# game_outcome_prediction
 
Aim: 
To predict SBU football game outcome using athletics’ physical metrics and past SBU football game outcome data

Data information:
A csv file containing SBU football athletics’ physical movement metrics and game outcome 
File name: AI_ML Project (For temp ML).csv

Target variable: game outcome

Columns: 'Metrics', 'Metric Value', 'Metric Date', 'SBUID', 'Game Date', 'Opponent', 'Game Outcome'
Metrics: accel_load_accum, distance_total, metabolic_power_avg, speed_avg,  physio_load, physio_intensity, metabolic_work, accel_load_accum, distance_total, metabolic_power_avg

Approach:
Used autoML from ML Jar to generate prediction model

Used ML Jar’s Income Classification Example as a starting point to generate the game outcome prediction model 

Link to example:
https://github.com/mljar/mljar-examples/blob/master/Income_classification/Income_classification.ipynb

Steps:

1. EDA:  Drop rows with NaN and duplicates. Also, convert Metric Dates and Game Date to the same date format.

2. Define X and Y using the following code:

Model 1:  All variables are used for the X value except for game outcome, while game outcome is used for the y value.

# defining X and Y
X = df[df.columns[:-1]] # Include all columns except the last one that is game outcome
y = df["Game Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Split the dataframe df into X and y


Model 2: Variables used for the x value does not include 'Game Outcome', 'SBUID', 'Game Date', 'Metric Date', 'Opponent'

# defining X and Y
X = df.drop(['Game Outcome', 'SBUID', 'Game Date', 'Metric Date', 'Opponent'], axis=1)
y = df['Game Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


3. Split data into train and test sets.  

4. Train the model 

5. Run the model on the test set to make a prediction

6. Evaluate the model’s performance


Findings: 

Different combinations of variables for X have an impact on the model’s performance. 

Model 1:
After training the model 1, autoML suggested that the best model would be Default RandomForest.

Model Evaluation:
Accuracy is a ratio of correctly predicted observation to the total observations.

Accuracy of 1, means that every prediction is correct.  This could be due to a small training data set used to train the model.  


AutoML best model: 6_Default_RandomForest
Accuracy: 1.00
Classification Report:
              precision    recall  f1-score   support

     SBU Win       1.00      1.00      1.00        32
Opponent Win       1.00      1.00      1.00        11

    accuracy                           1.00        43
   macro avg       1.00      1.00      1.00        43
weighted avg       1.00      1.00      1.00        43



Model 2:
After training the model 2, autoML suggested that the best model would be Ensemble

Accuracy of 0.74, which means that correct predictions are made 74% of the time.


AutoML best model: Ensemble
Accuracy: 0.74
Classification Report:
              precision    recall  f1-score   support

     SBU Win       0.74      0.97      0.84        30
Opponent Win       0.75      0.23      0.35        13

    accuracy                           0.74        43
   macro avg       0.75      0.60      0.60        43
weighted avg       0.75      0.74      0.69        43



Future work:
1. To better predict SBU football game outcomes, a larger data set should be used to train the model. The training data should also not be skewed towards losses or wins.  

2. Determine which combination of athletics’ physical metrics would be best to include into the dataset for model training to predict game outcome. This could be accomplished by including different combinations of athletics’ physical metrics into the dataset and then evaluating the model’s performance, which would ensure that the most relevant data is collected for training the model.

3. Predicting the number of points lost by the SBU football team when compared to their opponents per game. With this knowledge the football team can develop strategies to maximize the number of points gained per game.
