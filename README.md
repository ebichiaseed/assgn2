# dsa1101-assgn2
## note: please don't plagiarise my code ;-; it's for portfolio purposes thanks!
## About this repository

This repository contains an assignment that I did and scored full marks for in DSA1101, AY23/24.

## About the assignment

The assignment tasked us to predict the diabetic status (whether one is diabetic or not) of 70,692 survey respondents in the US for the year 2015. I analysed the provided dataset in terms of the features and response variables, proposed three possible classifiers (Decision tree, k-Nearest Neighbours and Logistic Regression) to predict the outcome of whether one has diabetes or not, and finally, conclude by selecting the most suitable classifier in this context by looking at the goodness-of-fit of models and the pros and cons of each model.

##*Dataset Information*

Dimensions | 70 692 observations and 22 variables
--- | --- 
Response variable | ‘Diabetes_binary’, categorical variable with 2 levels, 0 and 1
--- | --- 
Nominal Feature Variables | HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
--- | --- 
Quantitative Feature Variables | BMI, PhysHlth, MentHlth, Age, Education*, Income*, GenHlth*

*: denotes that these variables are ordinal variables. However, for simplicity, this report will treat them as quantitative variables instead of categorical ones, as ordinal variables have clear order ranking.
Nominal variables are unordered categorical variables. In this dataset, they are binary with levels of ‘0’ and ‘1’.

##*Checking the Association between Response and Input Variables*

The underlying assumption made is that all input variables are independent of each other (i.e no effect of one input variable on another)

Quantitative Variables: To visualise the association between quantitative features and the binary categorical response variable, boxplots were used. In this case, Education does not seem to have any association with whether one has diabetes or not, as the Education levels of both diabetic and non-diabetic people have the same distribution. Hence, we will not be using Education as a feature in our analysis.

Categorical Variables: To analyse the association between categorical features and the categorical response variable, odds ratio (OR) was used. OR helps to analyse the strength of association between inputs and response variables, and check the significance of a feature in affecting the response variable. The ‘epitools’ package was imported for the use of ‘oddsratio()’ function to calculate OR.

Odds Ratio (OR) = Odds of Event A occurringOdds of Event B occurring
If OR < 1, there is negative association between the feature and response variable. If OR > 1, the larger the magnitude of OR, the stronger the association between the feature and response variable. If OR = 1, there is no association between the feature and response variable.

Indicators | Odds Ratio
--- | --- 
HighBP | 5.088
--- | --- 
HighChol | 3.296
--- | --- 
CholCheck | 6.487
--- | --- 
Smoker | 1.412
--- | --- 
Stroke | 3.092
--- | --- 
HeartDiseaseorAttack | 3.656
--- | --- 
PhysActivity | 0.491
--- | --- 
Fruits | 0.801
--- | --- 
Veggies | 0.676
--- | --- 
HvyAlcoholConsump | 0.365
--- | --- 
AnyHealthcare | 1.252
--- | --- 
NoDocbcCost | 1.326
--- | --- 
DiffWalk | 3.807
--- | --- 
Sex | 1.195

##*Splitting of Data for Training and Testing*
The data has been split into 80:20 ratio, for training and testing respectively, as recommended. Additionally, set.seed(1101) was used for standardisation purposes.
Training Dataset | Testing Dataset
--- | --- 
56553 observations of 20 variables | 14139 observations of 20 variables


##Models and Statistical Findings
Classifiers are machine learning models that are trained to categorise features (input variables) into outcome classes. There are two types of classifiers: Supervised and Unsupervised Learning Methods.
Supervised Learning Methods are trained on datasets that have both response and feature variables and are then used to make predictions on new, unseen data. These include Decision Trees, Naive Bayes, k-Nearest Neighbours (KNN) algorithm, Linear Regression and Logistical Regression. Unsupervised Learning Methods handle unclassified data and draw patterns between data points to predict an outcome. These include k-Means Clustering and Association Rules.
In this case, since the response variable was provided, Supervised Learning Methods will be used to classify data points and make predictions. Additionally, as our response and most of our features are categorical, it would not be possible to use models such as linear regression that handle continuous data to predict outcomes.

###Model I: Decision Trees
Rationale for Model
- A Decision Tree is a classification method that uses a tree structure to specify sequences of decisions and consequences. This report will be utilising a Classification Tree that handles categorical outputs that are binary, which is perfect for our ‘Diabetes_binary’ response variable.
Data Preparation
- Libraries used: rpart, rpart.plot
- The ‘Training’ dataset was used to plot the Classification Tree, and the ‘Testing’ dataset was used to predict whether one is diabetic/ pre-diabetic based on a certain set of features. In this case, setting minsplit = 5 (smallest number of observations allowed in a terminal/leaf node), maxdepth = 20 (maximum depth the tree can go is 20) and parms = ‘gini’ (utilising gini index as our parameters), our classification tree has narrowed down the important features to be: HighBP, GenHlth, Age and BMI.
Assessing Goodness of Fit
- To assess the goodness-of-fit of our model, indicators such as True Positive Rate (TPR), False Negative Rate (FNR), False Positive Rate (FPR), accuracy and precision were used on our predictions that relied on our ‘Testing’ dataset. 

Indicator | Value
--- | ---
True Positive Rate | 0.792
--- | ---
False Positive Rate | 0.338
--- | ---
False Negative Rate | 0.208
--- | ---
Accuracy | 0.727
--- | ---
Precision | 0.703

Overall, the Classification Tree is relatively accurate and precise. It is relatively safe for classifying and predicting outcomes of whether one has diabetes or not, as seen by the high proportion of true positives and low proportion of false positives and negatives. This means that it is unlikely that one would be misdiagnosed based on their given set of features.
Limitations of Model
Decision Trees are prone to overfitting and may unintentionally capture specific insignificant details of the data that may influence outcomes of prediction, resulting in skewed data.

###Model II: k-Nearest Neighbours (KNN)
Rationale for Model
- KNN is an algorithm that works by looking at the neighbours of a data point. By observing the features of the k-nearest points (where k is an integer), the algorithm assigns the most common class among its k nearest neighbours to the data point of concern. Due to its ability to classify binary and/or multi-class datasets, KNN was used to classify and predict the outcome of whether one is diabetic or not based on a certain set of features. 

Data Preparation
- Libraries used: class
- Both the Training and Testing datasets were used to form the set of features and response for the algorithm’s training and testing purposes. In this case, we will be using all the features of the refined dataset as shown in Figure 4 to form our Training Dataset and Testing Dataset.
- In theory, in order to choose a suitable value of k, the square root of the number of observations in the training dataset was taken. However, due to our dataset being numerically large, it would be computationally inefficient to run the algorithm at a high value of k obtained through this method. Hence, a standard value of k= 20 was used to make the code run more smoothly and quickly.
- Based on k = 20, the probability of an individual who is diabetic or pre-diabetic is 0.731 (3sf).

Assessing Goodness of Fit
- To assess goodness-of-fit of the algorithm, n-fold cross validation was used, where n = 20. This assessed the probabilities of our predictions being erroneous, as well as how accurate the predictions are for this value of k.

Indicator | Value (3sf)
--- | ---
Error Rate | 0.328
---  ---
Accuracy Rate | 0.672


Overall, the relatively low error rate and high accuracy rate in prediction of outcome using this algorithm shows that it is a reliable model for predicting whether one has diabetes or not.

Limitations of Model
- Scaling of the model is required, as KNN relies on the euclidean distance between the datapoint of concern and its neighbouring points. This is not possible for our large proportion of categorical or ordinal features in our dataset, which may affect the outcome of prediction.

###Model III: Logistic Regression

Rationale for Model
- Logistic Regression is a generalised linear model that models the logarithm of the response variable. It shows the relationship between the features and response variable through coefficients, which is the change in the log-odds of the response with a one-unit change in the feature.

Data Preparation
- For the initial model, we will only be using categorical features with 2 variables (HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, AnyHealthcare, NoDocbcCost, DiffWalk, Sex), as our generalised linear model is under the family ‘binomial’, where the values of features can only be in the range of 0 to 1. 
- After running the model once, we observe the p-value of each feature to test its significance in relation to the response variable. As the feature ‘AnyHealthcare’ has a p-value of more than 0.5 (the standard value for testing significance), we will disregard it in the second iteration of the model so as to refine it further.


Goodness of Fit
- Library used: ROCR
- To assess goodness-of-fit of the model, we will be using the ROC curve to obtain the area under the curve (AUC). We will be utilising our Training Dataset with the class labels 
- The larger the magnitude of AUC, the better the prediction of the final class of whether one is diabetic or not. An AUC score of 1 means the classifier can perfectly distinguish between all class points labelled ‘0’ and ‘1’.
- Based on our algorithm, area under ROC Curve = 0.771 (3sf). This means that our algorithm is able to predict the outcome of whether one has diabetes or not relatively well.

Limitations of Model
- This model assumes a linear regression. However, most of our features are categorical and there is very weak linear correlation between features and response, which can influence outcome greatly.

Conclusion
- To obtain the final model, we would be quantitatively comparing the goodness of fit of all the models proposed.

Model | Indicator | Goodness of Fit
--- | --- | ---
Decision Tree | Accuracy Rate | 0.727
--- | --- | ---
k-Nearest Neighbours | Accuracy Rate | 0.672
--- | --- | ---
Logistic Regression | AUC | 0.771


By quantitative comparison, Logistic Regression is the most accurate in predicting the class of outcome; whether one has diabetes or not. However, regression models like the logistic regression model are not very suitable as most of the categorical features are not linearly related to the response. Hence, by evaluation of pros and cons of each model, the most appropriate model to be used is a Decision Tree, as it is able to classify based on each feature accordingly, which is good for multi-feature datasets, and especially in the context of diagnosing diabetes, as diagnosis relies on a number of features. Additionally, the limitations of Decision Trees can be combated by utilising other algorithms to prevent overfitting.



