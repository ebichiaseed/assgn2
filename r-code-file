##########################
###### Assignment 2 ######
##########################

data <- read.csv("C:/Users/chiask/Downloads/diabetes_5050.csv")

'
Part I Exploring the data set
2. You might check the association between the response and input variables before fitting
any models/classifier. Comment on the strength of the association if possible. This step is
to identify the potential features for the model/classifier.
3. You might consider to keep the ordinal input variables and treat it as quantitative instead
of using them as categorical variables.
4. It is advised to separate the full data set into two parts for training and for testing with
the ratio of 8:2, respectively.
'

dim(data)
head(data)
attach(data)

###############################################################################

#################
##### PLOTS #####
#################


# Observing quantitative variables


boxplot(data$BMI ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'BMI', col = "blue")
boxplot(data$Age ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'Age', col = "blue")
boxplot(data$PhysHlth ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'Physical Health',
        col = "blue")
boxplot(data$MentHlth ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'Mental Health',
        col = "blue")
boxplot(data$Education ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'Education',
        col = "blue")
boxplot(data$Income ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'Income', 
        col = "blue")
boxplot(data$GenHlth ~ data$Diabetes_binary, xlab = 'Diabetes', ylab = 'General Health',
        col = "blue")


# To analyse association between categorical and categorical variables,
# we use odds ratio

##################
### ODDS RATIO ###
##################
#first variable passed = row, second variable passed is column

library(epitools)

# HighBP

tableBP <- table(data$HighBP, data$Diabetes_binary); tableBP
oddsratio(tableBP)

# HighChol

tableHC <- table(data$HighChol, data$Diabetes_binary); tableHC
oddsratio(tableHC)

# CholCheck
tableCC <- table(data$CholCheck, data$Diabetes_binary); tableCC
oddsratio(tableCC)

# Smoker
tableS <- table(data$Smoker, data$Diabetes_binary)
oddsratio(tableS)

# Stroke
tableStroke <- table(data$Stroke, data$Diabetes_binary)
oddsratio(tableStroke)

# HeartDiseaseorAttack

tableHDA <- table(data$HeartDiseaseorAttack, data$Diabetes_binary)
oddsratio(tableHDA)


#PhysActivity
tablePA <- table(data$PhysActivity, data$Diabetes_binary)
oddsratio(tablePA)

# Fruits
tableF <- table(data$Fruits, data$Diabetes_binary)
oddsratio(tableF)

# Veggies
tableV <- table(data$Veggies, data$Diabetes_binary)
oddsratio(tableV)


# HvyAlcoholConsump
tableHAC <- table(data$HvyAlcoholConsump, data$Diabetes_binary)
oddsratio(tableHAC)

# AnyHealthcare
tableAH <- table(data$AnyHealthcare, data$Diabetes_binary)
oddsratio(tableAH)

# NoDocbcCost
tableNDC <- table(data$NoDocbcCost, data$Diabetes_binary)
oddsratio(tableNDC)

# DiffWalk
tableDW <- table(data$DiffWalk, data$Diabetes_binary)
oddsratio(tableDW)

# Sex
tableSex <- table(data$Sex, data$Diabetes_binary)
oddsratio(tableSex)


### observing possible linear relationships ###
# not using linear reg for report
plot(Age, Diabetes_binary, data, pch = 20, col = "darkblue")
cor(Age, Diabetes_binary) #0.2787381
cor(AnyHealthcare, Diabetes_binary) #0.02319075
cor(BMI, Diabetes_binary) #0.2937603
cor(CholCheck, Diabetes_binary) #0.1153816
cor(DiffWalk, Diabetes_binary) #0.272646
cor(Education, Diabetes_binary) # -0.1704806
cor(Fruits, Diabetes_binary) # -0.05407656
cor(GenHlth, Diabetes_binary) #0.4076116
cor(HeartDiseaseorAttack, Diabetes_binary) #0.2115234
cor(HighBP, Diabetes_binary) #0.3815155
cor(HighChol, Diabetes_binary) #0.2892128
cor(HvyAlcoholConsump, Diabetes_binary) # -0.09485314
cor(Income, Diabetes_binary) #-0.2244487
cor(MentHlth, Diabetes_binary)
cor(NoDocbcCost, Diabetes_binary)
cor(PhysActivity, Diabetes_binary)
cor(PhysHlth, Diabetes_binary)
cor(Sex, Diabetes_binary)
cor(Smoker, Diabetes_binary)
cor(Stroke, Diabetes_binary)
cor(Veggies, Diabetes_binary)



# Plotting Linear Regression Model

model <- lm(Diabetes_binary ~ Age + AnyHealthcare + BMI + CholCheck + DiffWalk + Education +
            Fruits + GenHlth + HeartDiseaseorAttack + HighBP + HighChol + HvyAlcoholConsump + 
            Income + MentHlth + NoDocbcCost + PhysActivity + PhysHlth + Sex + Smoker + Stroke + Veggies, data = data)
summary(model)


################################################################################


#########################
### Splitting Data up ###
#########################

smp_size <- floor(0.8 * nrow(data))

set.seed(1101)
randomised_data <- sample(seq_len(nrow(data)), size = smp_size)

train_data <- data[randomised_data, ]
test_data <- data[-randomised_data, ]

################################################################################

'
Part II Building Model/Classifier and Conclusion
5. Propose some models/classifiers.
6. For each model/classifier, examine its goodness of fit.
7. Comparing between models/classifiers fitted, propose the best one (final model)
8. Describe/examine more details about the final model on its goodness of fit. Comments on
its pros and cons if any.
'
#########################
##### DECISION TREE #####
#########################

# Plotting Tree using Train Data
library("rpart")
library("rpart.plot")
fit <- rpart(Diabetes_binary ~ Age + AnyHealthcare + BMI + CholCheck + DiffWalk +
              GenHlth + HeartDiseaseorAttack + HighBP + HighChol + 
              Income + MentHlth + NoDocbcCost + PhysHlth + Sex + Smoker + Stroke,
             method="class",
             data=train_data, 
             control=rpart.control(minsplit = 5, maxdepth = 30),
             parms=list(split='gini')
)


rpart.plot(fit, type=4, extra=4, varlen=0, faclen=0, clip.right.labs=FALSE)

# Predicting using Test Data
predictions <- predict(fit, test_data, type = 'class')

confusion_matrix <- table(predictions, test_data$Diabetes_binary); confusion_matrix

TP <- confusion_matrix[2,2];TP
TN <- confusion_matrix[1, 1]; TN
FP <- confusion_matrix[2, 1]; FP
FN <- confusion_matrix[1, 2]; FN

TPR <- TP / (TP+FN); TPR
FPR <- FP / (FP + TN); FPR
FNR <- FN / (TP +FN); FNR

accuracy <- (TP + TN) / (TP + TN + FP + FN); accuracy
precision <- TP / (TP + FP); precision

################################
##### k-Nearest Neighbours #####
################################


# form a SET OF FEATURES for the training; and for testing:

train.x = train_data[,c('HighBP', 'HighChol', 'CholCheck', 
                        'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                        'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex',
                        'BMI', 'PhysHlth', 'MentHlth', 'Age', 'Income', 
                        'GenHlth')] ## accessing the vectors of columns
test.x = test_data[,c('HighBP', 'HighChol', 'CholCheck', 
                      'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                      'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex',
                      'BMI', 'PhysHlth', 'MentHlth', 'Age', 'Income', 
                      'GenHlth')]


# form the RESPONSE for the traning; and for testing:
train.y = train_data[,c("Diabetes_binary")]
test.y = test_data[,c("Diabetes_binary")]

# choosing a suitable value of k using common formula

n = dim(train.x)
k = sqrt(n); k

# we will be using k = 237

# Forming classifier

library(
  class)

set.seed(1101)

knn.pred = knn(train.x,test.x,train.y,k=20)

confusion.matrix=table(knn.pred, test.y)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix) #0.7308862




########################## N-FOLD CROSS VALIDATION 


X= data[,c('HighBP', 'HighChol', 'CholCheck', 
                    'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                    'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex',
                    'BMI', 'PhysHlth', 'MentHlth', 'Age', 'Income', 
                    'GenHlth')]
Y= data[,c("Diabetes_binary")]

dim(data)


n_folds=20

folds_j <- sample(rep(1:n_folds, length.out = dim(filtered_data)[1] )) 

table(folds_j)

err=numeric(n_folds)
acc=numeric(n_folds)

for (j in 1:n_folds) {
  test_j <- which(folds_j == j) # get the index of the points that will be in the test set
  pred <- knn(train=X[ -test_j, ], test=X[test_j, ], cl=Y[-test_j ], k=1) # KNN with k = 1, 5, 10, etc
  
  err[j]=mean(Y[test_j] != pred)
  acc[j]=mean(Y[test_j] == pred) 
  # this acc[j] = sum(diag(confusion.matrix))/sum(confusion.matrix), where confusion.matrix=table(Y[test_j],pred)
  
}
err
acc
error=mean(err);error
accuracy = mean(acc); accuracy


###########################
### LOGISTIC REGRESSION ###
###########################

attach(data)


M1<- glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + 
         Smoker + Stroke + HeartDiseaseorAttack +
         AnyHealthcare + NoDocbcCost + DiffWalk + Sex,
         data = train_data,family = binomial)
summary(M1)

'
 Estimate Std. Error z value Pr(>|z|)    
(Intercept)          -2.94243    0.08570 -34.336  < 2e-16 ***
HighBP                1.20260    0.01777  67.691  < 2e-16 ***
HighChol              0.73579    0.01749  42.060  < 2e-16 ***
CholCheck             1.43043    0.07640  18.723  < 2e-16 ***
Smoker                0.06805    0.01736   3.919 8.90e-05 ***
Stroke                0.31603    0.03965   7.970 1.59e-15 ***
HeartDiseaseorAttack  0.57616    0.02703  21.319  < 2e-16 ***
AnyHealthcare         0.02407    0.04333   0.555 0.578574    
NoDocbcCost           0.10639    0.03101   3.430 0.000603 ***
DiffWalk              0.91578    0.02137  42.847  < 2e-16 ***
Sex                   0.18406    0.01743  10.557  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 98000  on 70691  degrees of freedom
Residual deviance: 81050  on 70681  degrees of freedom
AIC: 81072

Number of Fisher Scoring iterations: 4

'

# will remove AnyHealthcare because not significant
# hence creating new model M2

M2<- glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + 
          Smoker + Stroke + HeartDiseaseorAttack + PhysActivity + Fruits +
          NoDocbcCost + DiffWalk + Sex + Veggies + HvyAlcoholConsump,
         data = train_data,family = binomial)
summary(M2)


features = data[,c('HighBP', 'HighChol', 'CholCheck', 
                    'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                   'Fruits', 'Veggies', 'HvyAlcoholConsump',
                    'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex')]
head(features)
response= data[,c("Diabetes_binary")]

# model for testing and checking
M3<- glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + 
           Smoker + Stroke + HeartDiseaseorAttack +
           NoDocbcCost + DiffWalk + Sex,
         data = features,family = binomial)

### ROC CURVE FOR LOGISTIC MODEL ###
# To assess goodness of fit of log model

library(ROCR)
prob = predict(M3, type ="response")
# 1. Predict

prob[1:20]


train_data[1:20,]


pred = prediction(prob, Diabetes_binary)
roc = performance(pred , "tpr", "fpr")
auc = performance(pred , measure ="auc")
auc@y.values[[1]]
plot(roc, col = "red", main = paste(" Area under the curve :", round(auc@y.values[[1]] ,4)))
# The commented parts are optional, but it helps with readability


alpha <- round (as.numeric(unlist(roc@alpha.values)) ,4)
length(alpha) 


fpr <- round(as.numeric(unlist(roc@x.values)) ,4)
tpr <- round(as.numeric(unlist(roc@y.values)) ,4)

x = cbind(alpha, tpr, fpr)
x



par(mar = c(5 ,5 ,2 ,5))

plot(alpha ,tpr , xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "blue")


par( new ="True")
plot(alpha ,fpr , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "red" )
axis( side =4) # to create an axis at the 4th side
mtext(side =4, line =3, "False positive rate")
text(0.18 ,0.18 , "FPR")
text(0.58 ,0.58 , "TPR")

