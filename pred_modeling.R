########################################
# Assignment II - Predictive Modelling #
########################################

#####
#LIBRARIES
#####
library(tidyverse)
library(lubridate)
library(ggplot2)
library(caret)
library(caTools)
library(dplyr)
library(MASS)
library(randomForest)
library(parallel)
library(ROCR)
library(reshape2)
library(doParallel)
library(ROSE)
library(glmnet)
library(randomForest)
library(cowplot)
library(gridExtra)
registerDoParallel(cores = 2)

#####
# DATA DOWNLOAD
#####
setwd("C:/Users/user/Desktop/Assign2_big_data")

telco.train = read.csv("telco_train.csv", header = T)
telco.TEST = read.csv("telco_test.csv", header =  T)

telco.train$START_DATE = ymd(telco.train$START_DATE)
telco.TEST$START_DATE = ymd(telco.TEST$START_DATE)


#####
# DATA EXPLORATION AND CLEANING
#####

str(telco.train)
summary(telco.train)
#View(telco.train)
prop.table(table(telco.train$CHURN))
table(telco.train$CHURN)
summary(telco.train$FIN_STATE)
summary(telco.train$AVG_DATA_1MONTH)
summary(telco.train$AVG_DATA_3MONTH)
summary(telco.train$COUNT_CONNECTIONS_3MONTH)

# some issues
# NA's in FIN_STATE more or less 70% (3623) of the cases
# AVG_DATA_3MONTH
# COUNT_CONNECTIONS_3MONTH
# AVG_DATA_1MONTH more or less 25% (1343)

ind1 <- which(is.na(telco.train$AVG_DATA_1MONTH))
ind2 <- which(is.na(telco.train$AVG_DATA_3MONTH))
ind3 <- which(is.na(telco.train$COUNT_CONNECTIONS_3MONTH))
table(ind1==ind2)
table(ind2==ind3)
# no data subscrition for these 1343 people, only calls
# we can consider the NA cases as 0, no data consumption
telco.train$AVG_DATA_3MONTH <- replace_na(telco.train$AVG_DATA_3MONTH, 0)
telco.train$AVG_DATA_1MONTH <- replace_na(telco.train$AVG_DATA_1MONTH, 0)
telco.train$COUNT_CONNECTIONS_3MONTH <- replace_na(telco.train$COUNT_CONNECTIONS_3MONTH, 0)

# we will not include FIN_STATE in the analysis for the moment

telco.train <- dplyr::select(telco.train, -c(FIN_STATE, ID))

summary(telco.train)


#####
# DATA PARTITIONING
#####

# Partition into training and validation
set.seed(9)
tvt_split= function(dataset, probs=c(0.60,0.2,0.2)){
  g=sample(cut(
    seq(nrow(dataset)), 
    nrow(dataset)*cumsum(c(0, probs)), 
    labels=c('train', 'validation','test')
  ))
  split(dataset, g)
}

telco <- tvt_split(telco.train)

# check the number of churn cases
table(telco$train$CHURN)
table(telco$validation$CHURN)
table(telco$test$CHURN)



#######
# MODELLING
#######

### RANDOM FOREST
rf_model <- train(as.factor(CHURN)~., data=telco$train, method="rf",
                  trControl=trainControl(method="cv",number=5),
                  ntree=80)
print(rf_model)
print(rf_model$finalModel)
str(rf_model)
# ROC and AUC curve on train
#pred.train <- predict(rf_model, telco$train[,-1], type="prob")
#pred.train2 <- prediction(as.numeric(rf_model$finalModel$predicted), telco$train$CHURN)
#perf.train <- performance(pred.train2, "tpr", "fpr")
#plot(perf.train)
#auc.train <- performance(pred.train2, "auc")
#auc.train <- auc.train@y.values[[1]]

#ROC and AUC on validation
pred.rf <- predict(rf_model, telco$validation[,-1], type="prob")
pred.rf2 <- prediction(pred.rf$`1`, telco$validation$CHURN)
perf.rf <- performance(pred.rf2, "tpr", "fpr")
plot(perf.rf)

auc.rf <- performance(pred.rf2, "auc")
auc.rf <- auc.rf@y.values[[1]]
# Performance on validation

### NEURAL NETWORK

nn <- train(as.factor(CHURN) ~ . , data = telco$train, method = 'nnet', 
              preProcess = c('center', 'scale'), 
              trControl = trainControl(method="cv",number=5), 
              tuneGrid=expand.grid(size=c(10), decay=c(0.8)))
# size gives the number of neurons and decay is the regularization term

print(nn)
str(nn)
print(nn$finalModel)

#ROC and AUC on validation
pred.nn <- predict(nn, telco$validation[,-1], type="prob")
pred.nn2 <- prediction(pred.nn$`1`, telco$validation$CHURN)
perf.nn <- performance(pred.nn2, "tpr", "fpr")
plot(perf.nn)

auc.nn <- performance(pred.nn2, "auc")
auc.nn <- auc.nn@y.values[[1]]
auc.nn
# Performance on validation



### SUPPORT VECTOR MACHINES
#svm <- train(as.factor(CHURN) ~ . , data = telco$train, method = 'svmRadial', 
#            preProcess = c('center', 'scale'), 
#            trControl = trainControl(method="cv",number=5), 
#            tuneGrid=expand.grid(sigma=0.1, C=1))

#print(svm)
#str(svm)
#print(svm$finalModel)

#ROC and AUC on validation
#pred.svm <- predict(svm, telco$validation[,-1], type="prob")
#pred.svm2 <- prediction(pred.svm, telco$validation$CHURN)
#perf.svm <- performance(pred.svm2, "tpr", "fpr")
#plot(perf.svm)

#auc.svm <- performance(pred.svm2, "auc")
#auc.svm <- auc.svm@y.values[[1]]
# Performance on validation


### LOGISTIC REGRESSION
logit <- glm(CHURN ~ ., family = binomial(link = "logit"), telco$train)
step(logit)
summary(logit)
pred.train <- predict(logit, telco$train[,-1], type = "response")
pred.train <- ifelse(pred.train>0.5, "1", "0")
#confusion matrix on train set
confusionMatrix(as.factor(pred.train), as.factor(telco$train$CHURN))

#performance on validation set
pred.log <- predict(logit, telco$validation[,-1], type = "response")
pred.log2 <- prediction(pred.log, telco$validation$CHURN)
perf.log <- performance(pred.log2, "tpr", "fpr")
plot(perf.log)

auc.log <- performance(pred.log2, "auc")
auc.log <- auc.log@y.values[[1]]


### PENALISED LOGISTIC REGRESSION
x <- model.matrix(CHURN~., data = telco$train)

# Ridge Regression

ridge_model <- cv.glmnet(x, as.matrix(telco$train$CHURN), 
                         family = "binomial", alpha = 0)
ridge_model$lambda.min
x.val <- model.matrix(CHURN~., data = telco$validation)
pred.ridge <- predict(ridge_model, x.val, type="response")

pred.ridge2 <- prediction(as.numeric(pred.ridge), telco$validation$CHURN)
perf.ridge <- performance(pred.ridge2, "tpr", "fpr")
plot(perf.ridge)

auc.ridge <- performance(pred.ridge2, "auc")
auc.ridge <- auc.ridge@y.values[[1]]

#Lasso Regression

lasso_model <- cv.glmnet(x, as.matrix(telco$train$CHURN), 
                         family = "binomial", alpha = 1)

x.val <- model.matrix(CHURN~., data = trend.val)
pred.lasso <- predict(lasso_model, x.val, type="response")

pred.lasso2 <- prediction(as.numeric(pred.lasso),  telco$validation$CHURN)
perf.lasso <- performance(pred.lasso2, "tpr", "fpr")
plot(perf.lasso)

auc.lasso <- performance(pred.lasso2, "auc")
auc.lasso <- auc.lasso@y.values[[1]]


# PLOT of ROC of different models

cols <- c("Random Forest"="#f04546","Neural Network"="#3591d1","Logistic Regression"="#62c76b")
ggplot() + geom_line(aes(x=perf.rf@x.values[[1]],y=perf.rf@y.values[[1]], col="Random Forest")) +
  geom_line(aes(x=perf.nn@x.values[[1]],y=perf.nn@y.values[[1]], color="Neural Network")) + 
  geom_line(aes(x=perf.log@x.values[[1]],y=perf.log@y.values[[1]], color="Logistic Regression")) + 
  scale_colour_manual(name="",values=cols)+
  ylab('TP')+xlab('FP') + ggtitle("ROC curves on Validation set")

AUC <- data.frame(t(c(auc.log, auc.lasso, auc.ridge, auc.nn, auc.rf)))
colnames(AUC) <- c('Logistic', 'Lasso', 'Ridge', 'Neural Network', 'Random Forest')
AUC



#####################
# OVERSAMPLING AND TREND VARIABLES
####################

###### 

trend.train <- as.tibble(telco$train) %>% mutate(com1week = COMPLAINT_1WEEK,
                                                 com1to2week = COMPLAINT_2WEEKS - COMPLAINT_1WEEK,
                                                 com2week1month = COMPLAINT_1MONTH - COMPLAINT_2WEEKS,
                                                 com1mo3mo = COMPLAINT_3MONTHS - COMPLAINT_1MONTH,
                                                 com3mo6mo = COMPLAINT_6MONTHS - COMPLAINT_3MONTHS) %>%
  dplyr::select(-(contains('COMPLAINT')))

trend.val <- as.tibble(telco$validation) %>% mutate(com1week = COMPLAINT_1WEEK,
                                                    com1to2week = COMPLAINT_2WEEKS - COMPLAINT_1WEEK,
                                                    com2week1month = COMPLAINT_1MONTH - COMPLAINT_2WEEKS,
                                                    com1mo3mo = COMPLAINT_3MONTHS - COMPLAINT_1MONTH,
                                                    com3mo6mo = COMPLAINT_6MONTHS - COMPLAINT_3MONTHS) %>%
  dplyr::select(-(contains('COMPLAINT')))

set.seed(15)
oversampled <- ovun.sample(CHURN~.,data=trend.train, method="both")


### LOGISTIC REGRESSION

logit <- glm(CHURN ~ ., family = binomial(link = "logit"), oversampled$data)

summary(logit)
pred.train <- predict(logit, oversampled$data[,-1], type = "response")
pred.train <- ifelse(pred.train>0.5, "1", "0")
#confusion matrix on train set
confusionMatrix(as.factor(pred.train), as.factor(oversampled$data$CHURN))

#performance on validation set
pred.log <- predict(logit, trend.val[,-1], type = "response")
pred.log2 <- prediction(pred.log, trend.val$CHURN)
perf.log <- performance(pred.log2, "tpr", "fpr")
plot(perf.log)

auc.log <- performance(pred.log2, "auc")
auc.log <- auc.log@y.values[[1]]


### NEURAL NETWORK

nn <- train(as.factor(CHURN) ~ . , data = oversampled$data, method = 'nnet', 
            preProcess = c('center', 'scale'), 
            trControl = trainControl(method="cv",number=5), 
            tuneGrid=expand.grid(size=c(10), decay=c(0.8)))
# size gives the number of neurons and decay is the regularization term

print(nn)
str(nn)
print(nn$finalModel)

#ROC and AUC on validation
pred.nn <- predict(nn, trend.val[,-1], type="prob")
pred.nn2 <- prediction(pred.nn$`1`, trend.val$CHURN)
perf.nn <- performance(pred.nn2, "tpr", "fpr")
plot(perf.nn)

auc.nn <- performance(pred.nn2, "auc")
auc.nn <- auc.nn@y.values[[1]]
# Performance on validation


#### RANDOM FOREST

rf_model <- train(as.factor(CHURN)~., data=oversampled$data, method="rf",
                  trControl=trainControl(method="cv",number=5),
                  ntree=80)

pred.rf <- predict(rf_model, trend.val[,-1], type="prob")
pred.rf2 <- prediction(pred.rf$`1`, trend.val$CHURN)
perf.rf <- performance(pred.rf2, "tpr", "fpr")
plot(perf.rf)

auc.rf <- performance(pred.rf2, "auc")
auc.rf <- auc.rf@y.values[[1]]


#### PENALISED LOGISTIC REGRESSION
x <- model.matrix(CHURN~., data = oversampled$data)

# Ridge Regression

ridge_model <- cv.glmnet(x, as.matrix(oversampled$data$CHURN), 
                         family = "binomial", alpha = 0)
ridge_model$lambda.min

x.val <- model.matrix(CHURN~., data = trend.val)
pred.ridge <- predict(ridge_model, x.val, type="response")

pred.ridge2 <- prediction(as.numeric(pred.ridge), trend.val$CHURN)
perf.ridge <- performance(pred.ridge2, "tpr", "fpr")
plot(perf.ridge)

auc.ridge <- performance(pred.ridge2, "auc")
auc.ridge <- auc.ridge@y.values[[1]]

# Lasso Regression

lasso_model <- cv.glmnet(x, as.matrix(oversampled$data$CHURN), 
                         family = "binomial", alpha = 1)

x.val <- model.matrix(CHURN~., data = trend.val)
pred.lasso <- predict(lasso_model, x.val, type="response")
as.numeric(pred.lasso)
pred.lasso2 <- prediction(as.numeric(pred.lasso), trend.val$CHURN)
perf.lasso <- performance(pred.lasso2, "tpr", "fpr")
plot(perf.lasso)

auc.lasso <- performance(pred.lasso2, "auc")
auc.lasso <- auc.lasso@y.values[[1]]

cols <- c("Random Forest"="#f04546","Neural Network"="#3591d1","Logistic Regression"="#62c76b")
ggplot() + geom_line(aes(x=perf.rf@x.values[[1]],y=perf.rf@y.values[[1]], col="Random Forest")) +
  geom_line(aes(x=perf.nn@x.values[[1]],y=perf.nn@y.values[[1]], color="Neural Network")) + 
  geom_line(aes(x=perf.log@x.values[[1]],y=perf.log@y.values[[1]], color="Logistic Regression")) + 
  scale_colour_manual(name="",values=cols)+
  ylab('TP')+xlab('FP') + ggtitle("ROC curves on Validation set (Oversampling)")

AUC.over <- data.frame(t(c(auc.log, auc.lasso, auc.ridge, auc.nn, auc.rf)))
colnames(AUC.over) <- c('Logistic', 'Lasso', 'Ridge', 'Neural Network', 'Random Forest')

#### AUC COMPARISON
AUC  
AUC.over  # with oversampling and trend variables



#############
# NN AND RANDOM FOREST PERFORMANCES ON TEST SET
#############

##variable importance if it doesn't work->dev.off()
### RANDOM FOREST ON TRAIN+VALIDATION

DATA <- rbind(telco$train, telco$validation)

rf_model <- train(as.factor(CHURN)~., data=DATA, method="rf",
                  trControl=trainControl(method="cv",number=5),
                  ntree=80, importance=T)


pred.rf <- predict(rf_model, telco$test[,-1], type="prob")
pred.rf2 <- prediction(pred.rf$`1`, telco$test$CHURN)
perf.rf <- performance(pred.rf2, "tpr", "fpr")
plot(perf.rf)

auc.test.rf <- performance(pred.rf2, "auc")
auc.test.rf <- auc.test.rf@y.values[[1]]


### NN ON TRAIN+VALIDATION

nn <- train(as.factor(CHURN) ~ . , data = DATA, method = 'nnet', 
            preProcess = c('center', 'scale'), 
            trControl = trainControl(method="cv",number=5), 
            tuneGrid=expand.grid(size=c(10), decay=c(0.8)))
# size gives the number of neurons and decay is the regularization term

#ROC and AUC on test
pred.nn <- predict(nn, telco$test[,-1], type="prob")
pred.nn2 <- prediction(pred.nn$`1`, telco$test$CHURN)
perf.nn <- performance(pred.nn2, "tpr", "fpr")
plot(perf.nn)

auc.test.nn <- performance(pred.nn2, "auc")
auc.test.nn <- auc.test.nn@y.values[[1]]


# AUC comparison
auc.test.rf
auc.test.nn

######
# FINAL MODEL
######


rf_model_final <- train(as.factor(CHURN)~., data=telco.train, method="rf",
                  trControl=trainControl(method="cv",number=5),
                  ntree=80, importance=T)


#######PREDICTING ACTUAL TEST SET

telco.TEST$AVG_DATA_3MONTH <- replace_na(telco.TEST$AVG_DATA_3MONTH, 0)
telco.TEST$AVG_DATA_1MONTH <- replace_na(telco.TEST$AVG_DATA_1MONTH, 0)
telco.TEST$COUNT_CONNECTIONS_3MONTH <- replace_na(telco.TEST$COUNT_CONNECTIONS_3MONTH, 0)

# we will not include FIN_STATE in the analysis 

telco.TEST <- dplyr::select(telco.TEST, -c(-c(FIN_STATE, ID)))


finalprediction.rf <- predict(rf_model_final, telco.TEST[,-1], type="prob")


churnprediction <- cbind(telco.TEST$ID,finalprediction.rf[,2])
churnprediction <- as.data.frame(churnprediction)
colnames(churnprediction) <- c("ID","CHURN")
str(churnprediction)
churnprediction$CHURN <- ifelse(churnprediction$CHURN==0,0.0001,churnprediction$CHURN)
write.csv(churnprediction, "pred-randomforest.csv",row.names=FALSE)

