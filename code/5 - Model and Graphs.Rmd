```{r}

```

---
title: "Social Dynamics Lab Project - statistical analysis"
author: "Patrick Montanari"
date: '2023-01-02'
output:
  html_document: default
  pdf_document: default
editor_options: 
  markdown: 
    wrap: sentence
---

# Setup

```{r}
setwd("C:/Users/patri/Desktop/PATRICK/Università/Didattica/Corsi/Data Science/Social Dynamics Lab/Project")
library(ggplot2)
library(igraph)
library(dplyr)
library(car)
library("ISLR2")
library("e1071")
library(glmnet)
library(tidymodels)
library(gbm)
library(randomForest)
library(lmerTest)
```

The first thing I do is load the dataset to work with; this script assumes all data has already been re-coded and merged (see python scripts).

```{r}
tsdata <- read.csv('merged_data.csv')
```

The average number of touch screen events for an half hour is 210, with a maximum of 1400; since the range isn't as wide as using userid's average, I'll use a linear model to see which are the most important deciding factors. That, however, needs further data cleaning.

### Data cleaning

Age needs to be turned into a numerical variable instead of ordered categories. Since there are some intervals, I'll just get an average; 31+ was meant to indicate all those above 31; I chose 32, which is a semplification but I didn't want to risk going to far from the lower bound.

```{r}
categories <- unique(tsdata$age)
categories

```

```{r}
tsdata["age"][tsdata["age"] == "17-18"] <- 17.5
tsdata["age"][tsdata["age"] == "25-26"] <- 25.5
tsdata["age"][tsdata["age"] == "27-30"] <- 28.5
tsdata["age"][tsdata["age"] == "31+"] <- 32                    
tsdata$age <- as.numeric(tsdata$age)                            

summary(tsdata)
```

We want to examine touch screen events distributio; for this reason, we proceed with a density plot.

```{r}
touch <- density(tsdata$touch_screen_events)
plot(touch)
summary(tsdata$touch_screen_events)
```

As we can see, data has a strong negative skewness; I proceed with a logistic transformation of the touch screen events variable, using their logarithm instead to improve the predictive power and handle data which doesn't have a normal distribution.

```{r}
log_touch <- log(tsdata$touch_screen_events)
tsdata$log_touch <- log_touch

log_touch <- density(log_touch)
plot(log_touch)
summary(tsdata$log_touch)
```

Before proceeding I remove students from Agricultural and Medicine department, as they only had 2 students each.

```{r}
tsdata = subset(tsdata, department !='Medicine and veterinary medicine' & department !='Agricultural')
```

I also create a new hour variable considering minutes as well.

```{r}
tsdata$hours <- tsdata$hour + (tsdata$minute/100*5/3)
tsdata$hour <- tsdata$hours
```

I have already grouped data by ID and ranked them based on the total id usage on the python script; I repeat the same operation here in case I'd need it later, using the same split in three groups.

```{r}
ranking <- tsdata %>% 
  group_by(userid) %>% 
  summarise(touch_screen_events = sum(touch_screen_events)) 
ranking <- ranking[order(ranking$touch_screen_events),]

log_r_touch <- log(ranking$touch_screen_events)
ranking$log_touch <- log_r_touch

length(ranking$userid)/3
bottom_group <- ranking[1:61,]
middle_group <- ranking[62:123,]
top_group <- ranking[123:185,]
```

We remove missing values for the personality traits to avoid complications.

```{r}
tsdata["withw"][tsdata["withw"] == "with_others"] <- "with others"     
#To improve displayment of results

tsdata <- tsdata[!is.na(tsdata$Extraversion), ]
tsdata <- tsdata[!is.na(tsdata$Neuroticism), ]
tsdata <- tsdata[!is.na(tsdata$Openness), ]
tsdata <- tsdata[!is.na(tsdata$Agreeableness), ]
tsdata <- tsdata[!is.na(tsdata$Conscientiousness), ]
```

# Part 1 - Touch Screen Usage Model

### Multilevel Model - Touch Screen Usage

Possible multilevel factors: day of the week, with who, activity.

```{r}
model_1 <- lmer(log_touch~Extraversion + Conscientiousness + Agreeableness+              Neuroticism + Openness + gender + department + poly(age,2, raw = TRUE) + what + poly(hour, 3, raw=TRUE) +(1| userid), data=tsdata)
summary(model_1)
```

Now I compute the model's accuracy

```{r}
anova(model_1)
```

Better description found in the report.

# Part 2 - Personality Traits Models

## Extraversion trait

The first thing we need is to understand whether the touch screen usage has a linear or quadratic relation with our dependent variable, and if it's better to use the absolute values or their logarithm.

We perform a test/train split on our data, with 50% of the sample for each set.

```{r}
train <- sample(nrow(tsdata), 0.5*nrow(tsdata))
lm_raw.fit <- lm(Extraversion ~ touch_screen_events, data=tsdata, subset=train)
ext.pred <- predict(lm_raw.fit, tsdata) 
mean((tsdata$Extraversion - ext.pred)[-train]^2) #MSE for absolute values
```

```{r}
lm_raw2.fit <- lm(Extraversion ~ poly(touch_screen_events,2, raw = TRUE), data=tsdata, subset=train)
ext.ped <- predict(lm_raw2.fit, tsdata) 
mean((tsdata$Extraversion - ext.pred)[-train]^2)
```

```{r}
lm_log.fit <- lm(Extraversion ~ log_touch, data=tsdata, subset=train)
ext.pred <- predict(lm_log.fit, tsdata) 
mean((tsdata$Extraversion - ext.pred)[-train]^2)
```

```{r}
lm_log2.fit <- lm(Extraversion ~ poly(log_touch,2, raw = TRUE), data=tsdata, subset=train)
ext.ped <- predict(lm_log2.fit, tsdata) 
mean((tsdata$Extraversion - ext.pred)[-train]^2)
```

Since the prediction improves using the logarithm values but the quadratic has almost no improvements over the linear, we choose to only use the first degree.

I drop the absolute values from the dataset, together with the time information and userid as I will not consider them to predict personality. I will keep activities to see if any are more frequent, as those not useful will be removed as Lasso Regression also performs variable selection.

```{r}
tsdata = subset(tsdata, select = -c(touch_screen_events, day, minute, hours))
```

### Lasso Regression

First, I prepare the data by performing a train/test split and removing the first column to only get the predictors. I will have 4 iterations: with only age and gender, with only hour and weekday, with all and including touch screen.

```{r}
set.seed(10)
y1 <- tsdata$Extraversion
```

```{r}
# Compute R^2 from true and predicted values!
evaluator <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - (SSE / SST)
  MSE <- SSE/nrow(df)

  
  # Model performance metrics
data.frame(
  MSE <- MSE,
  Rsquare <- R_square
)
}
```

#### Model 1

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Extraversion ~ poly(hour,3, raw=TRUE)+ week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Extraversion ~ poly(hour,3, raw=TRUE)+ week, data=tsdata[-train, ])[,-1]

y_train <- y1[train]
y_test <- y1[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)

```

The MSE for hour and week is 676.40 for train data and 675.81 for test data; R-squared corresponds to around 0.039% of variance when predicting over train data and 0.027% for test data.

#### Model 2

Now, I repeat the same operation for hour and weekday.

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Extraversion ~ poly(age,2, raw = TRUE)+gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Extraversion ~ poly(age,2, raw = TRUE)+gender, data=tsdata[-train, ])[,-1]

y_train <- y1[train]
y_test <- y1[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

The MSE for hour and week is 675.66 for train data and 675.21 for test data; R-squared corresponds to around 0.148% of variance when predicting over train data and 0.116% for test data. Hour and weekday of phone usage seem to be less important for predicting someone's Extraversion; age's relationship was proven to be quadratic with the dependent variable.

#### Model 3

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Extraversion ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE) + gender, data=tsdata[train, ])
x_test <- model.matrix(Extraversion ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE) + gender, data=tsdata[-train, ])

y_train <- y1[train]
y_test <- y1[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

Considering age, gender, hour and day the MSE are 675.23 and 675.18 , but the R-squared now corresponds to 0.21% of variance for train data and 0.12% of variance when predicting test data.

```{r}
# display coefficients using lambda proven to be the best one by CV
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

#### Model 4

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Extraversion ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Extraversion ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[-train, ])[,-1]

y_train <- y1[train]
y_test <- y1[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

If we include the touch screen events variable the model becomes more accurate (670.89 and 672.30 MSE), and the R-squared further increases, now corresponding to respectively 0.85% of Extroversion's variance for train data and 0.55% for test data

#### Model 5

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Extraversion ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch + department, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Extraversion ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch + department, data=tsdata[-train, ])[,-1]

y_train <- y1[train]
y_test <- y1[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

### Random Forest

#### Model 1

First, only using hour and weekday.

```{r}
set.seed(30)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Extraversion
y_test <- test$Extraversion


forest_model <- randomForest(Extraversion ~ hour+week, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model

```

```{r}
yhat.forest_train <- predict(forest_model, newdata=train)
yhat.forest_test <- predict(forest_model, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

Only using weekday and hour isn't sufficient; now, I try with age and gender.

#### Model 2

```{r}
set.seed(30)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Extraversion
y_test <- test$Extraversion


forest_model2 <- randomForest(Extraversion ~ age+gender, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model2
```

```{r}
yhat.forest_train <- predict(forest_model2, newdata=train)
yhat.forest_test <- predict(forest_model2, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

The results are much better: MSE are 583.10 and 580.09, while R-squared scores corresponds to 13.98% and 13.63% of variance.

#### Model 3

I consider all four previous factors.

```{r}
set.seed(30)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Extraversion
y_test <- test$Extraversion


forest_model3 <- randomForest(Extraversion ~ age+gender+hour+week, data=train, mtry=4, ntree=50, importance=TRUE)
forest_model3
```

```{r}
yhat.forest_train <- predict(forest_model3, newdata=train)
yhat.forest_test <- predict(forest_model3, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

The results have further improved, with new MSE of 538 and 575 and R-squared corresponding to 20.59% and 14.31% of variance.

```{r}
varImpPlot(forest_model3)
```

The first column represents the mean decrease in accuracy of the predictions when that variable is removed from the model. The second column is a measure of the total decrease in node impurity resulting from splits over that variable (averaged over all of the trees). Age is the most important factor for predictions, while hour and week have a very low impact on the MSE.

Now, I'll repeat the computations including touch screen events.

#### Model 4

```{r}
set.seed(30)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Extraversion
y_test <- test$Extraversion


forest_model4 <- randomForest(Extraversion ~ age+gender+hour+week+log_touch, data=train, mtry=5, ntree=50, importance=TRUE)
forest_model4
```

```{r}
yhat.forest_train <- predict(forest_model4, newdata=train)
yhat.forest_test <- predict(forest_model4, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

```{r}
varImpPlot(forest_model4)
```

The models are more accurate by including log_touch, proven by new MSE of 115.06 and 415.11, and R-squared corresponding to 83.03% of variance for training and 38.19% for test data. Do note that this is also due to the nature of the model, performing 50 decision trees and averaging results.

#### v Model 5

```{r}
set.seed(30)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Extraversion
y_test <- test$Extraversion


forest_model5 <- randomForest(Extraversion ~ age+gender+hour+week+log_touch + department, data=train, mtry=6, ntree=50, importance=TRUE)
forest_model5
```

```{r}
yhat.forest_train <- predict(forest_model5, newdata=train)
yhat.forest_test <- predict(forest_model5, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 57.95 / 222.94

R-squared: 0.9144 / 0.6701

```{r}
varImpPlot(forest_model5)
```

## Neuroticism Trait

I want to see again whether the relationship is linear or logarithm again, and if it's on first or second degree.

```{r}
set.seed(2)
train <- sample(nrow(tsdata), 0.5*nrow(tsdata))
lm_raw.fit <- lm(Neuroticism ~ touch_screen_events, data=tsdata, subset=train)
ext.pred <- predict(lm_raw.fit, tsdata) 
mean((tsdata$Neuroticism - ext.pred)[-train]^2) #MSE for absolute values
```

```{r}
set.seed(2)
lm_raw2.fit <- lm(Neuroticism ~ poly(touch_screen_events,2), data=tsdata, subset=train)
ext.ped <- predict(lm_raw2.fit, tsdata) 
mean((tsdata$Neuroticism - ext.pred)[-train]^2)
```

```{r}
set.seed(2)
lm_log.fit <- lm(Neuroticism ~ log_touch, data=tsdata, subset=train)
ext.pred <- predict(lm_log.fit, tsdata) 
mean((tsdata$Neuroticism - ext.pred)[-train]^2)
```

```{r}
set.seed(2)
lm_log2.fit <- lm(Neuroticism ~ poly(log_touch,2), data=tsdata, subset=train)
ext.ped <- predict(lm_log2.fit, tsdata) 
mean((tsdata$Neuroticism - ext.pred)[-train]^2)
```

Even just with one terms we can see that touch screen events are more suited for predicting the Neuroticism trait, due to the much lower MSE. The difference between absolute values and their logarithm is minimal, but logarithm remains better. The quadratic term seem to be unnecessary.

I again drop the absolute values from the dataset, together with the time information and userid as I will not consider them to predict personality. I also make a version without touch screen events.

### Lasso Regression

```{r}
y2 <- tsdata$Neuroticism
```

#### Model 1

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Neuroticism ~ poly(hour,3,raw = TRUE) + week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Neuroticism ~ poly(hour,3,raw = TRUE)+ week, data=tsdata[-train, ])[,-1]

y_train <- y2[train]
y_test <- y2[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)

```

Only using hour and weekday we have MSE values of 412.45 for train and 410.08 for test data; R-squared amounts to less than e-3 of Neuroticism's variance for both train and test data; neuroticism has no correlation with phone usage trends.

#### Model 2

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Neuroticism ~ poly(age,2)+gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Neuroticism ~ poly(age,2)+gender, data=tsdata[-train, ])[,-1]

y_train <- y2[train]
y_test <- y2[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

Only using age and gender we have MSE values of 380.81 for train and 375.88 for test data; R-squared amounts to 7.67% and 8.32% of Neuroticism's variance for train and test data. Neuroticism seem to be much more predictable based on gender and age (again on the second power).

#### Model 3

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Neuroticism ~ poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender, data=tsdata[train, ])
x_test <- model.matrix(Neuroticism ~  poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender, data=tsdata[-train, ])

y_train <- y2[train]
y_test <- y2[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

Considering all 4 parameters amount to a total MSE of 380.70 and 375.99, while the R-squared are 7.69% and 8.29%; the results are very similar to the previous one, I check to see if any variable selection was performed.

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

Gender and age are very strong predictors, with Males being less neurotic than females on average.

#### Model 4

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Neuroticism ~ poly(hour,3, raw = TRUE) +  week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Neuroticism ~  poly(hour,3, raw = TRUE) + week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[-train, ])[,-1]

y_train <- y2[train]
y_test <- y2[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

Adding touch screen events improved both versions of the model, with new MSE of 378.04 and 373.59.\

R-squared scores corresponds to around 8.34% of variance when predicting over train data and 8.87% for test data

#### Model 5

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Neuroticism ~ poly(hour,3, raw = TRUE) +  week+ poly(age,2, raw = TRUE)  + gender+log_touch + department, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Neuroticism ~  poly(hour,3, raw = TRUE) + week+ poly(age,2, raw = TRUE)  + gender+log_touch+ department, data=tsdata[-train, ])[,-1]

y_train <- y2[train]
y_test <- y2[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE of 371.30 and 364.46. R-squared of 0.1069 and 0.1119.

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

Consistently with our hypothesis, high neurotic individuals tend to use their phone less.

### Random Forest

#### Model 1

First, using hour and weekday.

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model <- randomForest(Neuroticism ~ hour+week, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model
```

```{r}
yhat.forest_train <- predict(forest_model, newdata=train)
yhat.forest_test <- predict(forest_model, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

This time MSE are much lower, but R-squared remain close to zero.

#### Model 2

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model2 <- randomForest(Neuroticism ~ age+gender, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model2
```

```{r}
yhat.forest_train <- predict(forest_model2, newdata=train)
yhat.forest_test <- predict(forest_model2, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

Once again age and gender are very important factors; new MSE are 330.41 and 330.84; R-squared corresponds to 19.79% and 19.16% of variance.

#### Model 3

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model3 <- randomForest(Neuroticism ~ age+gender+hour+week, data=train, mtry=4, ntree=50, importance=TRUE)
forest_model3
```

```{r}
yhat.forest_train <- predict(forest_model3, newdata=train)
yhat.forest_test <- predict(forest_model3, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

This time, gender is more important than age; MSE are 303.69 and 331.59, while R-squareds are 26.27% and 18.98%.

Now I repeat including log_touch.

#### Model 4

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model4 <- randomForest(Neuroticism ~ age+gender+hour+week+log_touch, data=train, mtry=5, ntree=50, importance=TRUE)
forest_model4
```

```{r}
yhat.forest_train <- predict(forest_model4, newdata=train)
yhat.forest_test <- predict(forest_model4, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

The results are very similar to the other model including log_touch; the new MSE are 66.19 and 251.55, while the R-squared are 83.92% and 38.53%.

```{r}
varImpPlot(forest_model4)
```

just like before, log_touch sits in between the two pairs of age / gender and hour / week.

#### Model 5

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model5 <- randomForest(Neuroticism ~ age+gender+hour+week+log_touch + department, data=train, mtry=6, ntree=50, importance=TRUE)
forest_model5
```

```{r}
yhat.forest_train <- predict(forest_model5, newdata=train)
yhat.forest_test <- predict(forest_model5, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 25.38 / 102.35

R-squared: 0.9383 / 0.7511

```{r}
varImpPlot(forest_model5)
```

## Openness Trait

```{r}
y3 <- tsdata$Openness
```

### Lasso Regression

#### Model 1

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Openness ~ poly(hour,3, raw=TRUE)+ week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Openness ~ poly(hour,3, raw=TRUE)+ week, data=tsdata[-train, ])[,-1]

y_train <- y3[train]
y_test <- y3[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)

```

MSE: 330.39 / 325.93

R-squared:0.0037 / 0.0054

#### Model 2

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Openness ~ poly(age,2, raw = TRUE)+gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Openness ~ poly(age,2, raw = TRUE)+gender, data=tsdata[-train, ])[,-1]

y_train <- y3[train]
y_test <- y3[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 313.47 / 309.88

R-squared: 0.0547 / 0.0544

#### Model 3

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Openness ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Openness ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender, data=tsdata[-train, ])[,-1]

y_train <- y3[train]
y_test <- y3[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 312.58 / 308.49

R-squared: 0.0547 / 0.0586

#### Model 4

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Openness ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Openness ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch, data=tsdata[-train, ])[,-1]

y_train <- y3[train]
y_test <- y3[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 312.54 / 308.35

R-squared: 0.0575 / 0.0590

#### Model 5

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Openness ~ poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch + department, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Openness ~  poly(hour,3, raw=TRUE)+ week+ poly(age,2, raw = TRUE)  + gender+log_touch + department, data=tsdata[-train, ])[,-1]

y_train <- y3[train]
y_test <- y3[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 294.79 / 291.96

R-squared: 0.1111 / 0.1091

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

### Random Forest

#### Model 1

```{r}
set.seed(40)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Neuroticism
y_test <- test$Neuroticism


forest_model <- randomForest(Openness ~ hour+week, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model
```

```{r}
yhat.forest_train <- predict(forest_model, newdata=train)
yhat.forest_test <- predict(forest_model, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 323.09 / 328.09

R-squared: 0.0186 / 0.0088

#### Model 2

```{r}
set.seed(40)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Openness
y_test <- test$Openness


forest_model2 <- randomForest(Openness ~ age+gender, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model2
```

```{r}
yhat.forest_train <- predict(forest_model2, newdata=train)
yhat.forest_test <- predict(forest_model2, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 258.27 / 261.97

R-squared: 0.2155 / 0.2086

#### Model 3

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Openness
y_test <- test$Openness


forest_model3 <- randomForest(Openness ~ age+gender+hour+week, data=train, mtry=4, ntree=50, importance=TRUE)
forest_model3
```

```{r}
yhat.forest_train <- predict(forest_model3, newdata=train)
yhat.forest_test <- predict(forest_model3, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 235.80 / 235.33

R-squared: 0.2898 /0.2890

#### Model 4

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Openness
y_test <- test$Openness


forest_model4 <- randomForest(Openness ~ age+gender+hour+week+log_touch, data=train, mtry=5, ntree=50, importance=TRUE)
forest_model4
```

```{r}
yhat.forest_train <- predict(forest_model4, newdata=train)
yhat.forest_test <- predict(forest_model4, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 45.414 / 179.62

R-Squared: 0.8620 / 0.4578

#### Model 5

```{r}
set.seed(38)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Openness
y_test <- test$Openness


forest_model5 <- randomForest(Openness ~ age+gender+hour+week+log_touch+department, data=train, mtry=6, ntree=50, importance=TRUE)
forest_model5
```

```{r}
yhat.forest_train <- predict(forest_model5, newdata=train)
yhat.forest_test <- predict(forest_model5, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 24.182 / 93.631

R-squared: 0.9265 / 0.7174

```{r}
varImpPlot(forest_model5)
```

## Agreeableness Trait

```{r}
y4 <- tsdata$Agreeableness
```

### Lasso Regression

#### Model 1

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Agreeableness ~ poly(hour,3, raw=TRUE) + week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Agreeableness ~ poly(hour,3, raw=TRUE) + week, data=tsdata[-train, ])[,-1]

y_train <- y4[train]
y_test <- y4[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)

```

MSE: 262.69 / 260.35

R-squared: 0 / 0.0001

#### Model 2

```{r}
set.seed(10)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Agreeableness ~ poly(age,2, raw=TRUE) + gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Agreeableness ~ poly(age,2, raw=TRUE) + gender, data=tsdata[-train, ])[,-1]

y_train <- y4[train]
y_test <- y4[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 231.21 / 228.49

R-squared: 0.1198 / 0.1223

#### Model 3

```{r}

set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Agreeableness ~ poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender, data=tsdata[train, ])
x_test <- model.matrix(Agreeableness ~  poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender, data=tsdata[-train, ])

y_train <- y4[train]
y_test <- y4[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 231.62 / 228.15

R-squared: 0.1219 / 0.1199

#### Model 4

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Agreeableness ~ poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender+log_touch, data=tsdata[train, ])
x_test <- model.matrix(Agreeableness ~  poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender+log_touch, data=tsdata[-train, ])

y_train <- y4[train]
y_test <- y4[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 231.55 / 228.12

R-squared: 0.1221 / 0.1201

#### Model 5

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Agreeableness ~ poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender+log_touch + department , data=tsdata[train, ])
x_test <- model.matrix(Agreeableness ~  poly(hour,3,raw = TRUE) + week+ poly(age,2, raw = TRUE) + gender+log_touch + department, data=tsdata[-train, ])

y_train <- y4[train]
y_test <- y4[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 224.18 / 220.96

R-squared: 0.1501 / 0.1477

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

### Random Forest

#### Model 1

```{r}
set.seed(42)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Agreeableness
y_test <- test$Agreeableness


forest_model <- randomForest(Agreeableness ~ hour+week, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model
```

```{r}
yhat.forest_train <- predict(forest_model, newdata=train)
yhat.forest_test <- predict(forest_model, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 258.169 / 264.55

r-squared: 0.0090 / 0.00031

#### Model 2

```{r}
set.seed(42)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Agreeableness
y_test <- test$Agreeableness


forest_model2 <- randomForest(Agreeableness ~ age+gender, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model2
```

```{r}
yhat.forest_train <- predict(forest_model2, newdata=train)
yhat.forest_test <- predict(forest_model2, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 186.82 / 189.55

R-squared: 0.2850 / 0.2769

#### Model 3

```{r}
set.seed(42)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Agreeableness
y_test <- test$Agreeableness


forest_model3 <- randomForest(Agreeableness ~ hour+week+age+gender, data=train, mtry=4, ntree=50, importance=TRUE)
forest_model3

```

```{r}
yhat.forest_train <- predict(forest_model3, newdata=train)
yhat.forest_test <- predict(forest_model3, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 161.08 / 191.99

R-squared: 0.3816 / 0.2740

#### Model 4

```{r}
set.seed(40)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Agreeableness
y_test <- test$Agreeableness


forest_model4 <- randomForest(Agreeableness ~ hour+week+log_touch + age + gender, data=train, mtry=5, ntree=50, importance=TRUE)
forest_model4

```

```{r}
yhat.forest_train <- predict(forest_model4, newdata=train)
yhat.forest_test <- predict(forest_model4, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)

```

MSE: 34.571 / 131.88

R-squared: 0.8673 / 0.5013

#### Model 5

```{r}
set.seed(40)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Agreeableness
y_test <- test$Agreeableness


forest_model5 <- randomForest(Agreeableness ~ hour+week+log_touch + department + age + gender, data=train, mtry=6, ntree=50, importance=TRUE)
forest_model5
```

```{r}
yhat.forest_train <- predict(forest_model5, newdata=train)
yhat.forest_test <- predict(forest_model5, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)

```

MSE: 14.396 / 53.352

R-squared: 0.9447 / 0.7982

```{r}
varImpPlot(forest_model5)
```

## Conscientiousness Trait

```{r}
y5 <- tsdata$Conscientiousness
```

### Lasso Regression

#### Model 1

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Conscientiousness ~ poly(hour,3,raw = TRUE) + week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Conscientiousness ~ poly(hour,3,raw = TRUE)+ week, data=tsdata[-train, ])[,-1]

y_train <- y5[train]
y_test <- y5[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 403.06 / 410.01

R-squared: 0.0027 / 0.0031

#### Model 2

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE) + gender, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE)+ gender, data=tsdata[-train, ])[,-1]

y_train <- y5[train]
y_test <- y5[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 389.55 / 396.68

R-squared: 0.0307 / 0.0318

#### Model 3

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE) + gender + poly(hour, 3, raw=TRUE) + week, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE)+ gender + poly(hour, 3, raw=TRUE) + week, data=tsdata[-train, ])[,-1]

y_train <- y5[train]
y_test <- y5[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 387.47 / 395.44

R-squared: 0.0359 / 0.0349

#### Model 4

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE) + gender + poly(hour, 3, raw=TRUE) + week+ log_touch, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE)+ gender + poly(hour, 3, raw=TRUE) + week+ log_touch, data=tsdata[-train, ])[,-1]

y_train <- y5[train]
y_test <- y5[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 386.46/ 394.31

R-squared: 0.0384 / 0.0377

#### Model 5

```{r}
set.seed(20)
train <- sample(1:nrow(tsdata), nrow(tsdata)/2)   #random sampling with a 50/50 split
test <- -train

x_train <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE) + gender + poly(hour, 3, raw=TRUE) + week+ log_touch + department, data=tsdata[train, ])[,-1]
x_test <- model.matrix(Conscientiousness ~ poly(age,2,raw = TRUE)+ gender + poly(hour, 3, raw=TRUE) + week+ log_touch + department, data=tsdata[-train, ])[,-1]

y_train <- y5[train]
y_test <- y5[test]


lasso_model <- glmnet(x_train, y_train, alpha=1)
cv.out <- cv.glmnet(x_train, y_train, alpha=1, lambda=lasso_model$lambda) 
#Cross-validation to choose best model to minimize MSE
bestlambda <- cv.out$lambda.min

lasso_pred_train <- predict(lasso_model, s=bestlambda, newx=x_train)
lasso_pred_test <- predict(lasso_model, s=bestlambda, newx=x_test)

evaluator(y_train, lasso_pred_train, x_train)
evaluator(y_test, lasso_pred_test, x_test)
```

MSE: 352.61 / 361.28

R-squared: 0.1226 / 0.1182

```{r}
lasso_coef <- predict(lasso_model, type="coefficients", s=bestlambda)
lasso_coef
```

## Random Forest

#### Model 1

```{r}
set.seed(44)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Conscientiousness
y_test <- test$Conscientiousness


forest_model <- randomForest(Conscientiousness ~ hour+week, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model

```

```{r}
yhat.forest_train <- predict(forest_model, newdata=train)
yhat.forest_test <- predict(forest_model, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 399.81 / 412.57

R-squared: 0.0150 / 0.0017

#### Model 2

```{r}

set.seed(44)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Conscientiousness
y_test <- test$Conscientiousness


forest_model2 <- randomForest(Conscientiousness ~ age+gender, data=train, mtry=2, ntree=50, importance=TRUE)
forest_model2

```

```{r}
yhat.forest_train <- predict(forest_model2, newdata=train)
yhat.forest_test <- predict(forest_model2, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)

```

MSE: 364.13 / 369.75

R-squared: 0.1029 / 0.1053

#### Model 3

```{r}
set.seed(44)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Conscientiousness
y_test <- test$Conscientiousness


forest_model3 <- randomForest(Conscientiousness ~ hour+week+age + gender, data=train, mtry=4, ntree=50, importance=TRUE)
forest_model3

```

```{r}
yhat.forest_train <- predict(forest_model3, newdata=train)
yhat.forest_test <- predict(forest_model3, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)

```

MSE: 310.41 / 371. 88

R-squared: 0.2353 / 0.1001

#### Model 4

```{r}
set.seed(44)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Conscientiousness
y_test <- test$Conscientiousness


forest_model4 <- randomForest(Conscientiousness ~ hour+week+age+gender+log_touch, data=train, mtry=5, ntree=50, importance=TRUE)
forest_model4
```

```{r}
yhat.forest_train <- predict(forest_model4, newdata=train)
yhat.forest_test <- predict(forest_model4, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)
```

MSE: 65.154 / 259.69

R-squared: 0.8395 / 0.3717

#### Model 5

```{r}
set.seed(44)
split <- initial_split(tsdata, prop=0.75)
train <- training(split)
test <- testing(split)
y_train <- train$Conscientiousness
y_test <- test$Conscientiousness


forest_model5 <- randomForest(Conscientiousness ~ hour+week+age+gender+department+log_touch, data=train, mtry=6, ntree=50, importance=TRUE)
forest_model5

```

```{r}
yhat.forest_train <- predict(forest_model5, newdata=train)
yhat.forest_test <- predict(forest_model5, newdata=test)

evaluator(y_train, yhat.forest_train, train)
evaluator(y_test, yhat.forest_test, test)

```

MSE: 27.436 / 109.37

R-squared: 0.9324 / 0.7353

```{r}
varImpPlot(forest_model5)
```
