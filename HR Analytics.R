getwd()
rm(list=ls())
ls()
setwd('/Users/..')

hr_data <- read.csv("turnover.csv")
glimpse(hr_data)



#------------------------------------------------------------------------
# Employee Insight for HR, applied statistical and Machine learning approach
# to understand the propensity of an employee to leave 

#------------------------------------------------------------------------

################[   packages to be installed ]############################

packages <- c("caret","deepnet","devtools","dplyr","dummies","e1071","ggfortify",
              "ggplot2","ggpubr","grid","gridExtra","pROC","randomForest","reshape2","rpart",
              "rpart.plot","bindrcpp","xgboost")
for(p in packages){
  suppressPackageStartupMessages(library(p,quietly = TRUE,character.only = TRUE))
}

install.packages("keras")
library(keras)
install_keras()
?option


################ Global Variables #############################################################

listRocCurves <<- c() #List which will store the data for each model to be used in ROC curves
dfModelPerformance <<- data.frame() #Dataframe which will store the performance data for each model
i <<- 1 #f lag initialized by 1 
CostFactorOfNewHiring <<- 4.0
CostFactorOfRetainingOldEmployee <<- 1.0
CostToRetainEmployee <<- 5 # in terms of 1000$
options(warn=-1)

#################################[ Data Familarization Module ]####################################



#Single function for the familiarity with Data.

DataFamilarization <- function(hr_data){
  
  # Hr-Data Dimension check
  dimen <- dim(hr_data)
  print(paste("Size of Data: Columns= ",dimen[2],"Rows=",dimen[1]))
  
  # Check for First & last 6 data
  head(hr_data)
  tail(hr_data)
  
  # View condensed summary & Class,Names,Summary
  str(hr_data)
  class(hr_data)
  names(hr_data)
  glimpse(hr_data)
  summary(hr_data)
  
  # boxplot & coefficient
  boxplot(hr_data)
  coef(hr_data)

}


#####################################[ Corelation Heatmap Module]######################################

PrintCorelationHeatMap <- function(hr_data){
  
  #Corellation matrix
  library(reshape2)
  library(ggplot2)
  ?melt
  melted_cormat <- melt()
  
  ggplot(melted_cormat,aes(Var2,Var1,fill=value)) +
    geom_tile(color="black") + 
      scale_fill_gradient(low = "red",medium="blue",limit=c(-1,1),name="Correlation") +
        theme(axis.text.x = element_text(angle = 10,hjust = 0.5))
    
    
}


#########################[ Train Test Split Module ]###############################################

TrainTestSplit <- function(hr_data,splitFactor=0.7,train=TRUE){
  # Split the Train and Test Data, remember we have cleaned the NA values using imputation
  
  trainData <- sample(1:nrow(hr_data),splitFactor*nrow(hr_data))
  
  # Check the splits 
  if(train==TRUE){
    return(hr_data[trainData,])
  }else{
    return(hr_data[-trainData,])
  }
}

############################ Confusin Matrix Module ##################################

GetConfusionMatrix <- function(testLabelData,test,testPrediction,modelType){
  
  # Prepre Confusion Matrix
  print(modelType)
  cm <- caret::confusionMatrix(as.factor(testPrediction),
                               as.factor(testLabelData),
                               dnn=c("Prediction","Actual"))
  class(cm)
  print(paste("Test data Confusion Matrix for ", modelType,"Model :"))
  
  # Performance Metrics
  
  accuracy <- paste(round(100*cm$overall["Accuracy"],digits = 2),"%")
  sensitivity <- paste(round(cm$byClass["Sensitivity"],digits = 4))
  specificity <- paste(round(cm$byClass["Specificity"],digits = 4))
  precision <- paste(round(cm$byClass["Precission"],digits = 4))
  recall <- paste(round(cm$byClass["Recall"],digits = 4))
  
  #Get Roc and Auc of a particular Model
  #library(pROC)
  
  roc_obj <- roc(testLabelData,as.numeric(testPrediction))
  areaUnderCurve <- round(auc(roc_obj),digits = 3)
  falsePositive <- cm$table[2,1]
  falseNegative <- cm$table[1,2]
  
  # Calcualting the CTC for employers retenetion+newHired
  costToCompany <- (CostFactorOfNewHiring*falseNegative) + 
                   (CostFactorOfRetainingOldEmployee*falsePositive)
  
  costToCompany <- paste("$", round(costToCompany * CostToRetainEmployee, digits = 2), "K")
  
  # Store model performance data into data frame so at the end of all models, we can print summary
  #df <- data.frame(i,modelType);  print(df)
  
  df <- data.frame(MODEL= modelType,
                   Accuracy = accuracy,
                   AUC = areaUnderCurve,
                   Specificity = specificity,
                   Precision = precision,
                   Sensitivity_Recall= sensitivity,
                   CostToCompany = costToCompany)
 
   print(paste("Model Performance:"))
   print(df)
   
   print(paste("False Negative =",
               falseNegative),
              "False Positive =",
              falsePositive,
              "Cost To Company=",
              costToCompany)
  
   # Store model ROC curve data into list so at the end of all models, we can plot combined ROC curve
    listRocCurves[[i]] <<- roc_obj
    names(listRocCurves)[i] <<- modelType
  
    dfModelPerformance <<- rbind(df,dfModelPerformance) 
    i <<- i + 1 # increment flag
}

######################### CompareModelsAndPlotCombinedRocCurve ##################################

#This function does 2 things
#1. print the performance metrices of each model in table format.
#2. plot combined ROC curve for all the mdoels.

CompareModelsAndPlotCombinedRocCurve <- function(){
  
  #print(dfModelPerformance)
  library(gridExtra)
  library(grid)
  library(ggplot2)
  
  grid.newpage()
  grid.table(dfModelPerformance)
  
  # Plot ROC curve
  rocCurve <- ggroc(listRocCurves,alpha=1,size=1) +
    ggtitle("ROC[Receiver Operating Characteristics] curve") +
     theme(axis.text = element_text(colour = "blue"))
  plot(rocCurve)
  
}


############################# Normalize Data ################################

#Function to normalize the data by dividing maximum of that data.
# This function to be used for data which is +ve so that it will transform in range 0-1

normalize_DivideByMax <- function(x)
{
  return(x/max(x))
}

######################### Logistic Regression Model ############################

CreateLogisticRegressionModel <- function(trainlabel, testLabelData, IndependentVariables,
                                          train, test)
{
  
  modelType = "Logistic Regression"
  print(paste("Creating Model for ",modelType))
  formula <- as.formula(paste(trainLabel, paste(IndependentVariables), sep = " ~ "))
  model <- glm(formula, data = train, binomial())
  
  #Summary of Logistic Regression Model
  print(summary(model))
  
  ##Stepwise regression to take only relevant variables
  #modelStep<-step(model, direction = "both", trace = 1)
  #print(summary(modelStep))
  
  #Predictions using Logistic regression Model with test Dataset
  #considering 1 for prob >=.5 and 0 0therwise
  predictions <- predict(model, test)
  testprediction <- ifelse(predictions >= .5, 1, 0)
  
  #Create confusion Matrix  
  GetConfusionMatrix(testLabelData, test, testprediction, modelType)
}


###################### Decision Tree Model ###############################

CreateDecisionTreeModel <- function(trainlabel,testlabelData,IndependentVariables,train,test){
  modelType ="Decission Tree"
  
  print(paste("Creating Model for ",modelType))
  
  # library for decision tree
  library(rpart)
  
  # Fitting Formula 
  formula <- as.formula(paste(trainlabel,paste(IndependentVariables),sep="~"))
  model <- rpart::rpart(formula,data = train,method = 'class') # Class to fit a binary Model
  
  summary(model)
  
  
  # Plot 
  install.packages("rpart.plot")
  library(rpart.plot)
  library(rpart)
  
  rpart.plot(model,type = 5,fallen.leaves = T,extra=8,
             cex=.58,trace=1,main="Decision Tree",cex.main=1.5,
             leaf.round=1,prefix="",branch.col="blue",branch.lwd = 2,box.palette = "RdGn",
             nn = F, branch.lty = 1) #3 dotted branch lines)
  
  #Predictions using Model with test Dataset
  predictions <- predict(model, test, type = "class")
  
  #Create confusion Matrix  
  GetConfusionMatrix(testLabelData, test, predictions, modelType)
  
}

###################### Random Forest Model ################################

CreateRandomForestModel <- function(trainlabel, testLabelData,
                                    IndependentVariables, train, test, numTree)
{
  modelType = "Random Forest"
  print(paste("Creating Model for ", modelType))
  
  #Import Library
  library(randomForest)
  
  #Fitting model
  formula <- as.formula(paste(trainLabel, paste(IndependentVariables), sep = " ~ "))
  model <- randomForest::randomForest(formula, data = train, importance = TRUE, ntree = numTree)
  print(model)
  
  #Summary of Model
  summary(model)
  
  ## Look at variable importance:
  print(round(importance(model), 5))
  
  
  #Predictions using Model with test Dataset
  predictions <- predict(model, test)
  
  ##Append the predicted values in the test dataset and save the file
  # test[ ,(ncol(test)+1)] <- predictions
  # names(test)[(ncol(test))]<-paste("Predictions")
  # write.csv(test, file = "Data\\testWithPredictionsWithRandomForest.csv")
  #Create confusion Matrix  
  GetConfusionMatrix(testLabelData, test, predictions, modelType)
  # graphics.off() 
  # par("mar") 
  # par(mar=c(1,1,1,1))
  # plot(model)
  # print (paste("Minimum Error at Tree : ",which.min(model$err.rate[,1])))
  
  #Create Variable importance plot
  
  varImpPlot(model)
}

############################## Kernel SVM Model #####################################################

CreateKernalSvmModel <- function(trainlabel, testLabelData, dependentVariables, train, test)
{
  modelType = "Kernal SVM"
  print(paste("Creating Model for ", modelType))
  
  #Guassuian Kernal SVM Model
  library(e1071)
  
  #Fitting model
  formula <- as.formula(paste(trainLabel, paste(dependentVariables), sep = " ~ "))
  model <- e1071::svm(formula = train$left ~ .,
                      data = train, type = 'C-classification', kernel = 'radial')
  
  #Summary of Model
  model
  
  #Predictions using Model with test Dataset
  predictions <- predict(model, test)
  
  #Create confusion Matrix  
  GetConfusionMatrix(testLabelData, test, predictions, modelType)
}


################################# Naive Bayes Model #################################################

CreateNaiveBayesModel <- function(trainlabel, testLabelData, dependentVariables, train, test)
{
  modelType = "Naive Bayes"
  print(paste("Creating Model for ", modelType))
  
  #Fitting the Naive Bayes Model
  library(e1071)
  formula <- as.formula(paste(trainLabel, paste(dependentVariables), sep = " ~ "))
  model <- e1071::naiveBayes(formula, data = train)
  summary(model)
  
  #Predictions using Model with test Dataset
  predictions <- predict(model, test)
  
  #Making the Confusion-Matrix.
  GetConfusionMatrix(testLabelData, test, predictions, modelType)
}


############################ XGBoost Model ###################################################



CreateXGBoostModel <- function(train, test,yActual,number = 10, classification = TRUE) 
{
  modelType = "Extreme Gradient Boost"
  #library(tidyverse)
  library(caret)
  library(xgboost)
  
  # Fit the model on the training set
  set.seed(123)
  model <- train(satisfaction_level ~., data = train, method = "xgbTree",
                 trControl = trainControl("cv", number = number))
  # Best tuning parameter
  model$bestTune
  
  #Variabe importance of Model
  varImp(model)
  
  # Make predictions on the test data
  predictions <- predict(model,test)
  
  if (classification == TRUE) {
    # for classification : Compute model prediction accuracy rate
    print(mean(predictions == yActual))
  } else {
    # Compute the average prediction error RMSE
    print(data.frame(RMSE = caret::RMSE(predictions, yActual),R2 = caret::R2(predictions, yActual)))
  }
}

###################################### PCA #######################################################


CreatePCA <- function(data,numComponents) 
{
  print(paste("Creating Principle Components Analysis(PCA)"))
  
  # Creating a data set of numeric variables as PCA is applicable on Numeric data 
  #first create dummies(one hot encoding) for non-numeric Data
  library(dummies)
  dataWithDummies <- dummy.data.frame(hr_data, sep = ".")
  nums <- sapply(dataWithDummies, is.numeric)
  dataNumeric <-dataWithDummies[ , nums]
  
  #principal component analysis
  model <- prcomp(dataNumeric, scale. = T,center = T, rank. = 9)
  print(summary(model))
  #str(model) #look at your PCA object.
  plot(model)
  library(devtools)
  #install_github("vqv/ggbiplot")
  #library(ggbiplot)
  #ggbiplot(model, ellipse = TRUE, obs.scale = 1, var.scale = 1) +
  # scale_colour_manual(name = "Origin", values = c("forest green", "red3", "dark blue")) +
  # ggtitle("PCA") + theme_minimal() + theme(legend.position = "bottom")
  #take first 5 components
  PCADATA <- model$rotation[1:ncol(dataNumeric),1:numComponents]
  print(PCADATA)
  #library(devtools)
  install_github('sinhrks/ggfortify')
  install.packages("ggfortify")
  library(ggfortify); 
  library(ggplot2)
  autoplot(model,shape = FALSE, data=data,label=TRUE,label.size = 1,
           loadings = TRUE, loadings.colour = 'blue',loadings.label = TRUE,
           loadings.label.size = 4,loadings.label.colour="blue")
  # library(ggbiplot)
  # g <- ggbiplot(model, obs.scale = 1, var.scale = 1, 
  #               groups = data.class, ellipse = TRUE, circle = TRUE)
  # g <- g + scale_color_discrete(name = '')
  # g <- g + opts(legend.direction = 'horizontal', 
  #               legend.position = 'top')
  # print(g)
}


######################### Polynomial Regression #########################################

#This is a very specialized function to get the Xi+Xi^2+Xi^3 polynomial
# Note: To be used very carefully for polynomial regression
# Limitation: it takes only one column as ignored column and need to be enhanced in case more 
# columns to be ignored

GetSquareAndCubePolynomialInputs <- function(hr_data, ignoredColumn) 
{
  data_sq<- sapply(hr_data[,ignoredColumn], function(x) x^2)
  data_sq<- as.data.frame(data_sq)
  colnames(data_sq) <- paste(colnames(data_sq), "Square", sep = "_")
  data_cube<- sapply(data[,ignoredColumn], function(x) x^3)
  data_cube<- as.data.frame(data_cube)
  colnames(data_cube) <- paste(colnames(data_cube), "Cube", sep = "_")
  data_poly<- cbind(data,data_sq,data_cube)
  return(data_poly)
}
############################ Step Wise Regression #############################################

CreateStepwiseLinearRegressionModel <- function(hr_data,targetColumnNumber,isPoly=FALSE) 
{
  if(isPoly){
    hr_data <- GetSquareAndCubePolynomialInputs(hr_data,-1)
  }
  #Split data into Train and Test
  train<-TrainTestSplit(hr_data, splitFactor = 0.7, train = TRUE)
  test<-TrainTestSplit(hr_data, splitFactor = 0.7, train = FALSE)
  model <- lm(satisfaction_level ~ ., data = train)
  print(summary(model))
  modelStep<-step(model, direction = "both", trace = 1)
  print(summary(modelStep))
  
  anova(model)
  model$residuals
  #plot residuals
  plot(model$residuals, pch = 16, col = "red")
  
  #The Akaike's information criterion - AIC (Akaike, 1974) 
  AIC(model)  
  #Bayesian information criterion - BIC (Schwarz, 1978)
  BIC(model) 
  
  confint(model)
  
  pred <- predict(model, test, interval = "predict")
  test_Y <- test[,targetColumnNumber]
  print(data.frame(RMSE = caret::RMSE(pred, test_Y),R2 = caret::R2(pred, test_Y)))
  
}






