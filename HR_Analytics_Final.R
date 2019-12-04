getwd()

source("HR Analytics.R")

dataOriginal <- read.csv("turnover.csv")
dim(dataOriginal)

data_hr <- dataOriginal

#ls()
#rm(data)
# Compute the attrition

dataAttritionFreq <- dataOriginal
attrition <- as.factor(dataAttritionFreq$left)
summary(attrition)

AttritionRate <- sum(dataAttritionFreq$left / length(dataAttritionFreq$left)*100)
print(paste("Attrition Rate =",round(AttritionRate,2),"%"))

library(ggplot2);library(ggpubr);library(dplyr)
theme_set(theme_pubr())

dataAttritionFreq$left <- ifelse(dataAttritionFreq$left=='1',"Left","Stay")

df <- dataAttritionFreq %>% group_by(left) %>% summarise(counts=n())
df

# Plot The Result of Left employees

ggplot(df, aes(x = left, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.1) + 
  theme_pubclean()


#############[   Basic EDA    ]#################################################

PercentLeft <- function(X,Y,colnames,color){
  library(ggplot2)
  DataWithLeft<-as.data.frame(prop.table(table(X,Y, dnn=c("Var1","Left")), 1))
  DataWithLeft <- subset(DataWithLeft, Left==1)
  names(DataWithLeft) <- c("Var1","Left","PercentLeft")
  DataWithLeft$PercentLeft <- round(DataWithLeft$PercentLeft * 100.00,2)
  ggplot(DataWithLeft, aes(x=reorder(Var1, -PercentLeft),y=PercentLeft,fill=Left)) +
    geom_bar(stat='identity', fill=color) +
    geom_text(aes(label=PercentLeft), vjust=0) + 
    ggtitle("---- Bar char for Left percenatages ----") + xlab(colnames) + 
    ylab("% of Employees Left ----> ")
}

#PercentLeft(data$salary,data$left, "<-------Salary--------->", "Red")
PercentLeft(hr_data$salary,hr_data$left, "<-------Salary--------->", "Red")
PercentLeft(hr_data$sales,hr_data$left, "<-------Department--------->", "Blue")
PercentLeft(hr_data$Work_accident,hr_data$left,"Work Accident : 0=No Accident 1=Encountered accident", "Green")
PercentLeft(hr_data$promotion_last_5years,  hr_data$left,
            "Promotion : 0=Not Promoted 1=Promoted", "Orange")




#Prepare Data of employess Left and Stayed 

LeftData <- subset(hr_data, left == 1)
StayData <- subset(hr_data, left == 0)

ggplot() + geom_density(aes(x = satisfaction_level), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = satisfaction_level), colour = "blue" , data = StayData)

ggplot() + geom_density(aes(x = last_evaluation), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = last_evaluation), colour = "blue" , data = StayData)

ggplot() + geom_density(aes(x = number_project), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = number_project), colour = "blue" , data = StayData)

ggplot() + geom_density(aes(x = average_montly_hours), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = average_montly_hours), colour = "blue" , data = StayData)

ggplot() + geom_density(aes(x = time_spend_company), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = time_spend_company), colour = "blue" , data = StayData)

ggplot() + geom_density(aes(x = salary), colour = "red"  , data = LeftData) + 
  geom_density(aes(x = salary), colour = "blue" , data = StayData)


################[ Predictive Modeling ]######################

#convert salary into numeric low=1, medium=2, High=3
hr_data$salary <- ifelse(hr_data$salary=='low',1,
                         ifelse(hr_data$salary=='medium',2,
                                ifelse(hr_data$salary=='high',3,0)))

#Print correlation heat-map; 

PrintCorrelationHeatMap <- function(hr_data){
  
  #Corellation matrix
  library(reshape2)
  library(ggplot2)
  
  melted_cormat <- melt()
  
  ggplot(melted_cormat,aes(Var2,Var1,fill=value)) +
    geom_tile(color="black") + 
    scale_fill_gradient(low = "red",medium="blue",limit=c(-1,1),name="Correlation") +
    theme(axis.text.x = element_text(angle = 10,hjust = 0.5))
  
  
}


PrintCorrelationHeatMap(hr_data[,-9]) #remove department as it is still category

#Create one-hot-encoding for categorical data (Department)
install.packages("dummies")
library(dummies)
dataWithDummies <- dummy.data.frame(hr_data, sep = ".")

#Normalize the data by diving max(x)
source("/Users/binay/Desktop/Rajesh/Analytics Edge/HR Analytics.R")
data_norm <- as.data.frame(lapply(dataWithDummies, normalize_DivideByMax))


#Final Data
dataFinal<- data_norm
dataFinal$left <- factor(dataFinal$left) #required for random forest and Navie Bayes

#Split data into Train and Test
train<-TrainTestSplit(dataFinal, splitFactor = 0.7, train = TRUE)
test<-TrainTestSplit(dataFinal, splitFactor = 0.7, train = FALSE)

write.csv(train,file = "/Users/binay/Desktop/Rajesh/Analytics Edge/train_final.csv")
write.csv(test,file = "/Users/binay/Desktop/Rajesh/Analytics Edge/test_final.csv")


################## Prepare Data for Models ##################

trainLabel<- train$left
numericIndeDependentVariables <-paste("satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+salary")
departments <-  paste("Department.accounting+Department.hr+Department.IT+Department.management+Department.marketing+Department.product_mng+Department.RandD+Department.sales+Department.support+Department.technical")
inputVariables <- paste(numericIndeDependentVariables, "+", departments)

CreateLogisticRegressionModel (trainLabel, test$left, inputVariables, train, test)

###############################################################################################



#To get maximum benefit of decision Tree, we need original data and convert numeric left to 
# "Still_Active" and "left". Also we wont use department for decision tree as from 
# logistic regression, we came to know that department is not that important variable
dataDecisionTree <- hr_data[,-9]
dataDecisionTree$left <- ifelse(dataDecisionTree$left == '0',"Still_Active",
                                ifelse(dataDecisionTree$left == "1",'Left',"NA"))
#Split data into Train and Test
trainDecisionTree <- TrainTestSplit(dataDecisionTree, splitFactor = 0.7, train = TRUE)
testDecisionTree <- TrainTestSplit(dataDecisionTree, splitFactor = 0.7, train = FALSE)

CreateDecisionTreeModel ( "trainDecisionTree$left", 
                          testDecisionTree$left, numericIndeDependentVariables, 
                          trainDecisionTree, testDecisionTree)


##################################################

CreateRandomForestModel( trainLabel, test$left, inputVariables, train, test, 50)

##################################################

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


CreateKernalSvmModel ( trainLabel, test$left, inputVariables, train, test    )

##################################################

CreateNaiveBayesModel( trainLabel, test$left, inputVariables, train, test    )

##################################################

CreateDeepnetNNModel(train,test,targetColumnNumber=7,hiddenLayers=c(50, 30, 10),numepochs =700)

##################################################

CreateKerasNNModel(train,test,targetColumnNumber=7,batchSize=128,numepochs=25,validationSplit=0.2,
                   lossFunction= "categorical_crossentropy",errorMetrics= "accuracy")

##################################################

CompareModelsAndPlotCombinedRocCurve()


#########   Principle Components Analysis #######################################
#for PCA, remove the department

CreatePCA(data[,-9], numComponents=3)

#######################  Models for Satisfaction_level #############################
#make of copy of original Data
data<-dataOriginal

#convert salary into numeric low=1, medium=2, High=3

data$salary <-ifelse(data$salary == 'low',1, ifelse(data$salary == 'medium', 2,
                                                    ifelse(data$salary == 'high', 3, 0)))

#eliminate left and department


dataFinal <- data[,-c(7,9)]

#Split data into Train and Test
train<-TrainTestSplit(dataFinal, splitFactor = 0.7, train = TRUE)
test<-TrainTestSplit(dataFinal, splitFactor = 0.7, train = FALSE)

#Prepare Data for Models
trainLabel  <- "train$satisfaction_level"
test_Y      <- test$satisfaction_level
inputVariables <-paste("last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+salary")


#without polynomial Regression
CreateStepwiseLinearRegressionModel(dataFinal,targetColumnNumber=1,isPoly=FALSE) 

#With Polynomial regression
CreateStepwiseLinearRegressionModel(dataFinal,targetColumnNumber=1,isPoly=TRUE)  

#using XGBoost
CreateXGBoostModel(train,test,test$satisfaction_level,number=10,classification=FALSE) 

