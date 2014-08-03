###############################################################
## Code for Applied Predictive Modeling in R by Max Kuhn 2014
## 

###############################################################
## Package installs:

install.packages(c("caret", "pROC", "rpart", "partykit", 
                   "C50", "kernlab", "AppliedPredictiveModeling",
                   "earth", "mda", "nnet"),
                 dependencies = c("Depends", "Imports", "Suggests"))

###############################################################
## Slide 21: Illustrative Data: Image Segmentation

library(caret)
data(segmentationData)
# get rid of the cell identifier
segmentationData$Cell <- NULL

training <- subset(segmentationData, Case == "Train")
testing  <- subset(segmentationData, Case == "Test")

training$Case <- NULL
testing$Case <- NULL

###############################################################
## Slide 22: Illustrative Data: Image Segmentation

str(training[,1:9])

cell_lev <- levels(testing$Class)


###############################################################
## Slide 34: Centering and Scaling

trainX <- training[, names(training) != "Class"]
## Methods are "BoxCox", "YeoJohnson", center", "scale",
## "range", "knnImpute", "bagImpute", "pca", "ica",
## "spatialSign", "medianImpute", "expoTrans"
preProcValues <- preProcess(trainX, method = c("center", "scale"))
preProcValues


###############################################################
## Slide 35: Centering and Scaling

scaledTrain <- predict(preProcValues, trainX)


###############################################################
## Slide 57: An Example

library(rpart)
rpart1 <- rpart(Class ~ ., data = training, 
                control = rpart.control(maxdepth = 2))
rpart1


###############################################################
## Slide 58: Visualizing the Tree

librry(partykit)
rpart1a <- as.party(rpart1)
plot(rpart1a)


###############################################################
## Slide 61: The Final Tree

rpartFull <- rpart(Class ~ ., data = training)


###############################################################
## Slide 62: The Final Tree

rpartFull

###############################################################
## Slide 64: The Final rpart Tree

rpartFulla <- as.party(rpartFull)
plot(rpartFulla)


###############################################################
## Slide 65: Test Set Results

rpartPred <- predict(rpartFull, testing, type = "class")
confusionMatrix(rpartPred, testing$Class)   # requires 2 factor vectors


###############################################################
## Slide 72: The train Function

cvCtrl <- trainControl(method = "repeatedcv", repeats = 5,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE)
set.seed(1)
rpartTune <- train(Class ~ ., data = training, 
                   method = "rpart",
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = cvCtrl)


###############################################################
## Slide 73: train Results

rpartTune


###############################################################
## Slide 75: Resampled ROC Profile

trellis.par.set(caretTheme())
plot(rpartTune, scales = list(x = list(log = 10)))

###############################################################
## Slide 76: Resampled ROC Profile

ggplot(rpartTune) +scale_x_log10()


###############################################################
## Slide 78: Test Set Results

rpartPred2 <- predict(rpartTune, testing)
confusionMatrix(rpartPred2, testing$Class)


###############################################################
## Slide 79: Predicting Class Probabilities

rpartProbs <- predict(rpartTune, testing, type = "prob")
head(rpartProbs)


###############################################################
## Slide 80: Creating the ROC Curve

library(pROC)
rpartROC <- roc(testing$Class, rpartProbs[, "PS"], 
                levels = rev(cell_lev))
plot(rpartROC, type = "S", print.thres = .5)


###############################################################
## Slide 91: Tuning the C5.0 Model

grid <- expand.grid(model = "tree",
                    trials = c(1:100),
                    winnow = FALSE)
set.seed(1)
c5Tune <- train(trainX, training$Class,
                method = "C5.0",
                metric = "ROC",
                tuneGrid = grid,                    
                trControl = cvCtrl)


###############################################################
## Slide 92: Model Output

c5Tune

###############################################################
## Slide 93: Boosted Tree Resampling Profile

ggplot(c5Tune)

###############################################################
## Slide 94: Test Set Results

c5Pred <- predict(c5Tune, testing)
confusionMatrix(c5Pred, testing$Class)


###############################################################
## Slide 95: Test Set ROC Curve

c5Probs <- predict(c5Tune, testing, type = "prob")
head(c5Probs, 3)

library(pROC)
c5ROC <- roc(predictor = c5Probs$PS,
             response = testing$Class,
             levels = rev(levels(testing$Class)))


###############################################################
## Slide 96: Test Set ROC Curve

c5ROC

plot(rpartROC, type = "S")
plot(c5ROC, add = TRUE, col = "#9E0142")

###############################################################
## Slide 98: Test Set Probabilities

histogram(~c5Probs$PS|testing$Class, xlab = "Probability of Poor Segmentation")

###############################################################
## Slide 109: SVM Example 

set.seed(1)
svmTune <- train(x = trainX,
                 y = training$Class,
                 method = "svmRadial",
                 # The default grid of cost parameters go from 2^-2,
                 # 0.5 to 1, 
                 # We'll fit 9 values in that sequence via the tuneLength
                 # argument.
                 tuneLength = 9,
                 preProc = c("center", "scale"),
                 metric = "ROC",   
                 trControl = cvCtrl)



###############################################################
## Slide 110: SVM Example

svmTune


###############################################################
## Slide 111: SVM Example

svmTune$finalModel


###############################################################
## Slide 112: SVM ROC Profile

plot(svmTune, metric = "ROC", scales = list(x = list(log = 2))))


###############################################################
## Slide 113: Test Set Results

svmPred <- predict(svmTune, testing[, names(testing) != "Class"])
confusionMatrix(svmPred, testing$Class)

###############################################################
## Slide 114: Test Set ROC Curves

svmROC <- roc(testing$Class, 
              predict(svmTune, testing[,-1], type = "prob")[, "PS"],
              levels = rev(levels(testing$Class)))
plot(rpartROC, type = "S")
plot(c5ROC, add = TRUE, col = "#9E0142")
plot(svmROC, add = TRUE, col = "#3288BD")

legend(.5, .5, c("CART", "C5.0", "SVM"), 
       col = c("black", "#9E0142", "#3288BD"), 
       lty = rep(1, 3))



###############################################################
## Slide 117: A Few Other Models 

set.seed(1)
fdaTune <- train(Class ~ ., data = training, 
                 method = "fda",
                 tuneLength = 12,
                 metric = "ROC",
                 trControl = cvCtrl)

set.seed(1)
plrTune <- train(Class ~ ., data = training, 
                 method = "multinom",
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(decay = c(0.1, 1, 10, 20, 40)),
                 trace = FALSE, maxit = 1000,
                 metric = "ROC",
                 trControl = cvCtrl)


###############################################################
## Slide 118: Collecting Results With resamples

cvValues <- resamples(list(CART = rpartTune, SVM = svmTune, 
                           C5.0 = c5Tune, FDA = fdaTune, 
                           logistic = plrTune))



###############################################################
## Slide 119: Collecting Results With resamples

summary(cvValues)


###############################################################
## Slide 120: Visualizing the Resamples

library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
splom(cvValues, metric = "ROC", pch = 16, 
      cex = .7, col = rgb(.2, .2, .2, .4), 
      pscales = 0)


###############################################################
## Slide 122: Visualizing the Resamples

trellis.par.set(caretTheme())
dotplot(cvValues, metric = "ROC")

###############################################################
## Slide 123: Comparing Models

rocDiffs <- diff(cvValues, metric = "ROC")
summary(rocDiffs)


###############################################################
## Slide 124: Visualizing the Differences

trellis.par.set(caretTheme())
dotplot(rocDiffs, metric = "ROC")



###############################################################
## Slide 128: Clustering the Models

plot(caret:::cluster.resamples(cvValues), sub = "", main = "")


###############################################################
## Slide 125: foreach and caret

## library(doMC)
## registerDoMC(cores = 2)

