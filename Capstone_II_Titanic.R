#############################################################################
## Capstone Project II - who lives and who dies on the Titanic             ##
#############################################################################

# The objective of the project is to predict whether a passenger on the Titanic
# survived or not based on criteria in the titanic dataset

#Load the test and training sets
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lda)) install.packages("lda", repos = "http://cran.us.r-project.org")
if(!require(titanic)) install.packages("titanic", repos = "http://cran.us.r-project.org")


# The R package contains a pre-made training and test set
# Load the test and training sets
data(titanic_test)
data(titanic_train)

##########################################################################################
# Checking out the data
##########################################################################################

glimpse(titanic_train)

# Checking completeness by checking for nulls (na) and blanks ("")
missing <- colSums(titanic_train == "" | is.na.data.frame(titanic_train))
# Getting the number of missing data in each column
missing
# Getting the percent of data missing in each column
missing/891

# With Cabin missing 77% of the data, I'm going to get rid of it.

# Changing Survived to a factor since it will be used as a categorization.

#titanic_train <- mutate(titanic_train, Survived = as.factor(Survived))

##########################################################################################################
#  Embarked Data Issue
##########################################################################################################

# Getting general populations to see what I should do about the missing Embarked data
pops <- titanic_train %>% group_by(Sex = factor(Sex), Class = factor(Pclass), 
                                   Embarked = factor(Embarked), 
                                   SibSp = factor(if_else(SibSp>0,1,0))) %>% 
                                    summarize(total_num = n())

# Looking at percent of populations excluding Embarked to see if there are any major factors that 
# would indicate the Embarkation point if missing
SexClassE <- inner_join(filter(pops, Embarked != ""), pops %>% 
                          filter(Embarked != "") %>% 
                          group_by(Sex, Class, SibSp) %>% 
                          summarize(tot = sum(total_num)),by = c("Sex", "Class", "SibSp"))

# Seeing if there is a majority with each group
filter(SexClassE %>% 
         group_by(Sex, Class, Embarked, SibSp) %>% 
         summarise(Pct = sum(total_num/tot)),Pct>.5)

# Taking a closer look at 1st class females since they are so close
filter(SexClassE %>% 
         group_by(Sex, Class, Embarked, SibSp) %>% 
         summarise(Pct = sum(total_num/tot)), 
       Sex == "female", Class == 1)


# Conclusion: Althought the 1st class female situation is close, there is a better than 50% chance that 
# under any group of Sex and Pclass that they embarked from S, so it's pretty safe to assign any 
# missing Embarked data to S.  

# It's not at all clear at this point whether the Embarked data is a type of determinant
# so that makes this assumption safer if it isn't.  If it is, I might want to look at this closer.

# So what is the look of the missing data?
pops %>% filter(Embarked == "")

# Great...it's the 51/48 S/C...I'll look at one more thing very quickly - Fares for 1st class females
titanic_train %>% filter(Sex == "female", Pclass == 1) %>% 
  group_by(Embarked) %>% 
  summarize(avg_fare = mean(Fare))

# Okay, I feel better about the assumption in this dataset because the fare indicates an S or Q embarkation and
# Q is very unlikely so S it is.  
# If Embarked does end up being a deciding factor, I'll want to look at something more complex, 
# but I'm not going to spend time on a maybe.
# filling in the null values so I can use caret in fitting
titanic_train <- titanic_train %>% 
  mutate(Embarked = if_else(Embarked == "","S",Embarked))

################################################################################################
# Checking out the data visually
################################################################################################

# Creating labels to use for Survived and Pclass
survived_labels <- as_labeller(c('0' = "perished", '1' = "survived"))
class_labels <- as_labeller(c('1' = "1st class", '2' = "2nd class", '3' = "3rd class"))

# Histogram of the age faceted by Survival
titanic_train %>% ggplot(aes(x = Age), rm.na = TRUE) + 
  geom_histogram(bins = 20) + 
  ggtitle("Distribution of reported age by outcome") +
  facet_grid(~Survived, labeller = survived_labels)

# Looking at the same but for only men.  Wasn't prepared for the emotional impact
titanic_train %>% filter(Sex == 'male') %>% 
  ggplot(aes(x = Age), rm.na = TRUE) + 
  geom_histogram( bins = 20) + 
  ggtitle("Distribution of reported male age by outcome") +
  facet_grid(~Survived, labeller = survived_labels)

# Adding Pclass
titanic_train %>% filter(Sex =='male') %>% 
  ggplot(aes(x = Age), rm.na = TRUE) + 
  geom_histogram(bins = 20) + 
  ggtitle("Distribution of reported male age by class and outcome") +
  facet_grid(Survived ~ Pclass, labeller = 
               labeller(Survived = survived_labels, Pclass = class_labels))

# Male survival rates by Age 
srates <- titanic_train %>% group_by(Sex, Age, Pclass, Survived) %>% 
  summarize(Total=n())
srates %>% filter(Sex == 'male') %>% group_by(Age) %>% 
  summarize(Percent_Survived = sum(Survived)/sum(Total))

srates_percent <- srates %>% filter(Sex == 'male') %>% 
  group_by(Age) %>% 
  summarize(Percent_Survived = sum(Survived)/sum(Total), 
            Total_at_Age = sum(Total))

ggplot(srates_percent, aes(x = Age, y = Percent_Survived)) + 
  geom_col(width = 2, color = "black", na.rm = TRUE) + 
  ggtitle("Percent surviving males by reported age")


################################################################################################
# Title might give additional information of gender and marital status
# Title is basically between first comma(,) and first dot (.)
# I will extract Title using Regex gsub()

titanic_train <- titanic_train %>% 
  mutate(Title = gsub('(.*, )|(\\..*)','', titanic_train$Name))

table(titanic_train$Title)

# Looking at survival rates by Title
titanic_train %>% group_by(Title) %>% summarize(lived = sum(Survived), 
                                                perished = (n()-lived), pct_survived = (lived/n()*100)) %>% knitr::kable()
# Looking at survival rates by Sex
titanic_train %>% group_by(Sex) %>% summarize(lived = sum(Survived), 
                                              perished = (n()-lived), pct_survived = (lived/n()*100)) %>% knitr::kable()

################################################################################################
# Checking out a baysian model 

if(!require(rstan)) install.packages("rstan", repos = "http://cran.us.r-project.org")
if(!require(rstanarm)) install.packages("rstanarm", repos = "http://cran.us.r-project.org")
if(!require(bayestestR)) install.packages("bayestestR", repos = "http://cran.us.r-project.org")

# Starting Title, Age, Sex, Pclass, Fare, Embarked, SibSp, and Parch
model <- stan_glm(Survived ~ Title + Age + Sex + Pclass + Fare + Embarked + SibSp + Parch, 
                  refresh = 0, data = titanic_train)
equivalence_test(model)
plot(equivalence_test(model))

# Removing Age and Fare
model <- stan_glm(Survived ~ Title + Sex + Pclass + Embarked + SibSp + Parch, 
                  refresh = 0, data = titanic_train)
equivalence_test(model)
plot(equivalence_test(model))

# Removing Embarked
model <- stan_glm(Survived ~ Title + Sex + Pclass + SibSp + Parch, 
                  refresh = 0, data = titanic_train)
equivalence_test(model)
plot(equivalence_test(model))

# Removing Sex
model <- stan_glm(Survived ~ Title + Pclass, 
                  refresh = 0, data = titanic_train)
equivalence_test(model)
plot(equivalence_test(model))

# I'm going to fit the titles into categories - Men and all others including boys
# The idea behind this is the culture of the time.  Men were expected to sacrafice for women and children
# If this actually played out on the Titanic, it should show as a simple binary factor
# The logic is if male and title does not equal master, then men, otherwise not men.

titanic_train <- titanic_train %>% mutate(adultMale = 
                                            ifelse(Sex == "male" & Title != "Master",1,0))

# loading the adultMale factor into the model
glm_model <- stan_glm(Survived ~ adultMale + Pclass + SibSp + Parch, refresh = 0, 
                      data = titanic_train)
equivalence_test(glm_model)
plot(equivalence_test(glm_model))

# Adding back Embarked
glm_model <- stan_glm(Survived ~ adultMale + Pclass + SibSp + Parch + Embarked, refresh = 0, 
                      data = titanic_train)
equivalence_test(glm_model)
plot(equivalence_test(glm_model))

##################################################################################################
# Fitting and testing
##################################################################################################

# Setting a baseline at Sex as the only determinant 
fit_glm <- glm(Survived ~ Sex, data = titanic_train)
p_hat_glm <- predict(fit_glm, titanic_train)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 1, 0))

x_result <- tibble(model = "Sex Only", method = "glm", 
                   accuracy = confusionMatrix(data = y_hat_glm, reference = factor(titanic_train$Survived))$overall["Accuracy"])


# Fitting the model
fit_glm <- glm(Survived ~ adultMale + Pclass + SibSp + Parch + Embarked, data = titanic_train)
p_hat_glm <- predict(fit_glm, titanic_train)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 1, 0))
x_result <- add_row(x_result, model = "final", method = "glm", 
                    accuracy = confusionMatrix(data = y_hat_glm, reference = factor(titanic_train$Survived))$overall["Accuracy"])

# using caret to find the best method

methods_lst <- c("rf", "parRF", "ranger", "bayesglm", "blackboost", "svmPoly")

# Creating a function to run through the different methods
itterate_methods <- function(x) {
  set.seed(50)
  train_x <- train(factor(Survived) ~ adultMale + Pclass + SibSp + Parch + Embarked,  method = methods_lst[x],  data = titanic_train)
    p_hat_x <- predict(train_x, titanic_train)
}

# Loop to run through all the methods
for(i in 1:length(methods_lst)) {
    p_hat_x <- itterate_methods(i)
    x_result <- add_row(x_result, model = "final", method = methods_lst[i], accuracy = 
                          confusionMatrix(data = p_hat_x, reference = factor(titanic_train$Survived))$overall["Accuracy"])
  }


# Looking at the results
x_result %>% knitr::kable()

# Conclusion
### The best I could get was .842, and I got this with the "rf" model so I'm going to go with that for the final solution

##################################################################################################
# Going for a solution
##################################################################################################

#### Data duties for the test set ################################################################

# 1. Adding the title
titanic_test <- titanic_test %>% mutate(Title = gsub('(.*, )|(\\..*)','', titanic_test$Name))
# Checking out the result
table(titanic_test$Title)
# 2. Adding the adultMale determiner
titanic_test <- titanic_test %>% mutate(adultMale = ifelse(Sex == "male" & Title != "Master",1,0))

# Look for missing data in other determinants
table(titanic_test$Embarked)
table(titanic_test$Pclass)
table(titanic_test$SibSp)
table(titanic_test$Parch)
# Wow, nothing is missing.  Good to go!

### Fitting the model ##########################################################################

# Using the decided on rf model
train_rf <- train(factor(Survived) ~ adultMale + Pclass + SibSp + Parch + Embarked,  method = "rf", data = titanic_train)
p_hat_rf <- predict(train_rf, titanic_test)
solution <- data.frame(Survived = p_hat_rf, PassengerId = titanic_test$PassengerId)

write.csv(solution, file = 'rf_model_sol.csv', row.names = F)

#Kaggle returned a score of 0.79425 and a rank of 2,214 out of 11,322  
#The base Sex Only base case returned a score of 0.76555

improvement_over_base <- 1-(.76555/.79425)
ranking_in_top_percent <- 2214/11322
improvement_over_base
ranking_in_top_percent
