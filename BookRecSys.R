########################################################################
################## BOOK RECOMMENDATION SYSTEM PROJECT ################## 
##########################  Reza Ameri, Ph.D. ########################## 
########################################################################

########################## Description of the Project

# Recommender systems have become prevalent in recent years as they tackle the problem of 
# information overload by suggesting the most relevant products to end users. In fact, 
# recommender systems are information filtering tools that aspire to predict the rating 
# for users and items, predominantly from big data to recommend their likes. Specifically, 
# book recommendation systems provide a mechanism to assist users in classifying users with 
# similar interests. In this project, exploratory data analysis is used in order to develop 
# various machine learning algorithms that predict book ratings with reasonable accuracy. 
# The project is part of the capstone for the professional certificate in data science 
# program at Harvard University. The Book-Crossing Dataset that includes 1,149,780 ratings 
# (explicit/implicit) about 271,379 books is used for creating a book recommendation system. 

########################## Book-Crossing Dataset

# Create train and validation sets

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# Book-Crossing Dataset
# http://www2.informatik.uni-freiburg.de/~cziegler/BX/

dl <- tempfile()
download.file("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "BX-Book-Ratings.csv"))),
                 col.names = c("userId", "ISBN", "rating"))

books <- fread(text = gsub("::", "\t", readLines(unzip(dl, "BX-Books.csv"))),
               col.names = c("ISBN", "BookTitle", "BookAuthor","YearOfPublication",
                             "Publisher", "ImageURLS", "ImageURLM", "ImageURLL"))

bookcrossing <- left_join(ratings, books, by = "ISBN")
bookcrossing <- bookcrossing[, 1:(length(bookcrossing)-3)] # Remove the URLs
bookcrossing <- bookcrossing[complete.cases(bookcrossing), ] # Consider complete cases only
bookcrossing <- bookcrossing[apply(bookcrossing!=0, 1, all),] # Consider explicit ratings only

# Validation set will be 10% of bookcrossing data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = bookcrossing$rating, times = 1, p = 0.1, list = FALSE)
edx <- bookcrossing[-test_index,]
temp <- bookcrossing[test_index,]

# Make sure userId and ISBN in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "ISBN") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, books, test_index, temp, bookcrossing, removed)

########################## Data Processing

# Structure of the dataset
str(edx)

# Headers of the dataset
head(edx) %>%
  print.data.frame()

# Summary of the dataset
summary(edx)

# Ratings per book distribution 
edx %>%
  count(ISBN) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, color="black", fill="blueviolet") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Books") +
  ggtitle("Distribution of Book Ratings") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# User ratings distribution 
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, color="black", fill="blueviolet") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Users") +
  ggtitle("Distribution of Users Ratings") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Star ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_bar(color="black", fill="blueviolet") +
  scale_x_discrete(limits = c(seq(1,10))) +
  xlab("Rating") +
  ylab("Frequency") +
  ggtitle("Distribution of Star Ratings") + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Average book ratings by users
edx %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(binwidth = 1, color="black", fill="blueviolet") +
  xlab("Average Star Ratings") +
  ylab("Number of Users") +
  ggtitle("Average Book Ratings by Users") +
  scale_x_discrete(limits = c(seq(1,10))) + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Distribution of rated books based on the publication date
dat <- edx %>% filter(edx$YearOfPublication > 1950 & edx$YearOfPublication < 2005)

dat %>%
  ggplot(aes(dat$YearOfPublication)) +
  geom_histogram(binwidth = 0.5, color="black", fill="blueviolet") +
  xlab("Publication Date") +
  ylab("Frequency") + 
  ggtitle("Average Rating Based on Publication Date") + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Number of publications for different publishers
mydata <- edx %>% 
  group_by(Publisher) %>%
  summarise(count = n()) 

mydata <- mydata[order(-mydata$count),]

mydata %>% 
  filter(count > 2000) %>% 
  ggplot(aes(Publisher)) + 
  geom_bar(aes(weights=count), color="black", fill="blueviolet") +
  xlab("Publisher") +
  ylab("Number of Publications") + 
  ggtitle("Number of Publications for Different Publishers") + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(angle = 90, vjust = 0.5, hjust=1, size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

########################## Methodology and Procedure

RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

############## Method I: Average Book Rating

mu <- mean(edx$rating)
cat("mu = ", mu)

naive_rmse <- RMSE(validation$rating, mu)
cat("RMSE = ", naive_rmse)

rmse_results <- data_frame(Method = "Method I: Average Book Rating", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

############## Method II:  Book Effect Model

book_avgs <- edx %>%
  group_by(ISBN) %>%
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + validation %>%
  left_join(book_avgs, by='ISBN') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method II: Book Effect Model", 
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

############## Method III: Book & User Effect Model

user_avgs <- edx %>%
  left_join(book_avgs, by='ISBN') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation%>%
  left_join(book_avgs, by='ISBN') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method III: Book & User Effect Model",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

############## Method IV: Regularized Book & User Effect Model

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(ISBN) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="ISBN") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "ISBN") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))})

qplot(lambdas, rmses) + xlab("Lambda") + ylab("RMSE") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method IV: Regularized Book & User Effect Model"
                                     , RMSE = min(rmses)))
rmse_results %>% knitr::kable()

############## Method V: Parallel Matrix Factorization Model

edx_factorization <- edx %>% select(ISBN, userId, rating)

validation_factorization <- validation %>% select(ISBN, userId, rating)

edx_factorization <- as.matrix(edx_factorization)
validation_factorization <- as.matrix(validation_factorization)

write.table(edx_factorization, file = "trainingset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

write.table(validation_factorization, file = "validationset.txt", sep = " ", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

set.seed(1)
training_dataset <- data_file("trainingset.txt")

validation_dataset <- data_file("validationset.txt")

r = Reco() # this will create a model object

opts = r$tune(training_dataset, opts = list(dim = c(10, 20, 30), lrate = 
                                              c(0.1, 0.2), costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 10))

r$train(training_dataset, opts = c(opts$min, nthread = 1, niter = 20))
stored_prediction = tempfile() 

r$predict(validation_dataset, out_file(stored_prediction))
real_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
pred_ratings <- scan(stored_prediction)

model_5_rmse <- RMSE(real_ratings, pred_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method V: Parallel Matrix Factorization Model", 
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()
