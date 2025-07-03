#Author: Marcos Paulo Pazzinatto
#Load required libraries
#Here we import the necessary libs for the script to work.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)

#Download and prepare the MovieLens dataset
#We downloaded it directly from the grouplens.org website
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(ratings_file)) unzip(dl, files = ratings_file)
if(!file.exists(movies_file)) unzip(dl, files = movies_file)

#Here we perform a split to generate the columns for the ratings
ratings <- read_lines(ratings_file) %>%
  str_split("::", simplify = TRUE) %>%
  as.data.frame(stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

#Here we perform a split to generate the columns for the movies
movies <- read_lines(movies_file) %>%
  str_split("::", simplify = TRUE) %>%
  as.data.frame(stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

#This part is important, as we perform the join between ratings and movies
movielens <- left_join(ratings, movies, by = "movieId")

#Set a seed so that the split is reproducible (yields the same results every time the code is run).
set.seed(1, sample.kind = "Rounding")
#Creates a 10% partition of the data (p = 0.1) to serve as a test. The selected indices go to the final_holdout_test.
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
#The remaining 90% of the data becomes edx — the training (and internal validation) set.
#Temp receives the 10% previously separated — it will still be filtered to ensure integrity.
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
#Filters the final_holdout_test to keep only movies and users that already exist in edx.
#This is essential to avoid “unknown data” when predicting, which the model has never seen.
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
#Collects records removed from temp because they do not meet the above criteria (new user or movie).
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
#Clears temporary objects from memory to keep the environment lighter.
#This is very important, as it frees up memory on the server that is processing the data.
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Global Average Model
#This baseline model predicts all ratings using the overall average rating (mu).
#It does not consider user or movie-specific differences.
#Calculate the global average rating across all observations in the training set.
mu <- mean(edx$rating)
#Create a vector of predictions, repeating the average for every row in the dataset.
predictions_global <- rep(mu, nrow(edx))
#Compute the Root Mean Square Error (RMSE) between actual ratings and global average predictions.
#This serves as a reference point to evaluate more complex models.
rmse_global <- sqrt(mean((edx$rating - predictions_global)^2))

#Movie Effect Model
#This model refines predictions by incorporating movie-specific bias.
#It assumes that some movies are generally rated higher or lower than the global average.
#For each movie, calculate the average deviation (b_i) from the global average rating (mu).
#This captures how much a movie tends to be rated above or below average.
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
#Generate predictions by adding the movie-specific bias to the global average.
#This means we are adjusting the global mean based on each movie's tendency.
predicted_ratings_movie <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i)
#Compute RMSE to measure prediction accuracy using the movie effect model.
rmse_movie <- sqrt(mean((edx$rating - predicted_ratings_movie$pred)^2))

#Movie + User Effect Model

#Calculate the average user bias (b_u)
#Join the movie effect (b_i) to the original dataset using movieId
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
#Group the data by userId  
  group_by(userId) %>%
#Calculate the average difference between the actual rating and the expected rating (mu + b_i)
#This gives us the user-specific effect (b_u)
  summarize(b_u = mean(rating - mu - b_i))
#Predict ratings including both movie and user effects
predicted_ratings_user <- edx %>%
#Join the movie effect (b_i)
  left_join(movie_avgs, by = "movieId") %>%
#Join the user effect (b_u)
  left_join(user_avgs, by = "userId") %>%
#Compute the predicted rating using the formula: mu + b_i + b_u
  mutate(pred = mu + b_i + b_u)
#Evaluate the performance of the model
#Compute the Root Mean Square Error (RMSE) between actual and predicted ratings
rmse_movie_user <- sqrt(mean((edx$rating - predicted_ratings_user$pred)^2))

#Regularized Movie + User Effect Model

#Define a custom function to calculate RMSE (Root Mean Square Error)
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

#Define a range of lambda values to test for regularization
lambdas <- seq(4, 6, 0.05)

#For each lambda, calculate the RMSE using regularized movie and user effects
rmse_results <- sapply(lambdas, function(l) {
  
#Compute regularized movie effect b_i
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l), .groups = "drop")
  
#Compute regularized user effect b_u
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + l), .groups = "drop")
  
#Generate predictions using mu + b_i + b_u
  predicted_ratings <- edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
#Return the RMSE for this lambda
  RMSE(edx$rating, predicted_ratings)
})

# Select the lambda value that gives the smallest RMSE
lambda <- lambdas[which.min(rmse_results)]

# Recompute b_i and b_u using the best lambda

# Regularized movie effect
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda), .groups = "drop")

# Regularized user effect
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda), .groups = "drop")

# Predict ratings using the best regularized model
predicted_ratings <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Compute final RMSE with the optimal lambda
rmse_regularized <- RMSE(edx$rating, predicted_ratings)


# Final RMSE on holdout set
predicted_final <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
final_rmse <- RMSE(final_holdout_test$rating, predicted_final)
final_rmse