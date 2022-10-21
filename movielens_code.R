##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2) #added to perform analysis on plots
library(lubridate) #added to perform analysis on dates

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Exploratory data analysis

#checking the size of table and data types of variables
str(edx)

# printing first few rows of the data to check if data is in tidy format
head(edx)

# checking the number of unique users that provided ratings and how many unique movies were rate
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#checking the distribiution of movies and users
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# checking the number of rating for each movie genre 
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


# checking the most popular movies
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#checking how year of rating influences the average rating
#adding the column date
edx <- mutate(edx, date = as_datetime(timestamp))

edx %>% mutate(date = round_date(date, unit = "year")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()+
  ggtitle("Year Rated Effect")


#checking how ratings vary depends on release year
#adding column release_year to edx dataset
edx <- edx %>%
  mutate( release_year = as.numeric(substr(title,
                                        nchar(title)-4,
                                        nchar(title)-1)))

edx %>% 
  group_by(release_year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(release_year, rating)) +
  geom_point() +
  geom_smooth()+
  ggtitle("Release Year Rated Effect")

##########################################################
# Trying different methods to make rating predictions
##########################################################

#definig RMSE as a measurement of accuracy
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm=TRUE))
}

#partitioning data into test and train set
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

## Model 1: simple model
#the least squares estimate of μ, the average of all ratings:
mu_hat <- mean(train_set$rating)
mu_hat

#calculating RMSE for μ on test_set
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

#storing result in the RMSE table
rmse_all<- data_frame(Method = "Just the average", RMSE = naive_rmse)
rmse_all%>% knitr::kable()

## Model 2:  movie effect model
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#predicting ratings
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

#calculating RMSE for μ_m on test_set
rmse_m <- RMSE(predicted_ratings, test_set$rating)
rmse_m

#adding result to rmse_all table
rmse_all <- bind_rows(rmse_all,
                          data_frame(Method="Movie Effect Model",  
                                     RMSE = rmse_m))
rmse_all %>% knitr::kable()

## Model 3: movie and user effect model
#ploting the average rating for user u for those that have rated 100 or more movies
train_set %>%  
  group_by(userId) %>%  
  filter(n()>=100) %>% 
  summarize(b_u = mean(rating)) %>%  
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")+
  ggtitle("Avarage rate given by users who rated more than 100 movies")

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#predicting ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#calculating RMSE for μ_m on test_set
rmse_u <- RMSE(predicted_ratings, test_set$rating)
rmse_u

#adding result to rmse_all table
rmse_all <- bind_rows(rmse_all,
                      data_frame(Method="Movie and User Effect Model",  
                                 RMSE = rmse_u))
rmse_all %>% knitr::kable()

## Model 4: movie, user and release year effect model
ry_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(release_year) %>%
  summarise(b_ry = mean(rating - mu - b_i - b_u, na.rm=TRUE))

#predicting ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(ry_avgs, by='release_year') %>% 
  mutate(pred = mu + b_i + b_u + b_ry) %>%
  pull(pred)

#calculating RMSE for μ_ry on test_set
rmse_ry <- RMSE(predicted_ratings, test_set$rating)
rmse_ry

#adding result to rmse_all table
rmse_all <- bind_rows(rmse_all,
                      data_frame(Method="Movie, User and Realease Year Effect Model",  
                                 RMSE = rmse_ry))
rmse_all %>% knitr::kable()

## Model 5: movie, user, release year and genere effect model
genres_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(ry_avgs, by = 'release_year') %>% 
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu - b_i - b_u - b_ry, na.rm=TRUE))

#predicting ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(ry_avgs, by='release_year') %>% 
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_ry + b_g) %>%
  pull(pred)

#calculating RMSE for μ_ry on test_set
rmse_g <- RMSE(predicted_ratings, test_set$rating)
rmse_g

#adding result to rmse_all table
rmse_all <- bind_rows(rmse_all,
                      data_frame(Method="Movie, User, Realease Year and Genres Effect Model",  
                                 RMSE = rmse_g))
rmse_all %>% knitr::kable()

## Model 6: Regularization of movie, user, release year and genre effects
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%  
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_ry <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    group_by(release_year) %>% 
    summarize(b_ry = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_g <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_ry, by="release_year") %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - b_ry - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>%  
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>%
    left_join(b_ry, by = "release_year") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_ry + b_g) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})


# we plot the lambdas
qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

# we select min RMSE

rmse_r = min(rmses)

#adding result to rmse_all table
rmse_all <- bind_rows(rmse_all,
                      data_frame(Method="Regularization of movie, user and release year effects model",  
                                 RMSE = rmse_r))
rmse_all %>% knitr::kable()

#using the last model with validation set
#adding release_year column to the validation data set
validation <- validation %>%
  mutate( release_year = as.numeric(substr(title,
                                           nchar(title)-4,
                                           nchar(title)-1)))
#calculating RMSEs
lambdas <- seq(0, 10, 0.25)

rmses_v <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%  
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_ry <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    group_by(release_year) %>% 
    summarize(b_ry = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_g <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_ry, by="release_year") %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - b_ry - mu)/(n()+l))
  
  predicted_ratings_v <- 
    validation %>%  
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>%
    left_join(b_ry, by = "release_year") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_ry + b_g) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings_v, validation$rating))
})

#selecting the final RMSE
rmse_v <- min(rmses_v)
rmse_v
