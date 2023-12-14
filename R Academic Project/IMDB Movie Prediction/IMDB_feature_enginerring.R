`%ni%` <- Negate(`%in%`)
library(dplyr)
library(ggplot2)
library(cowplot)
View(imdb_data)

# Load data
imdb_data <- read.csv("C:/Users/xutian/Downloads/IMDB_data_Fall_2023.csv")

# Drop aspect_ratio
imdb_data <- imdb_data[,colnames(imdb_data) %ni% "aspect_ratio"]

# Only keep English movies, US and UK films. Dummy the country 
imdb_data <- imdb_data %>% filter(country %in% c("UK","USA"), language == "English")
imdb_data[,"country_dummy_USA"] <- as.numeric(imdb_data[,"country"] == "	
USA")
imdb_data[,"country_dummy_UK"] <- as.numeric(imdb_data[,"country"] == "	
UK")
# imdb_data <- imdb_data[,colnames(imdb_data) %ni% "country"]

# Drop release_day
imdb_data <- imdb_data[,colnames(imdb_data) %ni% "release_day"]

# Build academy months feature (sep, oct, nov), basically a dummy variable
imdb_data[,"academy_months"] <- as.numeric(imdb_data[,"release_month"] %in% c("Sep","Oct","Nov"))

# Transform year into (2023-year)
imdb_data[,"years_released"] <- 2023 - imdb_data[,"release_year"] + 1

# Bin the year by decade like the duration
imdb_data[,"years_released_decade"] <- ceiling(imdb_data[,"years_released"]/10)

# Standardize movie_budget
imdb_data[,"movie_budget_norm1"] <- scale(imdb_data[,"movie_budget"], center = TRUE, scale = TRUE)
## Another optional way to standardize (will result in different data of movie_budget*year_released)
imdb_data[,"movie_budget_norm2"] <- imdb_data[,"movie_budget"] / max(imdb_data[,"movie_budget"])

# Add an interactive variable: (2023-year) * standardized(movie_budget)
imdb_data[,"movie_budget_norm1_1"] <- imdb_data[,"years_released"] * imdb_data[,"movie_budget_norm1"]
imdb_data[,"movie_budget_norm1_2"] <- imdb_data[,"years_released_decade"] * imdb_data[,"movie_budget_norm1"]
## If we choose th standardize the movie_budget by dividing maximum, we need to use the following two lines
imdb_data[,"movie_budget_norm2_1"] <- imdb_data[,"years_released"] * imdb_data[,"movie_budget_norm2"]
imdb_data[,"movie_budget_norm2_2"] <- imdb_data[,"years_released_decade"] * imdb_data[,"movie_budget_norm2"]

# Check the distribution of the interactive variable 
# years_released or years_released_decade * movie_budget (two measures of standardization)
p1 <- ggplot(imdb_data, aes(x = factor(years_released), y = movie_budget_norm1_1)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p2 <- ggplot(imdb_data, aes(x = factor(years_released_decade), y = movie_budget_norm1_2)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p3 <- ggplot(imdb_data, aes(x = factor(years_released), y = movie_budget_norm2_1)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p4 <- ggplot(imdb_data, aes(x = factor(years_released_decade), y = movie_budget_norm2_2)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
plot_grid(p1, p2, p3, p4, nrow = 2)
# imdb_data <- imdb_data[,colnames(imdb_data) %ni% c("release_year","movie_budget")]

# Whether to drop nb_news?
# Use log transformation, and test the heteroscedasticity again
imdb_data[,"nb_news_articles_log"] <- log2(imdb_data[,"nb_news_articles"] + 1) 

#### We would better to delete the outliers (e.g., star war) to do this step. ##################
# Check the distribution of the log transformation of nb_news
p1 <- ggplot(imdb_data, aes(x = factor(years_released), y = nb_news_articles)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p2 <- ggplot(imdb_data, aes(x = factor(years_released_decade), y = nb_news_articles)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p3 <- ggplot(imdb_data, aes(x = factor(years_released), y = nb_news_articles_log)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
p4 <- ggplot(imdb_data, aes(x = factor(years_released_decade), y = nb_news_articles_log)) +
  geom_boxplot() +
  theme_cowplot() +
  theme(axis.text.x = element_text(angle = 30, hjust = 0.85, vjust = 0.75))
plot_grid(p1,p2,p3,p4,nrow = 2)
# imdb_data <- imdb_data[,colnames(imdb_data) %ni% c("nb_news_articles")]

# Bin the duration. (0-100) (100-150) (>150). Basically three dummy variable. 
imdb_data[,"duration_0_100"] <- as.numeric(imdb_data[,"duration"] <= 100)
imdb_data[,"duration_100_150"] <- as.numeric(imdb_data[,"duration"] > 100 & imdb_data[,"duration"] <= 150)
imdb_data[,"duration_150"] <- as.numeric(imdb_data[,"duration"] > 150)
imdb_data <- imdb_data[,colnames(imdb_data) %ni% c("duration")]

# Standarized movie_meter (May need to drop it) 
imdb_data[,"movie_meter_IMDBpro_norm"] <- scale(imdb_data[,"movie_meter_IMDBpro"], center = TRUE, scale = TRUE)
imdb_data <- imdb_data[,colnames(imdb_data) %ni% c("movie_meter_IMDBpro")]

library('fastDummies')
imdb_data <- dummy_cols(imdb_data, select_columns = 'years_released_decade')

data = imdb_data

# director, actor, cinematorgrapher average scores and film counts 
data <-data %>%
  group_by(director) %>%
  arrange(release_year) %>%
  mutate(director_film_count = row_number())%>%
  ungroup()

data$director_film_count <- data$director_film_count - 1


data <-data  %>%
  group_by(director) %>%
  arrange(release_year) %>%
  mutate(director_film_running_avaerge = lag(cumsum(imdb_score) / row_number(), default = 6))%>%
  ungroup()




data <-data %>%
  group_by(cinematographer) %>%
  arrange(release_year) %>%
  mutate(cinematographer_film_count = row_number())%>%
  ungroup()

data$cinematographer_film_count <- data$cinematographer_film_count - 1


data <-data  %>%
  group_by(cinematographer) %>%
  arrange(release_year) %>%
  mutate(cinematographer_film_running_avaerge = lag(cumsum(imdb_score) / row_number(), default = 6))%>%
  ungroup()


actor <- read.csv("C:/Users/xutian/Downloads/actor.csv")

actor <-actor %>%
  group_by(actor1) %>%
  arrange(release_year) %>%
  mutate(actor_film_count = row_number())%>%
  ungroup()


actor$actor_film_count <- actor$actor_film_count - 1


actor <-actor  %>%
  group_by(actor1) %>%
  arrange(release_year) %>%
  mutate(actor_film_running_avaerge = lag(cumsum(imdb_score) / row_number(), default = 6))%>%
  ungroup()

df2_selected <- actor %>%
  select(actor1,movie_id, actor_film_count)


data <- left_join(data, df2_selected, by = c("actor1" = "actor1","movie_id" = "movie_id"))

data$actor1_film_count<-data$actor_film_count

data <- subset(data, select = -actor_film_count)


data <- left_join(data, df2_selected, by = c("actor2" = "actor1","movie_id" = "movie_id"))

data$actor2_film_count<-data$actor_film_count

data <- subset(data, select = -actor_film_count)


data <- left_join(data, df2_selected, by = c("actor3" = "actor1","movie_id" = "movie_id"))

data$actor3_film_count<-data$actor_film_count

data <- subset(data, select = -actor_film_count)



df2_selected <- actor %>%
  select(actor1,movie_id, actor_film_running_avaerge)




data <- left_join(data, df2_selected, by = c("actor1" = "actor1","movie_id" = "movie_id"))

data$actor1_film_running_avaerge<-data$actor_film_running_avaerge

data <- subset(data, select = -actor_film_running_avaerge)


data <- left_join(data, df2_selected, by = c("actor2" = "actor1","movie_id" = "movie_id"))

data$actor2_film_running_avaerge<-data$actor_film_running_avaerge

data <- subset(data, select = -actor_film_running_avaerge)


data <- left_join(data, df2_selected, by = c("actor3" = "actor1","movie_id" = "movie_id"))

data$actor3_film_running_avaerge<-data$actor_film_running_avaerge

data <- subset(data, select = -actor_film_running_avaerge)

# new comedy feature, disney and universal. Remember to change pixar

data$comedy <- ifelse(grepl("comedy", data$genres, ignore.case = TRUE), 1, 0)


data$biography <- ifelse(grepl("biography", data$genres, ignore.case = TRUE), 1, 0)

data$documentary <- ifelse(grepl("documentary", data$genres, ignore.case = TRUE), 1, 0)

data$disney <- ifelse(grepl("disney", data$distributor, ignore.case = TRUE), 1, 0)

data$universal <- ifelse(grepl("universal", data$distributor, ignore.case = TRUE), 1, 0)

data$comedy_disnry<- data$comedy*data$disney

data$animation_universal<- data$animation*data$universal


# distributor
data <- data %>%
  group_by(distributor) %>%
  mutate(avg_distributor_imdb_score = mean(imdb_score)) %>%
  ungroup()

data <- data %>%
  group_by(distributor) %>%
  mutate(count_distributor_imdb_score = n()) %>%
  ungroup()


# maturity rating
data$maturity_rating[data$maturity_rating == "Approved"] <- "PG"

data$maturity_rating[data$maturity_rating == "PG-13"] <- "PG"

data$maturity_rating[data$maturity_rating == "Passed"] <- "PG"

data$maturity_rating[data$maturity_rating == "TV-14"] <- "PG"

data$maturity_rating[data$maturity_rating == "TV-G"] <- "G"

data$maturity_rating[data$maturity_rating == "GP"] <- "G"

data$maturity_rating[data$maturity_rating == "NC-17"] <- "R"
data$maturity_rating[data$maturity_rating == "X"] <- "R"
data$maturity_rating[data$maturity_rating == "M"] <- "R"

data <- dummy_cols(data, select_columns = 'maturity_rating')

write.csv(data, file = "documentary.csv")




data <- left_join(data, result, by = c("diretor" = "diretor"))

data$director_max<-result$max_value

data <- subset(data, select = -actor_film_running_avaerge)

result <- imdb_data %>%
  group_by(director) %>%
  summarize(max_value = max(imdb_score, na.rm = TRUE))


