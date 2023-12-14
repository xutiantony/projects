imdb = read.csv("C:/Users/xutian/Desktop/imdb_training_edited.csv")
attach(imdb)
View(imdb)

test = read.csv("C:/Users/xutian/Desktop/testimdb.csv")

test$disney_animation<-test$disney*test$animation
test$academy_months_drama<-test$academy_months*test$drama

imdb$disney_animation<-imdb$disney*imdb$animation
imdb$academy_months_drama<-imdb$academy_months*imdb$drama

Q1 <- quantile(imdb$nb_news_articles_log, 0.25) 
Q3 <- quantile(imdb$nb_news_articles_log, 0.75) 
# Calculating the IQR 
IQR_value <- IQR(imdb$nb_news_articles_log) 
# Identifying the rows where the value column is an outlier 
outlier_rows <- which(imdb$nb_news_articles_log < (Q1 - 1.5*IQR_value) | imdb$nb_news_articles_log > (Q3 + 1.5*IQR_value)) 
# Printing the rows with outliers 
print(outlier_rows)




Q1 <- quantile(imdb$movie_meter_IMDBpro_norm, 0.25) 
Q3 <- quantile(imdb$movie_meter_IMDBpro_norm, 0.75) 
# Calculating the IQR 
IQR_value <- IQR(imdb$movie_meter_IMDBpro_norm) 
# Identifying the rows where the value column is an outlier 
outlier_rows2 <- which(imdb$movie_meter_IMDBpro_norm < (Q1 - 1.5*IQR_value) | imdb$movie_meter_IMDBpro_norm > (Q3 + 1.5*IQR_value)) 
# Printing the rows with outliers 
print(outlier_rows2)

new_imdb=imdb[-c(817,722,597,545,1112),]
new_imdb = new_imdb[-outlier_rows2,]
new_imdb = new_imdb[-outlier_rows,]

new_imdb$actor1_std<-scale(new_imdb$actor1_star_meter)
new_imdb$actor2_std<-scale(new_imdb$actor2_star_meter)
new_imdb$actor3_std<-scale(new_imdb$actor3_star_meter)


fit = glm(imdb_score~
            nb_news_articles_log+
            #actor1_std+
            #actor2_std+
            #actor3_std+
            movie_meter_IMDBpro_norm+
            action+
            
            thriller+
            musical+
            
            horror+
            drama+
            #war+
            animation+
            crime+
            comedy+
            animation_universal+
            avg_distributor_imdb_score+
            country_dummy_USA+
            duration_0_100+
            duration_100_150+
            disney_animation+
            academy_months+
            # academy_months_drama+
            poly(director_film_running_avaerge, 2)+
            cinematographer_film_running_avaerge+
            actor1_film_running_avaerge+
            actor2_film_running_avaerge+
            actor3_film_running_avaerge+
            maturity_rating_G+
            maturity_rating_PG+
            director_max+
            biography+
            documentary, data=new_imdb)

library(boot)
mse=cv.glm(new_imdb, fit, K=10)$delta[1]
# Use the parameter "K= " to do K-fold
# For K, use the capital letter!!!
mse
summary(fit)

predict(fit,test)

outlierTest(fit)



