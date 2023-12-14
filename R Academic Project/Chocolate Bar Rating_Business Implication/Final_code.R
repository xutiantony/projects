`%ni%` <- Negate(`%in%`)
library(dplyr)
library(ggplot2)
library(cowplot)
library(stargazer)
# Load data
df <- read.csv("C:/Users/xutian/Desktop/Rfinal project/Chocolate bar ratings.csv")

############################################EDA###################################
#clean the percentage
df$Percent <- as.numeric(sub("%", "", df$Percent))

hist(df$Percent, main="Cocoa Percentage Distribution", xlab="Cocoa Percentage", col="blue", border="black")

fit <- lm(Rating ~ Percent, data = df)
stargazer(fit, type = "text")

ggplot(df, aes(x = Percent, y = Rating)) + geom_point()

ggplot(df, aes(x = Date, y = Rating)) + geom_point()
#drop ref column
df <- df[,colnames(df) %ni% "REF"]

df$lengths <- nchar(df$`Specific_Name`)

df <- df[,colnames(df) %ni% "Specific_Name"]

df %>% 
  group_by(Company) %>% 
  summarise(Count = n(),
            Percentage = n() / nrow(df) * 100)

df %>% 
  group_by(Location) %>% 
  summarise(Count = n(),
            Percentage = n() / nrow(df) * 100)
########################################Classification#####################
library(rpart)
library(rpart.plot)
df <- df %>% group_by(Company) %>% mutate(company_count = n())
df <- df %>% group_by(Broad_origin) %>% mutate(origin_count = n())
df <- df %>% group_by(Company) %>% mutate(company_mean = sum(Rating)/n())
df <- df %>% group_by(Broad_origin) %>% mutate(origin_mean = sum(Rating)/n())


df$low <- ifelse(df$Rating < 2.5, 1, 0)

dfa<-df

dfa <- dfa[,colnames(dfa) %ni% "Company"]

dfa <- dfa[,colnames(dfa) %ni% "Rating"]

attach(dfa)
myoverfittedtree=rpart(low~Date+Percent+Location+Type+origin_count+origin_mean
                       +company_count,control=rpart.control(cp=0.00000001)) 

opt_cp=myoverfittedtree$cptable[which.min(myoverfittedtree$cptable[,"xerror"]),"CP"]

mytree=rpart(low~Date+Percent+Location+Type++origin_count+origin_mean
                       +company_count,control=rpart.control(cp=0.01)) 

rpart.plot(mytree)
summary(mytree)


df$high <- ifelse(df$Rating > 3.5, 1, 0)

dfb<-df

dfb <- dfb[,colnames(dfb) %ni% "Company"]

dfb <- dfb[,colnames(dfb) %ni% "Rating"]

attach(dfb)
myoverfittedtree=rpart(high~Date+Percent+Location+Type++origin_count+origin_mean
                       +company_count,control=rpart.control(cp=0.00001)) 

opt_cp=myoverfittedtree$cptable[which.min(myoverfittedtree$cptable[,"xerror"]),"CP"]

mytree=rpart(high~Date+Percent+Location+Type+origin_count+origin_mean
             +company_count,control=rpart.control(cp=0.02)) 

rpart.plot(mytree)

summary(mytree)

########################################PCA
install.packages("ggfortify")
library(ggfortify)

dfc<- df[,colnames(df) %ni% "Company"]
dfc<- dfc[,colnames(dfc) %ni% "Broad_origin"]

dfc<- dfc[,colnames(dfc) %ni% "low"]
dfc<- dfc[,colnames(dfc) %ni% "high"]

dfc<- dfc[,colnames(dfc) %ni% "Type"]
dfc<- dfc[,colnames(dfc) %ni% "lengths"]


dfc <- dfc %>% group_by(Location) %>% mutate(location_count = n())
dfc <- dfc %>% group_by(Location) %>% mutate(location_mean = sum(Rating)/n())

dfc<- dfc[,colnames(dfc) %ni% "Location"]

pca=prcomp(dfc, scale=TRUE)
autoplot(pca, data = dfc, loadings = TRUE, loadings.label = TRUE )

pca













