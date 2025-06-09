install.packages("xtable")
install.packages("vcd")
library(jsonlite)
library(xtable)
library(vcd)
library(effsize)
setwd("data/")
df <- stream_in(file("sample_clf.json"))
df_train <- fromJSON(file("golden-standard-train.json"))
df_test <- fromJSON(file("golden-standard-test.json"))
df$delta_num <- as.numeric(df$delta)
# tables train
table(df_train$suspense)
table(df_train$curiosity)
table(df_train$surprise)
mean(df_train$suspense)
mean(df_train$curiosity)
mean(df_train$surprise)
# tables test
table(df_test$suspense)
table(df_test$curiosity)
table(df_test$surprise)
mean(df_test$suspense)
mean(df_test$curiosity)
mean(df_test$surprise)
combined <- rbind(df_train, df_test)
table(combined$suspense)
table(combined$curiosity)
table(combined$surprise)

# tables sample
table(df$predicted_suspense)
table(df$predicted_curiosity)
table(df$predicted_surprise)
table(df$predicted_story)


xtable(table(df$predicted_story, df$delta))
wilcox.test(df$predicted_suspense ~ df$delta)
wilcox.test(df$predicted_curiosity ~ df$delta)
wilcox.test(df$predicted_surprise ~ df$delta)

# Effect size
cliff.delta(df$predicted_suspense[df$delta == "TRUE"],
            df$predicted_suspense[df$delta == "FALSE"])
cliff.delta(df$predicted_curiosity[df$delta == "TRUE"],
            df$predicted_curiosity[df$delta == "FALSE"])
cliff.delta(df$predicted_surprise[df$delta == "TRUE"],
            df$predicted_surprise[df$delta == "FALSE"])

story_delta <- table(df$predicted_story, df$delta)
chisq.test(story_delta)
assocstats(story_delta)
# N of Comments
mean_nonzero <- df[df$num_comments != 0,]$num_comments
mn <- mean(mean_nonzero)
mn
df$comment_group <- ifelse(
  df$num_comments == 0, "zero",
  ifelse(df$num_comments > mn, "high", "low")
)
df_filt <- df[df$comment_group != "zero",]

wilcox.test(predicted_suspense ~ comment_group, data=df_filt)
wilcox.test(predicted_curiosity ~ comment_group, data=df_filt)
wilcox.test(predicted_surprise ~ comment_group, data=df_filt)
# Effect size
cliff.delta(df$predicted_suspense[df$comment_group == "high"],
            df$predicted_suspense[df$comment_group == "low"])
cliff.delta(df$predicted_curiosity[df$comment_group == "high"],
            df$predicted_curiosity[df$comment_group == "low"])
cliff.delta(df$predicted_surprise[df$comment_group == "high"],
            df$predicted_surprise[df$comment_group == "low"])

xtable(table(df$delta, df$comment_group))

delta_comment <- table(df_filt$delta, df_filt$comment_group)
story_comment <- table(df_filt$predicted_story, df_filt$comment_group)
chisq.test(story_comment)
assocstats(story_comment)
chisq.test(delta_comment)
assocstats(delta_comment)

cor.test(df_filt$predicted_suspense, df_filt$num_comments, method="spearman")
cor.test(df_filt$predicted_curiosity, df_filt$num_comments, method="spearman")
cor.test(df_filt$predicted_surprise, df_filt$num_comments, method="spearman")
cor.test(df$story_num, df$num_comments, method = "pearson")


cor.test(df$predicted_suspense, df$delta_num, method = "spearman")
cor.test(df$predicted_curiosity, df$delta_num, method = "spearman")
cor.test(df$predicted_surprise, df$delta_num, method = "spearman")
cor.test(df$story_num, df$delta_num, method = "pearson")
# correlations interactions
df$interaction_csr <- df$predicted_curiosity * df$predicted_surprise
df$interaction_ss <- df$predicted_suspense * df$predicted_surprise
df$interaction_csn <- df$predicted_curiosity * df$predicted_suspense

cor.test(df$interaction_csr, df$delta_num, method = "pearson")
cor.test(df$interaction_ss, df$delta_num, method = "pearson")
cor.test(df$interaction_csn, df$delta_num, method = "pearson")

