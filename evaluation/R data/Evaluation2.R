# Read input files (subtask 2)
En_Subtask2_scores <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-2/En-Subtask2-scores.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(Construction = col_factor
                         (
                             levels = c(
                                 'andtoo',
                                 'butnot',
                                 'comparatives',
                                 'ingeneral',
                                 'particular',
                                 'type',
                                 'unlike'
                             )
                         )),
        trim_ws = TRUE
    )
Fr_Subtask2_scores <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-2/Fr-Subtask2-scores.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(Construction = col_factor
                         (
                             levels = c(
                                 'andtoo',
                                 'butnot',
                                 'comparatives',
                                 'ingeneral',
                                 'particular',
                                 'type',
                                 'unlike'
                             )
                         )),
        trim_ws = TRUE
    )
It_Subtask2_scores <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-2/It-Subtask2-scores.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(Construction = col_factor
                         (
                             levels = c(
                                 'andtoo',
                                 'butnot',
                                 'comparatives',
                                 'ingeneral',
                                 'particular',
                                 'type',
                                 'unlike'
                             )
                         )),
        trim_ws = TRUE
    )


# Add language column
En_Subtask2_scores$Language <- factor(c("en"))
Fr_Subtask2_scores$Language <- factor(c("fr"))
It_Subtask2_scores$Language <- factor(c("it"))

answer_task2 <- read_delim("median_task2.tsv",
                           "\t",
                           escape_double = FALSE,
                           trim_ws = TRUE)
names(answer_task2)[names(answer_task2) == 'Score'] <- 'Scores'

# Merge dataframes
subtask2 <- merge(
    rbind(En_Subtask2_scores, Fr_Subtask2_scores, It_Subtask2_scores),
    answer_task2,
    by = "ID",
    suffixes = c("_gold", "_pred")
)

# Compute metrics
for (lang in levels(subtask2$Language)) {
    mse <- mean((subtask2[subtask2$Language == lang,]$Scores_gold -
                     subtask2[subtask2$Language == lang,]$Scores_pred) ^ 2)
    rmse <- sqrt(mse)
    rho <-
        cor(subtask2[subtask2$Language == lang,]$Scores_gold,
            subtask2[subtask2$Language == lang,]$Scores_pred,
            method = "spearman")
    metrics2 <- data.frame(total = c(mse, rmse, rho))
    print(sprintf("%s: %f", lang, rho))
    for (construction in levels(subtask2$Construction)) {
        mse <-
            mean((subtask2[subtask2$Language == lang &
                               subtask2$Construction == construction,]$Scores_gold -
                      subtask2[subtask2$Language == lang &
                                   subtask2$Construction == construction,]$Scores_pred) ^
                     2)
        rmse <- sqrt(mse)
        rho <-
            cor(subtask2[subtask2$Language == lang &
                             subtask2$Construction == construction,]$Scores_gold,
                subtask2[subtask2$Language == lang &
                             subtask2$Construction == construction,]$Scores_pred,
                method = "spearman")
        metrics2[[construction]] <- c(mse, rmse, rho)
    }
    row.names(metrics2) <- c("MSE", "RMSE", "rho")
    
    # Save to file as tex
    print(
        xtable(
            metrics2,
            type = "latex",
            caption = sprintf("Metrics for %s.", lang),
            align = rep("c", 9)
        ),
        file = sprintf("subtask2_score_table_%s.tex", lang)
    )
}
