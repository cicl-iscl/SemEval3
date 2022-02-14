# Imports
library(caret)
library(readr)
library(xtable)

# Read input files (subtask 1)
En_Subtask1_labels <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-1/En-Subtask1-labels.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(
            Construction = col_factor
            (
                levels = c(
                    "andtoo",
                    "butnot",
                    "comparatives",
                    "drather",
                    "except",
                    "generally",
                    "particular",
                    "prefer",
                    "type",
                    "unlike"
                )
            ),
            Labels = col_factor(levels = c("0", "1"))
        ),
        trim_ws = TRUE
    )
En_Subtask1_labels$Language <- factor(c("en"))
Fr_Subtask1_labels <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-1/Fr-Subtask1-labels.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(
            Construction = col_factor
            (
                levels = c(
                    "andtoo",
                    "butnot",
                    "comparatives",
                    "drather",
                    "except",
                    "generally",
                    "particular",
                    "prefer",
                    "type",
                    "unlike"
                )
            ),
            Labels = col_factor(levels = c("0", "1"))
        ),
        trim_ws = TRUE
    )
Fr_Subtask1_labels$Language <- factor(c("fr"))
It_Subtask1_labels <-
    read_delim(
        "data/test/official_test_set_with_labels/subtask-1/It-Subtask1-labels.tsv",
        "\t",
        escape_double = FALSE,
        col_types = cols(
            Construction = col_factor
            (
                levels = c(
                    "andtoo",
                    "butnot",
                    "comparatives",
                    "drather",
                    "except",
                    "generally",
                    "particular",
                    "prefer",
                    "type",
                    "unlike"
                )
            ),
            Labels = col_factor(levels = c("0", "1"))
        ),
        trim_ws = TRUE
    )
It_Subtask1_labels$Language <- factor(c("it"))
answer_task1 <- read_delim(
    "answer_task1.tsv",
    "\t",
    escape_double = FALSE,
    col_types = cols(Labels = col_factor(levels = c("0", "1"))),
    trim_ws = TRUE
)

# Merge dataframes
subtask1 <- merge(
    rbind(En_Subtask1_labels, Fr_Subtask1_labels,
          It_Subtask1_labels),
    answer_task1,
    by = "ID",
    suffixes = c("_gold", "_pred")
)

# Compute metrics
for (lang in levels(subtask1$Language)) {
    confusion_matrix <-
        confusionMatrix(
            subtask1[subtask1$Language == lang, ]$Labels_pred,
            subtask1[subtask1$Language == lang, ]$Labels_gold,
            mode = "everything",
            positive = "1"
        )
    
    
    metrics1 <-
        data.frame(total = c(
            confusion_matrix[["byClass"]][["Precision"]],
            confusion_matrix[["byClass"]][["Recall"]],
            confusion_matrix[["byClass"]][["F1"]]
        ))
    
    for (construction in levels(subtask1$Construction)) {
        confusion_matrix <-
            confusionMatrix(
                subtask1[(subtask1$Construction == construction) &
                             (subtask1$Language == lang), ]$Labels_pred,
                subtask1[(subtask1$Construction == construction) &
                             (subtask1$Language == lang), ]$Labels_gold,
                mode = "everything",
                positive = "1"
            )
        
        metrics1[[construction]] <-
            c(confusion_matrix[["byClass"]][["Precision"]],
              confusion_matrix[["byClass"]][["Recall"]],
              confusion_matrix[["byClass"]][["F1"]])
    }
    
    row.names(metrics1) <- c("Precision", "Recall", "F1")
    
    # Save to file as tex
    print(
        xtable(
            metrics1,
            type = "latex",
            caption = sprintf("Metrics for %s.", lang),
            align = rep("c", 12)
        ),
        file = sprintf("subtask1_score_table_%s.tex", lang)
    )
}
