library(pheatmap)

cor_mean <- as.matrix(read.csv("D:/Zhaoxi/mouse_vision/code/WF/cor_mean.csv", row.names = 1))
pheatmap(cor_mean, scale = "none", clustering_method = "complete", display_numbers = TRUE, number_format = "%.3f",
         main = "Correlation Coefficient of Visual Areas", fontsize = 15, 
         filename = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\patches_correlation_heatmap.png")


patch_movie_mean <- as.matrix(read.csv("D:/Zhaoxi/mouse_vision/data/patch_movie_20rep-mean.csv", row.names = 1))
pheatmap(patch_movie_mean, scale = "row", clustering_method = "complete", display_numbers = F, 
         main = "mean Δf/f", fontsize = 10, cellwidth = 20, cellheight = 20, angle_col = 45, cluster_rows = FALSE, 
         cluster_columns = FALSE, 
         filename = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\patch_movie_crow-scaled.png")


patch_movie_all_mean_norm_df <- as.matrix(read.csv("D:\\Zhaoxi\\mouse_vision\\data/patch_movie_all_mean_norm_df.csv", row.names = 1))
result <- pheatmap(patch_movie_all_mean_norm_df, scale = "none", clustering_method = "complete", display_numbers = F, 
         main = "mean Δf/f scaled", fontsize = 10, cellwidth = 20, cellheight = 20, angle_col = 45, 
         filename = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\patch_movie_all_mean_norm_df.png")
clustered_df <- patch_movie_all_mean_norm_df[result$tree_row[["order"]], result$tree_col[["order"]]]
write.csv(clustered_df, file = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\clustered_df.csv")

result$tree_col[["order"]]patch_movie_all_mean_norm1_df <- as.matrix(read.csv("D:\\Zhaoxi\\mouse_vision\\data/patch_movie_all_mean_norm1_df.csv", row.names = 1))
pheatmap(patch_movie_all_mean_norm1_df, scale = "none", clustering_method = "complete", display_numbers = F, 
         fontsize = 10, cellwidth = 20, cellheight = 20, angle_col = 45, cluster_rows = FALSE, 
         filename = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\patch_movie_all_mean_norm1_df.png")

all_mean_v1 <- as.matrix(read.csv("D:\\Zhaoxi\\mouse_vision\\data/all_mean_v1.csv", row.names = 1))
pheatmap(all_mean_v1, scale = "none", clustering_method = "complete", display_numbers = F, 
         main = "mean deltaf/f", fontsize = 15, cellwidth = 20, cellheight = 20, angle_col = 45, 
         filename = "D:\\Zhaoxi\\mouse_vision\\code\\WF\\all_mean_v1_df.png")
