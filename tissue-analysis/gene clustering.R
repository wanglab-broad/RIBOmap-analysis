setwd("/Users/zhouhaowen/Desktop/At_Broad/AD/RIBO_gene_cl")

library(readxl)
library(Seurat)
library(tidyverse)
library(magrittr)
library(ComplexHeatmap)
library(circlize)
library(viridis)
library(stringr)
require(pals)



data_ls <- list()
file_ls <- c("All_brain_region_3_filter_20221209.xlsx",
             "All_cell_type_3_filter_20221209.xlsx",
             "Astro_brain_region_3_filter_20221209(1).xlsx",
             "Astro_cell_type_3_filter_20221209.xlsx",
             "Oligo_brain_region_3_filter_20221209(1).xlsx",
             "Oligo_cell_type_6_14_15_20221217.xlsx",
             "OPC_brain_region_3_filter_20221209.xlsx",
             "Oligo_cell_type_rep2_filter.xlsx",
             "All_cell_type_rep2_filter.xlsx",
             "All_brain_region_rep2_filter.xlsx")


for(i in file_ls){
  data_df <- suppressWarnings(read_excel(i))
  #Check gene list
  if(sum(data_df$RIBOmap != data_df$STARmap) > 0){
    stop("Gene List Inconsist")
  }
  if(!is.null(data_df$STARmap)){
    tmp_df <- data.frame(gene = data_df$STARmap)
    message(i)
  }else if(!is.null(data_df$RIBOmap)){
    tmp_df <- data.frame(gene = data_df$RIBOmap)
  }else{#rep
    tmp_df <- data_df
    colnames(tmp_df)[1] <- "gene"
    tmp_title <- paste(i %>% str_split(pattern = "_") %>% unlist %>% .[1:4], collapse = "_")
    data_ls[[tmp_title]] <- tmp_df
    next
  }
  
  for(j in seq_len(which(colnames(data_df) == "STARmap")-1)){
    tmp <- str_split(colnames(data_df)[j],pattern = "\\.\\.\\.") %>% unlist() %>% .[1] %>% 
      str_replace(pattern = " ", replacement = "_")
    if(tmp != "RIBOmap" & tmp != ""){
      tmp_df[[paste0(tmp,".RIBO")]] <- data_df[,j] %>% unlist()
    }
  }
  for(j in seq(from = which(colnames(data_df) == "STARmap"), to = ncol(data_df) )){
    tmp <- str_split(colnames(data_df)[j],pattern = "\\.\\.\\.") %>% unlist() %>% .[1] %>% 
      str_replace(pattern = " ", replacement = "_")
    if(tmp != "STARmap" & tmp != ""){
      tmp_df[[paste0(tmp,".STAR")]] <- data_df[,j] %>% unlist()
    }
  }
  tmp_title <- paste(i %>% str_split(pattern = "_") %>% unlist %>% .[1:3], collapse = "_")
  tmp_df[["cor"]] <- apply(tmp_df, 1,
                           FUN = function(x){
                             tmp_vec <- colnames(tmp_df) %>% str_detect(pattern = "All_region|Mean|overall")
                             tmp_vec1 <- as.numeric(x[!tmp_vec & colnames(tmp_df) %>% str_detect(pattern = "RIBO")])
                             tmp_vec2 <- as.numeric(x[!tmp_vec & colnames(tmp_df) %>% str_detect(pattern = "STAR")])
                             cor(x = tmp_vec1, y = tmp_vec2)
                           })
  
  data_ls[[tmp_title]] <- tmp_df
}





#Seurat Clustering 
#Try Seurat Clustering



cl_title <- "All_brain_region"
cl_res <- 1.5

cl_title <- "All_cell_type"
cl_res <- 2.6

cl_title <- "Astro_brain_region"
cl_res <- 2.2

cl_title <- "Astro_cell_type"
cl_res <- 1

cl_title <- "Oligo_brain_region"
cl_res <- 2.2

cl_title <- "Oligo_cell_type"
cl_res <- 0.5

cl_title <- "OPC_brain_region"
cl_res <- 2

para_df <- data_ls[[cl_title]]

col_vec <- !(colnames(para_df) %in% c("gene","cor") | colnames(para_df) %>% str_detect(pattern = "All_region|Mean|overall|Mix|All_ctype"))
col_vec <- which(col_vec)
para_df[,col_vec[1:(length(col_vec)/2)]] <- as.matrix(para_df[,col_vec[1:(length(col_vec)/2)]] ) %>% t() %>% scale() %>% t()
para_df[,col_vec[(length(col_vec)/2+1):length(col_vec)]] <- as.matrix(para_df[,col_vec[(length(col_vec)/2+1):length(col_vec)]] ) %>% 
  t() %>% scale() %>% t()


tmp_obj <- CreateSeuratObject(counts = as.data.frame(para_df[,col_vec], row.names = as.character(para_df$gene)) %>% as.matrix() %>% t(),
                              project = paste0("Gene_clustering_",cl_title)) %>% ScaleData(do.scale = F, do.center = F)

#tmp_obj %<>% AddMetaData(metadata = data.frame(row.names = tmp_df$gene, cluster_old = tmp_df$cluster))

tmp_obj %<>% RunPCA(features = row.names(tmp_obj@assays[["RNA"]]), npcs = 10)

#ElbowPlot(tmp_obj)

tmp_obj %<>% FindNeighbors(dims = 1:10)
tmp_obj %<>% FindClusters(resolution = cl_res)# Brain 1.7 cell type 3

#tmp_obj %<>% RunUMAP(dims = 1:10)

#DimPlot(tmp_obj, reduction = "umap")
#DoHeatmap(tmp_obj, features = row.names(tmp_obj@assays$RNA)) + NoLegend()

para_df[["cluster_seurat"]] <- paste0("cl",tmp_obj@meta.data[["seurat_clusters"]])


tmp_hm <- Heatmap(matrix = para_df[,col_vec[1:(length(col_vec)/2)]], 
        cluster_rows  = F, show_row_names = F)

col_vec <- c(col_vec[1:(length(col_vec)/2)] %>% .[column_order(tmp_hm)],
             col_vec[(length(col_vec)/2+1):length(col_vec)] %>% .[column_order(tmp_hm)])
#Save All
brain_region_col_vec <- c(col_vec[1:(length(col_vec)/2)] %>% .[column_order(tmp_hm)],
                          col_vec[(length(col_vec)/2+1):length(col_vec)] %>% .[column_order(tmp_hm)])

col_vec <- brain_region_col_vec

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #all brain region
                                      levels = paste0("cl",c(8,2,12,7,5,4,6,11,9,1,10,3,0)))
col_vec <- c(4,2,9,6,3,5,7,8,13, 11, 18, 15, 12, 14, 16, 17)

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #all cell type
                                      levels = paste0("cl",c(6,11,9,14,10,5,7,12,8,13,0,1,3,2,4)))
col_vec <- c(7,8,9,10,6,12,11,4,3,2,5,20, 21, 22, 23, 19, 25, 24, 17, 16, 15, 18)

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Astro brain region
                                      levels = paste0("cl",c(6,7,3,10,9,1,8,0,5,2,4)))

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Astro brain region using All col_vec
                                      levels = paste0("cl",c(6,9,3,10,7,1,8,5,2,0,4)))

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Astro cell type
                                      levels = paste0("cl",c(4,0,3,1,6,2,5)))

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Oligo brain region
                                      levels = paste0("cl",c(5,3,2,8,6,7,0,10,4,9,1)))

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Oligo brain region using All col_vec
                                      levels = paste0("cl",c(5,3,0,8,7,6,10,4,2,1,9)))

para_df[["cluster_seurat"]] <- factor(para_df[["cluster_seurat"]], #Oligo cell type
                                      levels = paste0("cl",c(1,2,0)))
col_vec <- c(2,3,4,6,7,8)

para_df <- arrange(para_df,desc(cluster_seurat))

col_fun <- colorRamp2(quantile(unlist(para_df[,col_vec]),probs = c(0.1,0.5,0.9)), c("#0000FF", "#FFFFFF", "#FF0000"))

#Get column order
pdf(file = paste0("Dec29_",cl_title,"_hm_viridis.pdf"), width = 8, height = 10)
Heatmap(matrix = para_df[,col_vec], column_split = rep(c("RIBO","STAR"),each = length(col_vec)/2),
        cluster_columns = F,cluster_rows  = F, show_row_names = F, col = col_fun,
        column_order = colnames(para_df)[col_vec], row_split = as.factor(para_df$cluster_seurat), 
        border = TRUE,column_title = paste(cl_title),
        right_annotation = rowAnnotation(cor = para_df$cor,col = list(cor = colorRamp2(c(-0.5,-0.375,-0.25,-0.125,0,0.25,0.5,0.75,1), viridis(9)) ))) %>% print()
#tmp_hm %>% print()
dev.off()

#For All cell type/All brain region/Oligo cell type
#All cell type
tmp_df <- data_ls$All_cell_type_rep2
tmp_df[2:12] <- tmp_df[2:12] %>% t() %>% scale() %>% t()
colnames(tmp_df)[2:12] <- paste0(colnames(tmp_df)[2:12],".rep")
para_df2 <- left_join(para_df,tmp_df,by = "gene")
col_vec <- c(col_vec, 30:40 %>% .[column_order(tmp_hm)])

#All brain region
tmp_df <- data_ls$All_brain_region_rep2
tmp_df[2:9] <- tmp_df[2:9] %>% t() %>% scale() %>% t()
colnames(tmp_df)[2:9] <- paste0(colnames(tmp_df)[2:9],".rep")
para_df2 <- left_join(para_df,tmp_df,by = "gene")
col_vec <- c(col_vec, 22:29 %>% .[column_order(tmp_hm)])

#All cell type
tmp_df <- data_ls$Oligo_cell_type_rep2
tmp_df[2:4] <- tmp_df[2:4] %>% t() %>% scale() %>% t()
colnames(tmp_df)[2:4] <- paste0(colnames(tmp_df)[2:4],".rep")
para_df2 <- left_join(para_df,tmp_df,by = "gene")
col_vec <- c(col_vec, 12:14 %>% .[column_order(tmp_hm)])

pdf(file = paste0("Dec12_",cl_title,"_hm_add_rep.pdf"), width = 8, height = 10)
Heatmap(matrix = para_df2[,col_vec], column_split = rep(c("RIBO","STAR","REP"),each = length(col_vec)/3),
        cluster_columns = F,cluster_rows  = F, show_row_names = F, col = col_fun,
        column_order = colnames(para_df2)[col_vec], row_split = as.factor(para_df2$cluster_seurat), 
        border = TRUE,column_title = paste(cl_title),
        right_annotation = rowAnnotation(cor = para_df2$cor,col = list(cor = colorRamp2(seq(from = -1, to = 1, length.out = 7), viridis(7)) ))) %>% print()
dev.off()


write.csv(para_df,paste0("Dec19_gene_cl_top1k_",cl_title,".csv"))




