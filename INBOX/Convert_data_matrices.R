library(tidyverse)
library(arrow)

entries <- list.files("./data_matrices/")
entries <- entries[stringr::str_detect(entries, ".rds$")]

for(entry in entries){
  
  temp = readRDS(paste0("./data_matrices/", entry))
  # uncompressed G is 15.7 gb
  # write.csv(temp, paste0("./data_matrices/", stringr::str_remove(entry, ".rds"), ".csv"))
  
  temp = as.data.frame(temp)
  # arrow::write_feather(temp, paste0("./data_matrices/", stringr::str_remove(entry, ".rds"), ".feather"))
  # for G
  # Error: cannot allocate vector of size 324 Kb
  
  write.csv(diag(temp), paste0("./data_matrices/", stringr::str_remove(entry, "_diag.rds"), ".csv"))
  
}