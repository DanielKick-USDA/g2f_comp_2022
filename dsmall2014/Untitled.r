library(BGLR)
# Settings ---------------------------------------------------------------------
# EDITME #######################################################################
model_name = 'As'
save_suffix = 'small'
CV = 2014

# for(replicate in 1:3){
  # iterations based on those in 
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9434216/
  BGLR_nIter = 10000  
  BGLR_burnIn=  5000 
  BGLR_RegType = "RKHS"
  
  test_downsample = TRUE 
  
  Y_Train <- readRDS(paste0('../data/Processed/y_matrix_', 
                            as.character(CV), "_", save_suffix, '.rds'))
  GA <- readRDS(paste0(
      "../data/Processed/GA_matrix_", as.character(CV), "_", save_suffix, '.rds'))
  # GD <- readRDS(paste0(
  #     "../data/Processed/GD_matrix_", as.character(CV), "_", save_suffix, '.rds'))
  
  BGLR_Model_List <- list(
        G=list(K   = GA,         model = BGLR_RegType)
  )
  
  ################################################################################
  
  
  # Quick Confirmations ----------------------------------------------------------
  print("Current pwd is:")
  print(getwd())
  # Matrix Setup -----------------------------------------------------------------
  
  if(test_downsample){
    subset_idx <- sample(1:length(Y_Train))[1:100]#1000]
    Y_Train <- Y_Train[subset_idx]
    GA <- GA[subset_idx, subset_idx]    
  }
  

  # Fit the model! ---------------------------------------------------------------
  gc()
  model_start <- Sys.time()
  fm <- BGLR(
    y = Y_Train,
    ETA = BGLR_Model_List,
    nIter = BGLR_nIter,
    burnIn = BGLR_burnIn
  )
  model_end <- Sys.time()
  gc()
  
  model_time <- model_end - model_start
  # write out log file with times
  fm_run_time <- as.data.frame(
    list(model_start, model_end, model_time),
    col.names = c("Start", "End", "Duration")
  )

#   GA <- readRDS(paste0(
#       "../data/Processed/GA_matrix_", as.character(CV), "_", save_suffix, '.rds'))
#   GD <- readRDS(paste0(
#       "../data/Processed/GD_matrix_", as.character(CV), "_", save_suffix, '.rds'))

# dim(dd)






#   write.csv(fm_run_time, paste0("../data/Shared_Model_Output/", model_name, as.character(replicate), "BlupRunTime.csv"))
  
  # Save Model & Predictions  ----------------------------------------------------
  # plot(Y, fm$yHat)
  # save out model
  # saveRDS(fm, file = "../fm", model_name, ".rds")
  
  # save out predictions.
  Y_ObsPr = as.data.frame(
    list(Y_Train, fm$yHat), 
    col.names = c("YTrain", "YHat"))
  
  write.csv(Y_ObsPr, paste0("../data/Shared_Model_Output/", model_name, as.character(CV), as.character(replicate), "BlupYHats.csv"))
  
  # rm and gc
  rm(list = c('fm'))
  gc()
# }
