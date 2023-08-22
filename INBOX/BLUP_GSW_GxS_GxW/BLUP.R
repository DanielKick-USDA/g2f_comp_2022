library(foreach)
library(magrittr)
library(ggplot2)
library(BGLR)
# Settings ---------------------------------------------------------------------
clean_dir = FALSE
clean_dir_exempt_files = c("BLUP.R")
# EDITME #######################################################################
needed_matrices = c("G", 
                    "K.Soil", 
                    "K.Weather", 
                    "GKs",
                    "GKw"#,
                    # "KsKw"
)
################################################################################
# iterations based on those in 
# Pérez-Rodríguez and de los Campos 2022
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9434216/
BGLR_nIter = 10000  
BGLR_burnIn=  5000 
BGLR_RegType = "RKHS"

scale_ECs <- TRUE

# WARNING:
# Testing can only be done prior to running _any_ full model because the 
# matrices are shared between all models run within 3_finalize_model_BGLR_BLUPS
# test_on_subset = TRUE  
test_on_subset = FALSE 
test_on_subset_nTrain = 900
test_on_subset_nTest = 100

# Quick Confirmations ----------------------------------------------------------
print("Current pwd is:")
print(getwd())
# Clean Directory --------------------------------------------------------------
# This should only be done after confirming that a downsampled model works
if(clean_dir){
  for(entry in list.files()){
    if(entry %in% clean_dir_exempt_files){
      # DO NOTHING
    } else {
      unlink(paste0("./", entry))    
    }
  }  
}


# Matrix Setup -----------------------------------------------------------------
data_loc <- '../../../data/processed/'

## Response Variable and Train/Test Info ---------------------------------------
Y    <- read.csv(paste0(data_loc, "ForRY.csv"))    #  0.052 sec
Y    <- as.matrix(Y)[, 'Y']

Idx  <- read.csv(paste0(data_loc, "ForRIdx.csv"))  #  0.059 sec
Idx$X <- Idx$X+1 # account for Python/R's array indexing 


# set up downsampling indices if testing code
if( test_on_subset ){
  # downsize data for testing
  trainIdBig <- sample(Idx[Idx$Set == "Train", "X"], test_on_subset_nTrain)
  testIdBig <- sample(Idx[Idx$Set == "Test", "X"], test_on_subset_nTest)
}

# Mask test data from BGLR
if( test_on_subset ){ # downsample if testing code
  Y <- Y[c(trainIdBig, testIdBig)] 
  test_idx_to_rm = (test_on_subset_nTrain+1):length(Y)
}else{
  # Y_Train <- Y
  # Y_Train[ Idx[Idx$Set == "Test", "X"] ] <- NA
  test_idx_to_rm = Idx[Idx$Set == "Test", "X"]
}

Y_Train <- Y
Y_Train[test_idx_to_rm] <- NA

## Predictor Variables ---------------------------------------------------------
# these are massive csvs, esp. weather, but doing it this way avoids feather as 
# a dependency for ATLAS to speed up running models, create the needed matrices 
# only if they don't already exist and only create the ones we need.


if("G" %in% needed_matrices){
  ## G matrix ------------------------------------------------------------------
  matrix_file <- "G_matrix.rds"
  matrix_file_path = paste0("../../../data/processed/BLUP_matrices/", matrix_file)
  if (matrix_file %in% list.files("../../../data/processed/BLUP_matrices/")){
    if(!("G" %in% ls())){
      G <- readRDS(matrix_file_path)      
    }
  } else {
    ### Load -------------------------------------------------------------------
    gc()
    Gdat <- read.csv(paste0(data_loc, 'ForRG.csv'))    # 49.124 sec
    Gdat <- as.matrix(Gdat)
    if( test_on_subset ){ # downsample if testing code
      Gdat <- Gdat[c(trainIdBig, testIdBig), ]
    }
    
    ### Calc ------------------------------------------------------------------- 
    geno <- as.matrix(Gdat[, grep("^PC", colnames(Gdat))])
    rownames(geno) <- 1:(dim(Gdat)[1]) 
    G <- tcrossprod(geno)
    G <- G / mean(diag(G))
    
    rm(list = c("Gdat", "geno"))
    saveRDS(G, file = matrix_file_path)
    gc()
  }
}



if("K.Soil" %in% needed_matrices){
  ## S matrix ------------------------------------------------------------------
  matrix_file <- "Ksoil_matrix.rds"
  matrix_file_path = paste0("../../../data/processed/BLUP_matrices/", matrix_file)
  if (matrix_file %in% list.files("../../../data/processed/BLUP_matrices/")){
    if(!("K.Soil" %in% ls())){
      K.soil <- readRDS(matrix_file_path)
    }
  } else {
    ### Load -------------------------------------------------------------------
    gc()
    Sdat <- read.csv(paste0(data_loc, "ForRS.csv"))    #  0.233 sec
    Sdat <- as.matrix(Sdat)
    if( test_on_subset ){ # downsample if testing code
      Sdat <- Sdat[c(trainIdBig, testIdBig), ]
    }
    
    
    ### Calc ------------------------------------------------------------------- 
    # Environmental covariates
    env.soil <- scale(as.matrix(Sdat[, colnames(Sdat)[colnames(Sdat) != "X"]], 
                                center=TRUE, 
                                scale=scale_ECs))
    rownames(env.soil) <- 1:(dim(Sdat)[1]) 
    
    K.soil <- tcrossprod(env.soil)
    K.soil <- K.soil/mean(diag(K.soil))
    
    rm(list = c("Sdat", "env.soil"))
    saveRDS(K.soil , file = matrix_file_path)
    gc()
  }  
}



if("K.Weather" %in% needed_matrices){
  ## W (ERM) matrix ------------------------------------------------------------
  matrix_file <- "Kweather_matrix.rds"
  matrix_file_path = paste0("../../../data/processed/BLUP_matrices/", matrix_file)
  if (matrix_file %in% list.files("../../../data/processed/BLUP_matrices/")){
    if(!("K.weather" %in% ls())){
      K.weather <- readRDS(matrix_file_path)
    }
  } else {
    ### Load -------------------------------------------------------------------
    gc()
    Wdat <- read.csv(paste0(data_loc, "ForRW.csv"))    # 97.89 sec
    Wdat <- as.matrix(Wdat)
    if( test_on_subset ){ # downsample if testing code
      Wdat <- Wdat[c(trainIdBig, testIdBig), ]
    }
    
    
    ### Calc -------------------------------------------------------------------    
    weather_ECs <- c("N", "P", "K", "TempMin", "TempMean", "TempMax", "DewPointMean", "RelativeHumidityMean", "SolarRadiationMean", "WindSpeedMax", "WindDirectionMean", "WindGustMax", "SoilTempMean", "SoilMoistureMean", "UVLMean", "PARMean", "PhotoperiodMean", "VaporPresEst", "WaterTotalInmm")
    
    # env.weather is a list of 8 objects
    # each object contains a matrix of 145 measurements x nObs
    # these values are drawn from the input df
    env.weather <- setNames(
      foreach(weather_EC=weather_ECs) %do% {
        
        # get the cols for each day's reading in form 
        # "max_temp1", "max_temp2" ... max_temp145"
        indx_of_weather_daily_vals = grep(paste0("^", weather_EC, "_"), colnames(Wdat)) # <- Note: this underscore is critical to prevent "P" from not matching with photoperiod and PAR too. 
        
        # this is in nObs x Daily value (145)
        as.matrix(
          Wdat[, indx_of_weather_daily_vals]) %>% 
          t # now it's Daily value x nObs
      }, 
      weather_ECs) # setNames() makes the matrices in this list accessible by name
    
    make_ERM <- function(E_list, w, summary_functions = NULL) {
      # create a list of nObsxnObs matrices and `.combine` them by adding the matrices together
      # this results in a single nObsxnObs matrix aggregating all the enviromental covariates.
      
      foreach(E = E_list, .combine = "+") %do% {
        # Time bins
        windows <- cut_interval(1:nrow(E), length = w) # this function is from ggplot2
        # each 3 day period gets a new group:
        # [1] [0,3] [0,3] [0,3]     (3,6] (3,6] (3,6]     (6,9] ...  
        # [145] (144,147]
        # Why this isn't done with seq is beyond me. Possibly because it returns cuts?
        
        if (length(unique(windows)) == 1) {
          Z <- matrix(1, nrow = nrow(E), ncol = 1)
          # if the window length is one return a single column of ones
          
        } else {
          Z <- model.matrix( ~ windows)
          # for window size three this becomes
          
          #   intercept win2 win3 ...
          # 1         1    0    0      # The first group is the intercept
          # 2         1    0    0
          # 3         1    0    0
          # 4         1    1    0      # One hot encoded groups
          # 5         1    1    0
          # 6         1    1    0
          # 7         1    0    1
          # 8         1    0    1
          # 9         1    0    1
          # ...
        }
        
        # # Average by time bin
        # EC <- crossprod(Z,    # 145x49   window design matrix
        #                 E     # 145xnObs data matrix
        # ) %>% # 49xnObs 
        #   t %>%          # flip nObsx49
        #   scale %>%      # center and scale each column
        #   t %>%          # flip 49xnObs
        #   na.omit
        
        # Updated processing: The adapted code did not account for using 
        # management data where there might be no variability across groups (all
        # groups recieve no fertilizer on a given day). This adapted version 
        # selectively scales entries to avoid these days becoming NA and being 
        # removed.
        
        # Average by time bin
        EC <- crossprod(Z,    # 145x49   window design matrix
                        E     # 145xnObs data matrix
        ) %>% # 49xnObs
          t #%>%          # flip nObsx49
        # scale %>%      # center and scale each column
        # replaced because this is introducing nas into 0s for `N`, `P` where columns are all 0 (i.e. when no fertilizer was applied)
        # scale each column one at a time if and only if the sd != 0
        # This prevents values from becoming NA and thus keeps the dimensions to the expected size
        for(i in seq(1, ncol(EC))){
          if(sd(EC[,i]) != 0){
            EC[,i] = scale(EC[,i])
          } else {
            EC[,i] = scale(EC[,i], scale = FALSE)
          }
        }
        # and return to processing as normal
        EC <- EC %>%
          t %>%          # flip 49xnObs
          na.omit
        
        
        # Summary by time bin
        # This functionality is not used
        if (!is.null(summary_functions)) {
          EC_summary <- foreach(summary_function = summary_functions) %do% {
            summary_by_window <-
              by(E, windows, function(x)
                apply(x, 2, summary_function))
            
            do.call(rbind, summary_by_window) %>% t %>% scale %>% t %>% na.omit
          } %>% as.list
          
          EC <- do.call(rbind, append(list(EC), EC_summary))
        }
        # Environmental relationship matrix
        return(crossprod(EC)) # Return a nObsxnObs matrix
      }
    }
    
    K.weather <- make_ERM(env.weather, w = 3) # Make an environmental relationship matrix of nObs x nObs
    K.weather <- K.weather/mean(diag(K.weather))
    gc()
    
    # Don't remove objects here because this is within foreach
    saveRDS(K.weather , file = matrix_file_path)
    gc()
  }  
  # remove them here instead
  rm(list = c("Wdat", "env.weather"))
  gc()
  
}



# Interaction Matrices ---------------------------------------------------------
# These are likely faster to compute again than to load off the disk.
# because we're using BGLR, we need not pull out the training (calibration, CS 
# in the example code). Instead we can mask the values to be predicted and as a 
# result we'll use the full K matrices when defining interactions
if("GKs" %in% needed_matrices){
  ## GxS matrix ------------------------------------------------------------------
  GKs  <- G * K.soil
}

if("GKw" %in% needed_matrices){
  ## GxW matrix ------------------------------------------------------------------
  GKw  <- G * K.weather
}

if("KsKw" %in% needed_matrices){
  ## SxW matrix ------------------------------------------------------------------
  KsKw <- K.soil * K.weather   
}

# Fit the model! ---------------------------------------------------------------
gc()
model_start <- Sys.time()
fm <- BGLR(
  y = Y_Train,
  ETA = list(
# EDITME #######################################################################    
    G=list(K   = G,         model = BGLR_RegType),
    S=list(K   = K.soil,    model = BGLR_RegType),
    W=list(K   = K.weather, model = BGLR_RegType),
    GxS=list(K = GKs,       model = BGLR_RegType),
    GxW=list(K = GKw,       model = BGLR_RegType)#,
    # SxW=list(K = KsKw,      model = BGLR_RegType)
################################################################################
  ),
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
write.csv(fm_run_time, "./BlupRunTime.csv")


# Save Model & Predictions  ----------------------------------------------------
# plot(Y, fm$yHat)
# save out model
saveRDS(fm, file = "./fm.rds")

# save out predictions.
Y_ObsPr = as.data.frame(
  list(Y, Y_Train, fm$yHat), 
  col.names = c("Y", "YTrain", "YHat"))

write.csv(Y_ObsPr, "./BlupYHats.csv")
