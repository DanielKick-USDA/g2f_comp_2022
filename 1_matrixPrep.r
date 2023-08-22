library('tidyverse')
library("stringr")
library('purrr')

library('foreach')
library('magrittr')
library('ggplot2')
# library(BGLR)

restrict_to_dist_from_2022 = TRUE
restrict_to_geno = TRUE
restrict_to_2022_hybrids = TRUE
save_suffix = 'small'
CV = 2014

needed_matrices = c("GPC",
                    "GA",
                    "GD",
                    "K.Soil", 
                    "K.Weather"
)

g_path = './data/Maize_GxE_Competition_Data/Training_Data/5_Genotype_Data_All_Years.vcf/'
AMat_file =           'Centered_IBS_Imputed_Imputed_5_Genotype_Data_All_Years_pres80maf05mbp100_KNNimp_with_Probability.txt'
DMat_file = 'Dominance_Centered_IBS_Imputed_Imputed_5_Genotype_Data_All_Years_pres80maf05mbp100_KNNimp_with_Probability.txt'
GPC_file =                      'PC_Imputed_Imputed_5_Genotype_Data_All_Years_pres80maf05mbp100_KNNimp_with_Probability.txt'

list.files('./data/Processed/')



data_loc <- './data/Processed/'

phno <- read.csv(paste0(data_loc, 'phno0.csv'))
phno <- phno[, c("Env", "Hybrid", "Year", "Yield_Mg_ha")]
meta <- read.csv(paste0(data_loc, 'meta0.csv'))
soil <- read.csv(paste0(data_loc, 'soil0.csv'))
wthr <- read.csv(paste0(data_loc, 'wthrWide0.csv'), skip = 1)
# cgmv <- read.csv(paste0(data_loc, 'cgmv0.csv'))


# Mask out of fold obeservations
# phno[phno$Year != CV, "Yield_Mg_ha"] <- NA

# What's the smallest set that could be useful?

# ignore imputation
# in hybrid, in env

# list.files('./data/Preparation/')

disallow_miss_env <- function(dfname = 'wthr'){
    prep_path <- './data/Preparation/'
    expect_csv <- paste0(dfname, '_Envs_miss.csv')

    if(expect_csv %in% list.files(prep_path)){
        return(read.csv(paste0(prep_path, expect_csv)))
    }else{
        print('File not found.')
    }
}

disallow_imp <- function(dfname = 'wthr'){
    files_found <- list.files(prep_path)
    detections <- stringr::str_detect(files_found, paste0(dfname, '_Envs_imp_.+'))
    files_matched <- files_found[detections]

    out_list <- purrr::map(files_matched, function(file_matched){
        out = read.csv(paste0(prep_path, file_matched))

        colname = stringr::str_extract(file_matched, '_imp_.+')
        colname = stringr::str_replace(unlist(colname), '_imp_', '')
        colname = stringr::str_replace(unlist(colname), '\\.csv', '')

        out[['Imputed']] = colname
        return(out)        
    })
    
    out <- do.call(rbind, out_list)
    return(out)
}
# NULL == disallow_imp(dfname = 'wthr')
# disallow_imp(dfname = 'meta')

# # restrict weather -- This makes sense for othe kinds of data but not wthr 
# # since it's downloaded from POWER. Retained as an illustrative template

# restrict_to_nomiss_wthr = TRUE

# rm_Envs = disallow_miss_env(dfname = 'wthr')
# # confirm that no 2020 data is being removed
# list_Envs <- rm_Envs[['Absent_Envs']]
# list_2022_Envs <- 
# stopifnot({function (x) x == 0} (length(list_2022_Envs)))

# # Remove disallowed envs
# phno <- phno[!(phno$Env %in% list_Envs), ]

dim(phno)

# restrict_to_dist_from_2022 = TRUE

# with distance of 1 this only has a minor affect 146,057 -> 132,925
min_dist = 1

if(restrict_to_dist_from_2022){
    latlons <- meta[, c('Env', 'Year', 'Latitude_of_Field', 'Longitude_of_Field')] %>% distinct()
    latlons2022 <- latlons[latlons$Year == 2022, ]
    latlons['Pass'] = FALSE
    latlons2022['Distance'] = FALSE

    for(i in seq(1, nrow(latlons))){
        lat <- latlons[i, 'Latitude_of_Field']
        lon <- latlons[i, 'Longitude_of_Field']

        latlons2022['Distance'] <- sqrt((latlons2022['Latitude_of_Field'] - lat)**2 + (latlons2022['Longitude_of_Field'] - lon)**2)

        if(min(latlons2022['Distance']) <= min_dist){
              latlons[i, 'Pass'] <- TRUE
        }
    }

    pass_Envs <-unique(latlons[latlons$Pass, 'Env'])
    fail_Envs <-unique(latlons[!(latlons$Pass), 'Env'])

    # Confirm this isn't removing any 2022 observations
    stopifnot(!(TRUE %in% stringr::str_detect(fail_Envs, '2022')))

    phno <- phno[phno$Env %in% pass_Envs, ]   
}

dim(phno)

# restrict_to_2022_hybrids = TRUE

# Remove observations for hybrids not in 2022
# Note: Doing this _DRAMATICALLY_ reduces the number of observations
# 146,057 -> 25,030

if (restrict_to_2022_hybrids){
    hybrids_in_2022 <- unique(phno[phno$Year == 2022, 'Hybrid'] )
    phno <- phno[phno$Hybrid %in% hybrids_in_2022, ] 
}

dim(phno)

# restrict_to_geno = TRUE

if(restrict_to_geno){
    # PCA info
    geno <- read.table(paste0(
        g_path,    
        GPC_file
    ), skip = 2, header = TRUE)

    temp <- phno[!(phno$Hybrid %in% geno[['Taxa']]), 'Env']
    # assert there are no values being removed from 2022
    stopifnot(!(TRUE %in% stringr::str_detect(unique(temp), '2022')))

    phno <- phno[(phno$Hybrid %in% geno[['Taxa']]), ] 
    rm(list = c('geno'))
} 

dim(phno)

Y <- as.matrix(phno['Yield_Mg_ha'])
# Y 
# mask based on CV
Y[phno$Year == CV] <- NA

ystd = sd(Y, na.rm = T)
ybar = mean(Y, na.rm = T)

Y = (Y - ybar)/ystd





# TODO





# if('GA' %in% needed_matrices){
#     gc()
#     geno <- read.table(paste0(
#         g_path,    
#         AMat_file
#     ), skip = 3, header = FALSE)

#     # make a nxn matrix to hold the A matrix
#     geno_hybrid_index = geno[['V1']]

#     Gdat = as.matrix(geno[, 2:ncol(geno)])
    
#     rownames(Gdat) = geno_hybrid_index
#     colnames(Gdat) = geno_hybrid_index

#     AMat <- Gdat[phno$Hybrid, phno$Hybrid]
#     rm(list = c('Gdat', 'geno'))
#     gc()    
# }


# if('GD' %in% needed_matrices){
#     gc()
#     geno <- read.table(paste0(
#         g_path,    
#         DMat_file
#     ), skip = 3, header = FALSE)

#     # make a nxn matrix to hold the A matrix
#     geno_hybrid_index = geno[['V1']]

#     Gdat = as.matrix(geno[, 2:ncol(geno)])
    
#     rownames(Gdat) = geno_hybrid_index
#     colnames(Gdat) = geno_hybrid_index

#     DMat <- Gdat[phno$Hybrid, phno$Hybrid]
#     rm(list = c('Gdat', 'geno'))
#     gc()    
# }

# if('K.soil' %in% needed_matrices){
#     soil_Envs <- soil$Env

#     soil_cols <- names(soil)[!(names(soil) %in% c('X', 'Env', 'Year'))]

#     scalings = do.call(rbind, purrr::map(soil_cols, function(col){
#         data.frame(col = c(col),
#                    mean = mean(soil[soil$Year != CV, col], na.rm=TRUE),
#                    std = sd(soil[soil$Year != CV, col], na.rm=TRUE))    
#     }) )
#     # TODO allow for saving of this df
#     # scalings
#     for(col in scalings$col){
#         soil[col] <- ((soil[col]-scalings[scalings$col == col, 'mean'])/scalings[scalings$col == col, 'std'])
#     }

#     Sdat <- as.matrix(soil[, soil_cols])
#     rownames(Sdat) <- soil_Envs

#     K.soil <- tcrossprod(Sdat)
#     K.soil <- K.soil/mean(diag(K.soil))
#     K.soil <- K.soil[phno$Env, phno$Env]
#     rm(list = c('Sdat'))
#     gc()   
# }

# if('K.weather' %in% needed_matrices){
#     # wthr_Envs <- wthr$X # because wthr wide has two levels of indexing when I skip the 
#     # top level Env gets cut. 
#     wthr_Envs_Planted <- paste(wthr$X, wthr$X.1, sep = '__')

#     # wthr_Envs
#     weather_ECs <- c(
#         'QV2M', 'T2MDEW', 'PS', 'RH2M', 'WS2M', 'GWETTOP', 'ALLSKY_SFC_SW_DWN', 
#         'ALLSKY_SFC_PAR_TOT', 'T2M_MAX', 'T2M_MIN', 'T2MWET', 'GWETROOT', 'T2M', 
#         'GWETPROF', 'ALLSKY_SFC_SW_DNI', 'PRECTOTCORR')

#     # keep only ecs (have format EC_Day# )
#     Wdat <- wthr[, names(wthr)[!(names(wthr) %in% c('variable', 'X'))]]



#     fix_col_day_numbers <- function(colname = 'ALLSKY_SFC_PAR_TOT_Day1'){
#         if(stringr::str_detect(colname, 'Day\\d+$')){
#             daynum <- stringr::str_extract(colname, '\\d+$')
#             if (as.numeric(daynum) >= 100){
#                 daynum <- daynum
#             } else if (as.numeric(daynum) >= 10){
#                 daynum <- paste0('0', daynum)
#             } else if (as.numeric(daynum) >= 1){
#                 daynum <- paste0('00', daynum)
#             }
#             return(stringr::str_replace(colname, '\\d+$', daynum))
#         } else {
#             return(colname)
#         }
#     }

#     # make sure the ECs are in a reasonable order
#     newNames <- unlist(map(names(Wdat), function(e){fix_col_day_numbers(colname = e)}))
#     names(Wdat) <- newNames
#     Wdat <- Wdat[, sort(newNames)]

#     Wdat <- as.matrix(Wdat)

#     rownames(Wdat) <- wthr_Envs_Planted

#     # Wdat#[, weather_ECs]



#     # env.weather is a list of 8 objects
#     # each object contains a matrix of 145 measurements x nObs
#     # these values are drawn from the input df
#     env.weather <- setNames(
#       foreach(weather_EC=weather_ECs) %do% {

#         # get the cols for each day's reading in form 
#         # "max_temp1", "max_temp2" ... max_temp145"
#         indx_of_weather_daily_vals = grep(paste0("^", weather_EC, "_"), colnames(Wdat)) # <- Note: this underscore is critical to prevent "P" from not matching with photoperiod and PAR too. 

#         # this is in nObs x Daily value (145)
#         E <- as.matrix(
#           Wdat[, indx_of_weather_daily_vals]) %>% 
#           t # now it's Daily value x nObs

#         # New as of 2022 12 22 This forces the dates to be in the right order.
#         # i.e. so foo1 foo10, foo100... is replaced by foo1, foo2, foo3 ...
#         E <- E[sort(rownames(E)),]
#       }, 
#       weather_ECs) # setNames() makes the matrices in this list accessible by name

#     make_ERM <- function(E_list, w, summary_functions = NULL) {
#       # create a list of nObsxnObs matrices and `.combine` them by adding the matrices together
#       # this results in a single nObsxnObs matrix aggregating all the enviromental covariates.

#       foreach(E = E_list, .combine = "+") %do% {
#         # Time bins
#         windows <- cut_interval(1:nrow(E), length = w) # this function is from ggplot2
#         # each 3 day period gets a new group:
#         # [1] [0,3] [0,3] [0,3]     (3,6] (3,6] (3,6]     (6,9] ...  
#         # [145] (144,147]
#         # Why this isn't done with seq is beyond me. Possibly because it returns cuts?

#         if (length(unique(windows)) == 1) {
#           Z <- matrix(1, nrow = nrow(E), ncol = 1)
#           # if the window length is one return a single column of ones

#         } else {
#           Z <- model.matrix( ~ windows)
#           # for window size three this becomes

#           #   intercept win2 win3 ...
#           # 1         1    0    0      # The first group is the intercept
#           # 2         1    0    0
#           # 3         1    0    0
#           # 4         1    1    0      # One hot encoded groups
#           # 5         1    1    0
#           # 6         1    1    0
#           # 7         1    0    1
#           # 8         1    0    1
#           # 9         1    0    1
#           # ...
#         }

#         # # Average by time bin
#         # EC <- crossprod(Z,    # 145x49   window design matrix
#         #                 E     # 145xnObs data matrix
#         # ) %>% # 49xnObs 
#         #   t %>%          # flip nObsx49
#         #   scale %>%      # center and scale each column
#         #   t %>%          # flip 49xnObs
#         #   na.omit

#         # Updated processing: The adapted code did not account for using 
#         # management data where there might be no variability across groups (all
#         # groups recieve no fertilizer on a given day). This adapted version 
#         # selectively scales entries to avoid these days becoming NA and being 
#         # removed.

#         # Average by time bin
#         EC <- crossprod(Z,    # 145x49   window design matrix
#                         E     # 145xnObs data matrix
#         ) %>% # 49xnObs
#           t #%>%          # flip nObsx49
#         # scale %>%      # center and scale each column
#         # replaced because this is introducing nas into 0s for `N`, `P` where columns are all 0 (i.e. when no fertilizer was applied)
#         # scale each column one at a time if and only if the sd != 0
#         # This prevents values from becoming NA and thus keeps the dimensions to the expected size
#         for(i in seq(1, ncol(EC))){
#           if(sd(EC[,i]) != 0){
#             EC[,i] = scale(EC[,i])
#           } else {
#             EC[,i] = scale(EC[,i], scale = FALSE)
#           }
#         }
#         # and return to processing as normal
#         EC <- EC %>%
#           t %>%          # flip 49xnObs
#           na.omit


#         # Summary by time bin
#         # This functionality is not used
#         if (!is.null(summary_functions)) {
#           EC_summary <- foreach(summary_function = summary_functions) %do% {
#             summary_by_window <-
#               by(E, windows, function(x)
#                 apply(x, 2, summary_function))

#             do.call(rbind, summary_by_window) %>% t %>% scale %>% t %>% na.omit
#           } %>% as.list

#           EC <- do.call(rbind, append(list(EC), EC_summary))
#         }
#         # Environmental relationship matrix
#         return(crossprod(EC)) # Return a nObsxnObs matrix
#       }
#     }

#     K.weather <- make_ERM(env.weather, w = 3) # Make an environmental relationship matrix of nObs x nObs
#     K.weather <- K.weather/mean(diag(K.weather))
#     gc()

#     dim(K.weather)

#     # Expand matrix to be the target size

#     # there are some locations that have multiple planting dates for the same hybrids
#     # e.g. ARH1_2016	A3G-3-3-1-313/DK3IIH6
#     # Due to time constraints I'm going to take the earliest for each Env x Hybrid and 
#     # accept that there will be some error introduced.
#     # meta[, c("Env", "Hybrid", 'Date_Planted')]  %>% distinct() %>% group_by(Env, Hybrid) %>% tally()

#     meta_min <- meta[, c("Env", "Hybrid", 'Date_Planted')]  %>% 
#         distinct() %>% 
#         group_by(Env, Hybrid) %>% 
#         mutate(Date_Planted = min(Date_Planted, na.rm = TRUE)) %>% 
#         mutate(Date_Planted = round(Date_Planted)) %>% 
#         mutate(Date_Planted = as.character(Date_Planted)) %>% 

#         distinct() %>% 
#         ungroup()


#     wthr_matches <- dplyr::left_join(
#         phno, 
#         meta_min, by = c('Env', 'Hybrid')) %>% 
#         mutate(Match_String = paste(Env, Date_Planted, sep = '__'))


#     # There are a bunch of off by one errors in the names. This is the quick and
#     # rough fix
#     convert_Env_Planting <- data.frame(
#         inPhno =      c("DEH1_2022__120", "GAH2_2022__116", "IAH1_2022__132", "IAH3_2022__134", "ILH1_2022__132", "INH1_2022__133", "MIH1_2022__138", "MOH2_2022__131", "NCH1_2022__112", "NEH2_2022__135", "NEH3_2022__135", "NYH2_2022__144", "NYH3_2022__144", "TXH1_2022__107", "TXH2_2022__107", "TXH3_2022__107", "WIH2_2022__137", "WIH3_2022__137"),
#         inK.weather = c('DEH1_2022__121', 'GAH2_2022__117', 'IAH1_2022__133', 'IAH3_2022__135', 'ILH1_2022__133', 'INH1_2022__134', 'MNH1_2022__137', 'MOH2_2022__132', 'NCH1_2022__113', 'NEH2_2022__136', 'NEH3_2022__136', 'NYH2_2022__145', 'NYH3_2022__145', 'TXH1_2022__108', 'TXH2_2022__108', 'TXH3_2022__108', 'WIH2_2022__138', 'WIH3_2022__138')    
#     )

#     for(i in 1:nrow(convert_Env_Planting)){
#         mask <- wthr_matches$Match_String == convert_Env_Planting[i, 'inK.weather']
#         wthr_matches[mask, 'Match_String'] <- convert_Env_Planting[i, 'inPhno']
#     }
#     # assert all are in wthr_matches / K.weather
#     stopifnot(0==(wthr_matches$Match_String[!(wthr_matches$Match_String %in% rownames(K.weather))] %>% unique()))

#     K.weather <- K.weather[wthr_matches$Match_String, wthr_matches$Match_String]
    
           

#     rm(list = c('Wdat'))
#     gc()   
# }


# dim(K.weather)

if(TRUE){
    ## W (ERM) matrix ------------------------------------------------------------
    matrix_file <- paste0("y_matrix", "_", as.character(CV), "_", save_suffix, '.rds')        
    matrix_file_path = paste0("./data/Processed/", matrix_file)
    if (matrix_file %in% list.files("./data/Processed/")){
        doNothing <- TRUE
    } else {
        Y <- as.matrix(phno['Yield_Mg_ha'])
        # Y 
        # mask based on CV
        Y[phno$Year == CV] <- NA

        ystd = sd(Y, na.rm = T)
        ybar = mean(Y, na.rm = T)

        Y = (Y - ybar)/ystd
        saveRDS(Y , file = matrix_file_path)
     }
}


if("GA" %in% needed_matrices){
    ## W (ERM) matrix ------------------------------------------------------------
    matrix_file <- paste0("GA_matrix", "_", as.character(CV), "_", save_suffix, '.rds')        
    matrix_file_path = paste0("./data/Processed/", matrix_file)
    if (matrix_file %in% list.files("./data/Processed/")){
        doNothing <- TRUE
    } else {
        gc()
        geno <- read.table(paste0(
            g_path,    
            AMat_file
        ), skip = 3, header = FALSE)

        # make a nxn matrix to hold the A matrix
        geno_hybrid_index = geno[['V1']]

        Gdat = as.matrix(geno[, 2:ncol(geno)])

        rownames(Gdat) = geno_hybrid_index
        colnames(Gdat) = geno_hybrid_index

        AMat <- Gdat[phno$Hybrid, phno$Hybrid]
        saveRDS(AMat , file = matrix_file_path)
        rm(list = c('Gdat', 'geno'))
        gc()    
     }
}


if("GD" %in% needed_matrices){
    ## W (ERM) matrix ------------------------------------------------------------
    matrix_file <- paste0("GD_matrix", "_", as.character(CV), "_", save_suffix, '.rds')        
    matrix_file_path = paste0("./data/Processed/", matrix_file)
    if (matrix_file %in% list.files("./data/Processed/")){
        doNothing <- TRUE
    } else {
        gc()
        geno <- read.table(paste0(
            g_path,    
            DMat_file
        ), skip = 3, header = FALSE)

        # make a nxn matrix to hold the A matrix
        geno_hybrid_index = geno[['V1']]

        Gdat = as.matrix(geno[, 2:ncol(geno)])

        rownames(Gdat) = geno_hybrid_index
        colnames(Gdat) = geno_hybrid_index

        DMat <- Gdat[phno$Hybrid, phno$Hybrid]
        saveRDS(Gdat , file = matrix_file_path)
        rm(list = c('Gdat', 'geno'))
        gc()    
    }
}


if("K.Soil" %in% needed_matrices){
    ## W (ERM) matrix ------------------------------------------------------------
    matrix_file <- paste0("Ksoil_matrix", "_", as.character(CV), "_", save_suffix, '.rds')        
    matrix_file_path = paste0("./data/Processed/", matrix_file)
    if (matrix_file %in% list.files("./data/Processed/")){
        doNothing <- TRUE
    } else {
        soil_Envs <- soil$Env

        soil_cols <- names(soil)[!(names(soil) %in% c('X', 'Env', 'Year'))]

        scalings = do.call(rbind, purrr::map(soil_cols, function(col){
            data.frame(col = c(col),
                       mean = mean(soil[soil$Year != CV, col], na.rm=TRUE),
                       std = sd(soil[soil$Year != CV, col], na.rm=TRUE))    
        }) )
        # TODO allow for saving of this df
        # scalings
        for(col in scalings$col){
            soil[col] <- ((soil[col]-scalings[scalings$col == col, 'mean'])/scalings[scalings$col == col, 'std'])
        }

        Sdat <- as.matrix(soil[, soil_cols])
        rownames(Sdat) <- soil_Envs

        K.soil <- tcrossprod(Sdat)
        K.soil <- K.soil/mean(diag(K.soil))
        K.soil <- K.soil[phno$Env, phno$Env]
        
        saveRDS(K.soil , file = matrix_file_path)
        rm(list = c('Sdat'))
        gc()  
    }
}


if("K.Weather" %in% needed_matrices){
    ## W (ERM) matrix ------------------------------------------------------------
    matrix_file <- paste0("Kweather_matrix", "_", as.character(CV), "_", save_suffix, '.rds')        
    matrix_file_path = paste0("./data/Processed/", matrix_file)
    if (matrix_file %in% list.files("./data/Processed/")){
        doNothing <- TRUE
    } else {
        # wthr_Envs <- wthr$X # because wthr wide has two levels of indexing when I skip the 
        # top level Env gets cut. 
        wthr_Envs_Planted <- paste(wthr$X, wthr$X.1, sep = '__')

        # wthr_Envs
        weather_ECs <- c(
            'QV2M', 'T2MDEW', 'PS', 'RH2M', 'WS2M', 'GWETTOP', 'ALLSKY_SFC_SW_DWN', 
            'ALLSKY_SFC_PAR_TOT', 'T2M_MAX', 'T2M_MIN', 'T2MWET', 'GWETROOT', 'T2M', 
            'GWETPROF', 'ALLSKY_SFC_SW_DNI', 'PRECTOTCORR')

        # keep only ecs (have format EC_Day# )
        Wdat <- wthr[, names(wthr)[!(names(wthr) %in% c('variable', 'X'))]]



        fix_col_day_numbers <- function(colname = 'ALLSKY_SFC_PAR_TOT_Day1'){
            if(stringr::str_detect(colname, 'Day\\d+$')){
                daynum <- stringr::str_extract(colname, '\\d+$')
                if (as.numeric(daynum) >= 100){
                    daynum <- daynum
                } else if (as.numeric(daynum) >= 10){
                    daynum <- paste0('0', daynum)
                } else if (as.numeric(daynum) >= 1){
                    daynum <- paste0('00', daynum)
                }
                return(stringr::str_replace(colname, '\\d+$', daynum))
            } else {
                return(colname)
            }
        }

        # make sure the ECs are in a reasonable order
        newNames <- unlist(map(names(Wdat), function(e){fix_col_day_numbers(colname = e)}))
        names(Wdat) <- newNames
        Wdat <- Wdat[, sort(newNames)]

        Wdat <- as.matrix(Wdat)

        rownames(Wdat) <- wthr_Envs_Planted

        # Wdat#[, weather_ECs]



        # env.weather is a list of 8 objects
        # each object contains a matrix of 145 measurements x nObs
        # these values are drawn from the input df
        env.weather <- setNames(
          foreach(weather_EC=weather_ECs) %do% {

            # get the cols for each day's reading in form 
            # "max_temp1", "max_temp2" ... max_temp145"
            indx_of_weather_daily_vals = grep(paste0("^", weather_EC, "_"), colnames(Wdat)) # <- Note: this underscore is critical to prevent "P" from not matching with photoperiod and PAR too. 

            # this is in nObs x Daily value (145)
            E <- as.matrix(
              Wdat[, indx_of_weather_daily_vals]) %>% 
              t # now it's Daily value x nObs

            # New as of 2022 12 22 This forces the dates to be in the right order.
            # i.e. so foo1 foo10, foo100... is replaced by foo1, foo2, foo3 ...
            E <- E[sort(rownames(E)),]
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

#         dim(K.weather)

        # Expand matrix to be the target size

        # there are some locations that have multiple planting dates for the same hybrids
        # e.g. ARH1_2016	A3G-3-3-1-313/DK3IIH6
        # Due to time constraints I'm going to take the earliest for each Env x Hybrid and 
        # accept that there will be some error introduced.
        # meta[, c("Env", "Hybrid", 'Date_Planted')]  %>% distinct() %>% group_by(Env, Hybrid) %>% tally()

        meta_min <- meta[, c("Env", "Hybrid", 'Date_Planted')]  %>% 
            distinct() %>% 
            group_by(Env, Hybrid) %>% 
            mutate(Date_Planted = min(Date_Planted, na.rm = TRUE)) %>% 
            mutate(Date_Planted = round(Date_Planted)) %>% 
            mutate(Date_Planted = as.character(Date_Planted)) %>% 

            distinct() %>% 
            ungroup()


        wthr_matches <- dplyr::left_join(
            phno, 
            meta_min, by = c('Env', 'Hybrid')) %>% 
            mutate(Match_String = paste(Env, Date_Planted, sep = '__'))


        # There are a bunch of off by one errors in the names. This is the quick and
        # rough fix
        convert_Env_Planting <- data.frame(
            inPhno =      c("DEH1_2022__120", "GAH2_2022__116", "IAH1_2022__132", "IAH3_2022__134", "ILH1_2022__132", "INH1_2022__133", "MIH1_2022__138", "MOH2_2022__131", "NCH1_2022__112", "NEH2_2022__135", "NEH3_2022__135", "NYH2_2022__144", "NYH3_2022__144", "TXH1_2022__107", "TXH2_2022__107", "TXH3_2022__107", "WIH2_2022__137", "WIH3_2022__137"),
            inK.weather = c('DEH1_2022__121', 'GAH2_2022__117', 'IAH1_2022__133', 'IAH3_2022__135', 'ILH1_2022__133', 'INH1_2022__134', 'MNH1_2022__137', 'MOH2_2022__132', 'NCH1_2022__113', 'NEH2_2022__136', 'NEH3_2022__136', 'NYH2_2022__145', 'NYH3_2022__145', 'TXH1_2022__108', 'TXH2_2022__108', 'TXH3_2022__108', 'WIH2_2022__138', 'WIH3_2022__138')    
        )

        for(i in 1:nrow(convert_Env_Planting)){
            mask <- wthr_matches$Match_String == convert_Env_Planting[i, 'inK.weather']
            wthr_matches[mask, 'Match_String'] <- convert_Env_Planting[i, 'inPhno']
        }
        # assert all are in wthr_matches / K.weather
        stopifnot(0==(wthr_matches$Match_String[!(wthr_matches$Match_String %in% rownames(K.weather))] %>% unique()))

        K.weather <- K.weather[wthr_matches$Match_String, wthr_matches$Match_String]

        saveRDS(K.weather , file = matrix_file_path)

        rm(list = c('Wdat', 'K.weather'))
        gc()   
    }
}

matrix_file <- paste0("phno_ref", "_", as.character(CV), "_", save_suffix, '.csv')        
matrix_file_path = paste0("./data/Processed/", matrix_file)
write.csv(phno, matrix_file_path)


