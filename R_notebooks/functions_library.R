combine_all_files_into_one <- function(dir,file_type, pat=NULL){
  #' Reads all files of a given filetype from a given directory and returns a combined dataframe
  library("data.table")
  setwd(dir)
  file_data<-list.files(pattern = pat)
  for (j in 1:length(file_data)){
    if (!exists("dataset")){
      if(file_type=="tsv"){
        dataset <- fread(file_data[j], header=TRUE, sep="\t",stringsAsFactors=FALSE,data.table = FALSE)
        }
      else if(file_type=="csv"){
        dataset <- fread(file_data[j], header=TRUE, sep=",",stringsAsFactors=FALSE,data.table = FALSE)
      }
      if("Image" %notin% colnames(dataset)){
        dataset$Image <- file_data[j]
      }
    }
    else if (exists("dataset")){
      if(file_type=="tsv"){
        temp_dataset <- fread(file_data[j], header=TRUE, sep="\t",stringsAsFactors=FALSE,data.table = FALSE)
      }
      else if(file_type=="csv"){
        temp_dataset <-fread(file_data[j], header=TRUE, sep=",",stringsAsFactors=FALSE,data.table = FALSE)
      }
      
      if(nrow(temp_dataset)>0){
        if("Image" %notin% colnames(temp_dataset)){
          temp_dataset$Image <- file_data[j]
        }
        dataset<-rbind(dataset, temp_dataset)
      }
      rm(temp_dataset)
    }
  }
  return(dataset)
}

# make a not in function
`%notin%` <- Negate(`%in%`)

#Standard error given list

standard_error<-function(x){
  return(sd(x)/length(x))
}

# Tissue level predictions
Tissue_pred_classes<- function(dat){
  Tissues            = levels(as.factor(dat$Image))
  Stages             = levels(as.factor(dat$Stage))
  Tissue_predictions = as.data.frame(matrix(nrow = length(Tissues),
                                          ncol = length(Stages)+6))
  colnames(Tissue_predictions) = c("Image","act_Stage",
                                 paste("n_pred",Stages,sep="_"),"pred_Stage",
                                 "num_nuclei","n_correct_pred","TPR")
  for (i in 1:length(Tissues)) {
    dat_sub                              = subset(dat,dat$Image==Tissues[i])
    freq                                 = as.data.frame((table(dat_sub$predicted_class)))
    Tissue_predictions$Image[i]          = Tissues[i]
    Tissue_predictions$act_Stage[i]      = unique(dat_sub$Stage)
    Tissue_predictions$num_nuclei[i]     = nrow(dat_sub)
    Tissue_predictions$n_correct_pred[i] = sum(dat_sub$Correct_pred)
    Tissue_predictions$TPR[i]            = sum(dat_sub$Correct_pred)/nrow(dat_sub)
    for (j in 1:length(Stages)){
      col_num                       = which(colnames(Tissue_predictions)==paste("n_pred",Stages[j],sep="_"))
      Tissue_predictions[i,col_num] = ifelse(length(subset(freq,freq$Var1==Stages[j])$Freq)>0,
                                             (subset(freq,freq$Var1==Stages[j])$Freq),0)
                                             
    }
    Tissue_predictions$pred_Stage[i]    = as.character(freq$Var1[which(freq$Freq==max(freq$Freq))])
  
  }
  rm(dat_sub,freq,i,j,Tissues,Stages)
  return(Tissue_predictions)
}

Tissue_pred_mgs<- function(dat){
  Tissues            = levels(as.factor(dat$Image))
  Stages             = levels(as.factor(dat$predicted_class))
  Tissue_predictions = as.data.frame(matrix(nrow = length(Tissues),
                                            ncol = length(Stages)+4))
  colnames(Tissue_predictions) = c("Image","act_Stage",
                                   paste("n_pred",Stages,sep="_"),"pred_Stage",
                                   "num_nuclei")
  for (i in 1:length(Tissues)) {
    dat_sub                              = subset(dat,dat$Image==Tissues[i])
    freq                                 = as.data.frame((table(dat_sub$predicted_class)))
    Tissue_predictions$Image[i]          = Tissues[i]
    Tissue_predictions$act_Stage[i]      = unique(dat_sub$Clinical_Stage)
    Tissue_predictions$num_nuclei[i]     = nrow(dat_sub)
    for (j in 1:length(Stages)){
      col_num                       = which(colnames(Tissue_predictions)==paste("n_pred",Stages[j],sep="_"))
      Tissue_predictions[i,col_num] = subset(freq,freq$Var1==Stages[j])$Freq 
    }
    Tissue_predictions$pred_Stage[i]    = as.character(freq$Var1[which(freq$Freq==max(freq$Freq))])
    
  }
  rm(dat_sub,freq,i,j,Tissues,Stages)
  return(Tissue_predictions)
}

#Cross_Validation

require(MASS)
require(caret)


cross_validation_error_lda_mc <- function(data, kfolds) {
  
  ## Define fold_ids
  fold_ids      <- rep(seq(kfolds), ceiling(nrow(data) / kfolds))
  fold_ids      <- fold_ids[1:nrow(data)]
  fold_ids      <- sample(fold_ids, length(fold_ids))
  
  ## Initialize a matrix to store CV error
  CV_error      <- as.data.frame(matrix(ncol=kfolds,nrow = 12))
  ## Loop through the folds
  for (k in 1:kfolds){
    lda_model             <- MASS::lda(Stage ~. , data = data[which(fold_ids != k),]) 
    lda_pred              <- predict(lda_model, data[which(fold_ids == k),])
    class_pred            <- lda_pred$class
    res                   <- caret::confusionMatrix(class_pred,as.factor(data[which(fold_ids == k),]$Stage))
    AUC                   <- pROC::multiclass.roc(as.factor(data[which(fold_ids == k),]$Stage), lda_pred$posterior)
    CV_error[,k]          <- c(colMeans(res$byClass),as.numeric(sub(" curve: *", "", AUC$auc)))
    colnames(CV_error)[k] <- paste("Fold_",k,sep="")
    
  }
  rownames(CV_error)  <- c(colnames(res$byClass),"AUC")
  CV_error$Mean     <- apply(CV_error,1,mean)
  CV_error$StdDev     <- apply(CV_error,1,sd)
  return(CV_error[,which(colnames(CV_error) %in% c("Mean","StdDev"))])
}
cross_validation_error_rf_mc <- function(data, kfolds,rf_ntree,rf_mtry) {
  
  ## Define fold_ids
  fold_ids      <- rep(seq(kfolds), ceiling(nrow(data) / kfolds))
  fold_ids      <- fold_ids[1:nrow(data)]
  fold_ids      <- sample(fold_ids, length(fold_ids))
  
  ## Initialize a matrix to store CV error
  CV_error      <- as.data.frame(matrix(ncol=kfolds,nrow = 12))
  ## Loop through the folds
  for (k in 1:kfolds){
    rf_model             <- randomForest::randomForest(data[which(fold_ids != k),]$Stage ~ ., 
                                                       data[which(fold_ids != k), -ncol(data)],
                                                       ntree=rf_ntree, mtry=rf_mtry, 
                                                       importance=FALSE)
    res                   <- caret::confusionMatrix(rf_model$predicted,
                                                    as.factor(data[which(fold_ids != k),]$Stage))
    AUC                   <- pROC::multiclass.roc((data[which(fold_ids != k),]$Stage), 
                                                  rf_model$votes)
    CV_error[,k]          <- c(colMeans(res$byClass),as.numeric(sub(" curve: *", "", AUC$auc)))
    colnames(CV_error)[k] <- paste("Fold_",k,sep="")
    
  }
  rownames(CV_error)  <- c(colnames(res$byClass),"AUC")
  CV_error$Mean     <- apply(CV_error,1,mean)
  CV_error$StdDev     <- apply(CV_error,1,sd)
  return(CV_error[,which(colnames(CV_error) %in% c("Mean","StdDev"))])
}
cross_validation_error_nn_mc <- function(data, kfolds,nn_layers) {
  
  ## Define fold_ids
  fold_ids      <- rep(seq(kfolds), ceiling(nrow(data) / kfolds))
  fold_ids      <- fold_ids[1:nrow(data)]
  fold_ids      <- sample(fold_ids, length(fold_ids))
  stages_of_cancer_L    <-c("S0_Normal","S1_Hyperplasia",
                            "S2_Fibroadenoma","S3_DCIS",'S4_ILC',
                            "S5_IDC", "S6_Metastatic")
  
  ## Initialize a matrix to store CV error
  CV_error      <- as.data.frame(matrix(ncol=kfolds,nrow = 12))
  ## Loop through the folds
  for (k in 1:kfolds){
    nn_model             <- neuralnet::neuralnet(Stage ~ ., data = data[which(fold_ids != k),], 
                                                 linear.output = FALSE, err.fct = 'ce',
                                                 hidden =nn_layers,
                                                 likelihood = TRUE)
    pred                  <- neuralnet::compute(nn_model, data = data[which(fold_ids != k),])$net.result
    colnames(pred)        <-stages_of_cancer_L
    res                   <- caret::confusionMatrix(as.factor(stages_of_cancer_L[max.col(pred)]),
                                                    as.factor(data[which(fold_ids != k),]$Stage))
    AUC                   <- pROC::multiclass.roc((data[which(fold_ids != k),]$Stage),pred)
    CV_error[,k]          <- c(colMeans(res$byClass),as.numeric(sub(" curve: *", "", AUC$auc)))
    colnames(CV_error)[k] <- paste("Fold_",k,sep="")
    
  }
  rownames(CV_error)  <- c(colnames(res$byClass),"AUC")
  CV_error$Mean     <- apply(CV_error,1,mean)
  CV_error$StdDev     <- apply(CV_error,1,sd)
  return(CV_error[,which(colnames(CV_error) %in% c("Mean","StdDev"))])
}


cross_validation_error_lda <- function(data, kfolds) {
  
  ## Define fold_ids
  fold_ids      <- rep(seq(kfolds), ceiling(nrow(data) / kfolds))
  fold_ids      <- fold_ids[1:nrow(data)]
  fold_ids      <- sample(fold_ids, length(fold_ids))
  
  ## Initialize a matrix to store CV error
  CV_error      <- as.data.frame(matrix(ncol=kfolds,nrow = 12))
  ## Loop through the folds
  for (k in 1:kfolds){
    lda_model             <- MASS::lda(Stage ~. , data = data[which(fold_ids != k),]) 
    lda_pred              <- predict(lda_model, data[which(fold_ids == k),])
    class_pred            <- lda_pred$class
    res                   <- caret::confusionMatrix(class_pred,as.factor(data[which(fold_ids == k),]$Stage))
    AUC                   <- ROCR::performance(ROCR::prediction(lda_pred$posterior[,2],as.factor(data[which(fold_ids == k),]$Stage)), measure = "auc")@y.values[[1]]
    CV_error[,k]          <- c(res$byClass,AUC)
    colnames(CV_error)[k] <- paste("Fold_",k,sep="")
    
  }
  rownames(CV_error)  <- c(names(res$byClass),"AUC")
  CV_error$Mean     <- apply(CV_error,1,mean)
  CV_error$StdDev     <- apply(CV_error,1,sd)
  return(CV_error[,which(colnames(CV_error) %in% c("Mean","StdDev"))])
}


Mechano_Genomic_Score_nuclei <- function(feature_table,path_to_data){
  setwd(path_to_data)
  
  columns_to_drop<-c("V1","label","Image","nucid",'bbox-0', 'bbox-1','bbox-2', 'bbox-3','centroid-0', 'centroid-1','orientation','weighted_centroid-0', 'weighted_centroid-1')
  load('nucleus_pca_model.RData')
  load('Scaling_para_center.RData')
  load('Scaling_para_sd.RData')
  load('LDA_model.RData')
  load('mgs_upper_limit.RData')
  load('mgs_lower_limit.RData')
  
  #Cleaning the data
  allData=feature_table[,-c(which(colnames(feature_table)%in%columns_to_drop))]
  allData<-subset(allData, allData$area>30 & allData$area<=5000 & allData$Int_Median>10 & allData$Int_SD>=2)
  #Take care of NaN and Inf and remove columns with constant values
  allData[sapply(allData, is.infinite)] <- NA
  allData[is.na(allData)] <- 0
  allData=allData[,apply(allData, 2, var, na.rm=TRUE) != 0]
  allData<-allData[,which(colnames(allData) %in% names(pop_avg_feature_value))]
  
  #Scale data
  allData_scaled=sweep(allData, MARGIN=2, pop_avg_feature_value, `-`)
  allData_scaled=sweep(allData_scaled, MARGIN=2, pop_std_feature_value, `/`)
  
  
  # predict_pca_scores
  all_pca_scores<-as.data.frame(predict(nucleus_pca,newdata = allData_scaled))
  all_pca_scores$nucid<-rownames(all_pca_scores)
  all_pca_scores<-merge(all_pca_scores,feature_table[,which(colnames(feature_table) %in% c("nucid","Image"))], by="nucid",all=F)
  rownames(all_pca_scores)<-all_pca_scores$nucid
  
  prediction_all_stages=predict(LDA_normal_vs_cancer, newdata = all_pca_scores)
  MGS_all_stages=as.data.frame(prediction_all_stages$x)
  MGS_all_stages$predicted_class=prediction_all_stages$class
  MGS_all_stages$nucid=rownames(MGS_all_stages)
  MGS_all_stages<-merge(MGS_all_stages,all_pca_scores[,which(colnames(all_pca_scores) 
                                                             %in% c("nucid","Image","Clincal_Stage"))],by="nucid")
  mgs= ifelse(MGS_all_stages$LD1 <=4,MGS_all_stages$LD1,4)
  mgs= ifelse(mgs >= -4,mgs, -4)
  mgs= (mgs-lower)/(upper-lower)
  MGS_all_stages$MGS = mgs
  rm(mgs)
  return(MGS_all_stages)
}


Visualise_MGS <- function(features,TiD){
  #Set the color pallet
  rbPal <- colorRampPalette(c('red3','red1','burlywood','lemonchiffon','skyblue','skyblue2','blue1','blue3'))
  #Subset the data
  data<-subset(features,features$Image == TiD)
  #Set colors
  data$Col <- rbPal(100)[as.numeric(cut(data$MGS,
                                        breaks = seq(0,1,l=100)))]
  
  #plot 
  layout(matrix(1:2,ncol=2), width = c(2.5,0.5),height = c(1,1))
  par(mar=c(1,1,1,1))
  plot(data$`centroid-0`~data$`centroid-1`,pch=19,col=data$Col,cex=0.4,
       ylim=c(max(round(data$`centroid-0`,0)),0),xlab="",ylab="",xaxt='n',yaxt='n')
  #MGS legend
  legend_image <- as.raster(matrix(rbPal(20), ncol=1))
  plot(c(0,2),c(0,1),type = 'n', axes = F,xlab = '', ylab = '', main = 'MGS')
  text(x=1.5, y = seq(0,1,l=5),labels = seq(0,1,l=5))
  rasterImage(legend_image, 1, 1, 0,0)
}

plot_ind_lines_and_Avg<- function(data,feat,Clinical_condition, limits_y=c(0,6), limits_x=c(-0.1,1.1)){
  #' Distribution of a given feature for each TMAs of a given condition
  #'
  #' Provides the  distribution metrics and plot of density curves for each image 
  #'
  #' @param data Dataset
  #' @param feat feature to be plotted
  #' @param Clinical_condition clinical condition of interest
  
  #obtain the right data
  data_sub= subset(data,data$Clinical_Stage==Clinical_condition)
  #Set up the plot
  image = levels(as.factor(data_sub$Image))
  feature = data[,which(colnames(data)==feat)]
  line_colors <- rainbow(length(image))
  distri_charac <- as.data.frame(matrix(nrow =length(image), ncol = 11 ))
  colnames(distri_charac) <-c("Image",c("Mean","Median","Min","Max","SD",
                                                   "CV","CD","IQR","QCD","FWHM"))
  plot (NA, xlim=limits_x, ylim = limits_y, las=1,
        ylab="Prob.Density",xlab=feat, main=Clinical_condition)
  #plot density curves for individual TMAs
  for(index in 1:length(image)){
    temp= data_sub[which(data_sub$Image==image[index]),]
    d<-density(temp[,which(colnames(temp)==feat)])
    lines(d, col=line_colors[index], lwd=0.5)
    distri_charac[index,] <- c(image[index], 
                            distribution_chracteristic(temp[,which(colnames(temp)==feat)]))
    rm(temp,d)
  }
  distri_charac$Clincal_Stage = Clinical_condition
  return(distri_charac)
}

distribution_chracteristic <- function(feature){
  #' Distribution characteristics of a given feature
  #'
  #' Provides the Mean,  median, min, max, standard deviation (SD), Coefficient of Variation (CV),
  #' Coefficient of Dispersion (CD), Inter_Quartile_Range(IQR), Quartile Coeffient of Dispersrion (QCD)
  #' and Full Width Half Maxima
  #'  
  #'
  #' @param feature feature vector of interest
  
  distri_charac         <- rep(NA,12)
  names(distri_charac)  <- c("Mean","Median","Min","Max","SD","CV","CD","IQR","QCD","FWHM","Skewness","Kurtosis")
  distri_charac[1]      <- mean(feature)
  distri_charac[2]      <- median(feature)
  distri_charac[3]      <- min(feature)
  distri_charac[4]      <- max(feature)
  distri_charac[5]      <- sd(feature)
  distri_charac[6]      <- sd(feature)/mean(feature)
  distri_charac[7]      <- var(feature)/mean(feature)
  distri_charac[8]      <- quantile(feature)[4]-quantile(feature)[3]
  distri_charac[9]      <- (quantile(feature)[4]-quantile(feature)[3])/(quantile(feature)[3]+quantile(feature)[4])
  distri_charac[10]     <- FWHM(density(feature))
  distri_charac[11]     <- moments::skewness(feature)
  distri_charac[12]     <- moments::kurtosis(feature)
  
  return(distri_charac)
}

FWHM <- function(density_curve){
  #' Full-width-Half-maxima (FWHM) of a given density curve
  #'
  #' Provides the Full-width-Half-maxima of a curve
  #'
  #' @param density_curve The density curve for which FWHM is to be computed
  xmax  <- density_curve$x[density_curve$y==max(density_curve$y)]
  x1    <- density_curve$x[density_curve$x < xmax][which.min(abs(density_curve$y[density_curve$x < xmax]-max(density_curve$y)/2))]
  x2    <- density_curve$x[density_curve$x > xmax][which.min(abs(density_curve$y[density_curve$x > xmax]-max(density_curve$y)/2))]
  FWHM  <- x2-x1
  return(FWHM)
}
