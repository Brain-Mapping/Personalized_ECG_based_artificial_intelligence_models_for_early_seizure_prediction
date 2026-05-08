Personalized ECG-based artificial intelligence models for early seizure prediction

Harilal Parasuram 1,2,*, +, Mridul Sharma 1, +, Gowtham S 1, +, Harisankar R Menon1 ,Siby Gopinath 1,2, Akshaya Raman 1, Sonu Ravindran 1, Arjun Ramakrishnan3, Garima 3, Priya Bhasimon 1 and Anand Kumar 1,2

1Amrita Advanced Centre for Epilepsy (AACE), Amrita Institute of Medical Sciences, Amrita Vishwa Vidyapeetham, Kochi, Kerala, India.
2Department of Neurology, Amrita Institute of Medical Sciences, Amrita Vishwa Vidyapeetham, Kochi, Kerala, India.
3Department of Biological Sciences and Bioengineering, Indian Institute of Technology, Kanpur, Uttar Pradesh, India

+ authors contributed equally
* Corresponding author: 
Dr. Harilal Parasuram, Amrita Advanced Centre for Epilepsy, Amrita Institute of Medical Science, Amrita Vishwa Vidyapeetham, Kochi, Kerala 682041, India. Email: harilal.navami@gmail.com


This repository contains codes developed as part of a research study focused on early seizure prediction using ECG-derived features and artificial intelligence approaches.

The work explores both machine learning (ML) and deep learning (DL) frameworks for identifying preictal patterns and improving seizure prediction performance in a personalized setting.

Repository Contents:
 1. Data_Computing_Codes: This includes the two Matlab Codes which are used to compute chunks from the source edf file.
    
    1.1 The files included are: Time-based and peak-based signal processing scripts

 2. Model Codes: This includes two subfolders.
    
     2.1. Deep Learning Model Codes: This folder contains Raw signal chunks image generation code and for the same corresponding chunks Scalogram image generation code.
 
     2.2 Machine Learning Model Codes: This Folder Contains the Training and evaluation pipelines for classical ML models

Methods Overview

1. ECG signal preprocessing and segmentation

2. Feature computation

3. Preictal vs interictal classification
