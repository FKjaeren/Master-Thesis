# Master-Thesis
Frederik Kjær and Jonas Levin's Master Thesis:

As the datasets were to large to store on out github, the user should by ones self go and download them on this kaggle link, after cloning the repository:
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

In order to reproduce the results presented in the thesis one should:

1.  Run "FindRelevantCustomersItems.py" to find the most relevant records in the raw datasets. This will create raw datasets call  "'Name_of_dataset'_subset.csv".
2.  Run "DataPrep.py". This will do all the necessary preprocing of the data. Creating files in a subfolder in "Data" called "Preprocessed".
3.  Run "FMModelTraining.py", "DeepFMTraining.py", "MLPModelTraining.py", "MF_Model.py" and "BaseLineFactorizationModel.py" in order to train all the models presented in the thesis. These models will be saved in a folder called "Models"
4.  Run "Test_DeepFM.py", "Test_FM.py", "Test_MLP.py", "Test_MF.py" and "Test_baseline_model.py" to get the performance metrics of every model. This will be saved in a .txt file in the folder called "outputs"
6. In order to produce side results like finding cold start problems and train the datadrift detector run: "FindColdStartCases.py" and "Datadrift_exp.py"


