# Comparison-of-Feature-Selection-methods
This repository represents open-source research developed by [Ofir Arbili](https://www.linkedin.com/in/ofir-arbili-82375179/?originalSubdomain=il/), [Dan Presil](https://www.linkedin.com/in/dan-presil-674514152/) and [Or Katz](https://www.linkedin.com/in/or-katz-9ba885114/) for  Ben Gurion University of the Negev machine learning course

# TL;DR
In this work we implemented 3 Features selection algorithms:

1. The Feature Selection using Stochastic Gates is based on the existing open-source code from the authors.
2. FWDT is a combination of preprocessing and an adjustment to reliefF algorithm. We used the implementation of the reliefF algorithm from the following repository and adjusted the code according to the article.
3. The Ensemble algorithm is written in …. [Dan please add]

## Tests scope:
Used 62 datasets, with 5 different sources.

For each FS algorithm we used the following number of datasets:

For Each FS algorithm we used [1,2,5,10,30,50,100] selected k features and 5 different classification algorithms for each fold. Overall we run 180,220 combinations of  Filtering Algorithm, Learning algorithm and Number of features selected (K). Tests. 



# Install
```
pip install -r requirements.txt
pip install -U stg/stg/python/.
```
# Data
Download the data from https://drive.google.com/drive/folders/19XnSh4EvTb6VRKFNzDBk94L_O5gW14H9?usp=sharing \
move the data to ./data/microarrays/data
### recommended
Download the data from https://drive.google.com/drive/folders/1Szy1_kE7XyzIL-6K14jIgN2hwe01UUhv?usp=sharing
temp - Intermediate files containing all the results of the models
output - Contains the preprocessing files
```
unzip temp.zip
unzip output
```

```
Run ./notebooks/process_raw_datasets.ipynb
```
# Run full pipeline
```
cd notebook
toy example - sklearn_pipeline_toy_example.ipynb
full pipeline (CPU version) - Parallel_pipeline_cpu.ipynb
full pipeline (GPU version) - Parallel_pipeline_gpu.ipynb
train best models with pca - Parallel_pipeline_best.ipynb
show the results - results.ipynb
```
# Run fast

```
full pipeline (Parallel) 
python3 main_kfold.py --filtering STG --n_job 64  
python3 main_kfold.py --filtering new_STG --n_job 64
python3 main_kfold.py --filtering f_classif --n_job 64
python3 main_kfold.py --filtering mrmr --n_job 64
python3 main_kfold.py --filtering reliefF --n_job 64
python3 main_kfold.py --filtering RFE_SVM --n_job 64
python3 main_kfold.py --filtering new_FWDT --n_job 64
python3 main_kfold.py --filtering ensemble --n_job 64
python3 main_kfold.py --filtering new_ensemble --n_job 64
```

# Algo explain

# Results
AUC ranking chart (lower means better ranking)
![alt text](https://github.com/OrKatz7/Comparison-of-Feature-Selection-methods/blob/main/docs/raniking.png)

# Improve
Since the main concept of this algorithm is using simple gates in order to define which feature is relevant. Since the feature selection module is simple and based on a linear layer with gaussian distribution, our improvement was focusing on condense the module by adding attention and canceling out layers. In addition, we changed the activation function to sigmoid with the understanding it's more robust to the cancel out layer.

In comparing the best STG results per dataset with the best new_STG results, we see an improvement of 6.5% in the AUC and 1.8 in the ACC. Examining the results for specific k selected features reveals even greater improvement. For example, with 5 features selected, the ACC improvement is 11.3%. By adjusting the STG algorithm, we improve overall results and refine the information for the features as well.

![alt text](https://github.com/OrKatz7/Comparison-of-Feature-Selection-methods/blob/main/docs/improve.png)



# References

https://github.com/runopti/stg

https://runopti.github.io/stg/

https://arxiv.org/abs/1706.03762?context=cs

https://github.com/unnir/CancelOut


# license - MIT
