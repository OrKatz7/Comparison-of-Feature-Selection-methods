#for test the pipeline add --test

python3 main_kfold.py --filtering STG --n_job 64  
python3 main_kfold.py --filtering new_STG --n_job 64
python3 main_kfold.py --filtering f_classif --n_job 64
python3 main_kfold.py --filtering mrmr --n_job 64
python3 main_kfold.py --filtering reliefF --n_job 64
python3 main_kfold.py --filtering RFE_SVM --n_job 64
python3 main_kfold.py --filtering new_FWDT --n_job 64
python3 main_kfold.py --filtering ensemble --n_job 64
python3 main_kfold.py --filtering new_ensemble --n_job 64
