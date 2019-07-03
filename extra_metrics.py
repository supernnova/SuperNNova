import h5py
import numpy as np

"""
Reading unique nights/filter ocurrences for classification
this metric is biased because there points may not be part of the real SN but just before SN
I am using a loose -14 days
"""
file_name = f"tests/dump/processed/DES_database.h5"

hf = h5py.File(file_name,'r+')
idx_testset = np.where(hf['dataset_photometry_2classes'][:]==2)[0].tolist()
for k in ['-2','','+2']:
	for corr in ["","_lcstart"]:
		print('__',k,'__',corr)
		for flt in ['g','r','i','z']:
			mean = round(hf[f'PEAKMJD{k}_num_{flt}{corr}'][idx_testset].mean(),1)
			std = round(hf[f'PEAKMJD{k}_num_{flt}{corr}'][idx_testset].std(),1)
			print(f"{flt}: {mean} +- {std}")
		un_mean = round(hf[f'PEAKMJD{k}_unique_nights{corr}'][idx_testset].mean(),1)
		un_std = round(hf[f'PEAKMJD{k}_unique_nights{corr}'][idx_testset].std(),1)
		print(f'nights: {un_mean} +- {un_std}')
