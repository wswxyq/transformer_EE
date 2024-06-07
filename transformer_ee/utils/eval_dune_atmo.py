import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import binstat



# Check if variable exists (optional)
if 'model_path' in os.environ:  # Replace with the variable name you want to check
    model_path = os.environ['model_path']
    print("model_path : ",model_path)
else:
    print("model_path not found")

filepath = model_path + "/result.npz"

print("Contents of the npz file:")
with np.load(filepath) as file:
  for key in file.keys():
      print(key)  
      
file = np.load(filepath)
trueval = file['trueval']
prediction = file['prediction']

print("trueval shape: ",trueval.shape,prediction.shape)
n_val,dim = trueval.shape

en_true = trueval[:,0]
ct_true = trueval[:,1]
en_pred = prediction[:,0]
ct_pred = prediction[:,1]

binstat.plot_xstat(en_true,en_pred,name=model_path +"/en",title="nu_E",scale='linear',xlabel="en_true",ylabel="en_rec")
binstat.plot_xstat(ct_true,ct_pred,name=model_path +"/ct",title="nu_theta",xlabel="th_true",ylabel="th_rec")
binstat.plot_y_hist(en_pred,name=model_path+"/en_res")
binstat.plot_2d_hist_count(en_true,en_pred,name=model_path +"/en_hist2d",xrange=(0,10),yrange=(0,10),title="nu_E",scale='linear',xlabel="en_true",ylabel="en_rec")
#binstat.plot_2d_hist_count(ct_true,ct_pred,name=model_path +"/ct_hist2d",xrange=(-1.,1.),yrange=(-1.,1.))
binstat.plot_2d_hist_count(ct_true,ct_pred,name=model_path +"/ct_hist2d",xrange=(0.,180.),yrange=(0.,180.),title="nu_theta",xlabel="th_true",ylabel="th_rec")