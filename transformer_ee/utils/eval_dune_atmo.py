import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import binstat

#2-vars
filepath = "/home/tthakore/transformer_EE/save/model/DUNE_atmo/weights/model_d725daa6f63a078b236be8e15e0f6ede/result.npz"

print("Contents of the npz file:")
with np.load(filepath) as file:
  for key in file.keys():
      print(key)  
      
file = np.load(filepath)
trueval = file['trueval']
prediction = file['prediction']

en_true = trueval[:,0]
ct_true = trueval[:,1]
en_pred = prediction[:,0]
ct_pred = prediction[:,1]

binstat.plot_xstat(en_true,en_pred,name="en_binstats.pdf")
binstat.plot_xstat(ct_true,ct_pred,name="ct_binstats.pdf")
binstat.plot_y_hist(en_true)
binstat.plot_2d_hist_count(en_true,en_pred,name="en_hist2d",xrange=(0,100),yrange=(0,100))
binstat.plot_2d_hist_count(ct_true,ct_pred,name="ct_hist2d",xrange=(-1.,1.),yrange=(-1.,1.))