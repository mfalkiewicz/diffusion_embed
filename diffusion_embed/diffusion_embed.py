from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import erf
from .due import due, Doi

from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

import h5py
import os

def get_atlas(name):
    if name == "destrieux_2009":
        atlas = datasets.fetch_atlas_destrieux_2009()
        atlas_filename = atlas['maps']
    elif name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_filename = atlas['maps']
    elif name == "aal":
        atlas = datasets.fetch_atlas_aal()
        atlas_filename = atlas['maps']
    elif name == "smith_2009":
        atlas = datasets.fetch_atlas_smith_2009()
        atlas_filename = atlas['rsn70']
    else:
        raise ValueError('Atlas name unkown')
    return atlas_filename

def extract_correlation_matrix(data_filename, confounds_filename, atlas_name = "destrieux_2009", correlation_type = 'correlation'):
    
    atlas_filename = get_atlas(atlas_name)
    #labels = atlas['labels']
    
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    time_series = masker.fit_transform(data_filename,
                                   confounds= confounds_filename)
    correlation_measure = ConnectivityMeasure(kind=correlation_type)
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]  
    
    return correlation_matrix

def compute_matrices(data_volumes, confounds, subject_ids, runs, output_dir = os.getcwd()):
    #get confounds
    #get data files
    
#    data_name = data.func[0]
#    confounds_name = data.confounds
    f = h5py.File(output_dir + os.sep + "affinity-matrices.hdf5", "w")
    
    for i in xrange(len(data_volumes)):
        subject = subject_ids[i]
        run = runs[i]
        data_name = data_volumes[i]
        confounds_name = confounds[i]
        for atlas in ["destrieux_2009", "harvard_oxford", "aal"]:
            print "using atlas "+atlas
            for affinity_measure in ["correlation", "partial correlation", "precision"]:
                print "computing "+affinity_measure+" matrix"
                sgrp = f.create_group(subject+"/"+run+"/"+atlas+"/"+affinity_measure)
                correlation_matrix = extract_correlation_matrix(data_name, confounds_name, atlas_name = atlas, correlation_type=affinity_measure)
                dset = sgrp.create_dataset("affinity_matrix", correlation_matrix.shape, dtype=correlation_matrix.dtype)
                dset[...]=correlation_matrix
    f.close()

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='diffusion_embed')


