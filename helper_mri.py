import sys
import os
import shutil
import subprocess
import logging

import numpy as np
import nibabel as nib

import pandas as pd
import SimpleITK as sitk
import itk
import skimage

from scipy.signal import medfilt


def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine


def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return


def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)
  
  
# perform the bias field correction
def bias_field_correction(img):
    image = sitk.GetImageFromArray(img)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    corrector.SetMaximumNumberOfIterations([100] * numberFittingLevels)
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected_image_full_resolution)


# rescale the intensity of the image and binning
def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    #remove background pixels by the otsu filtering
    t = skimage.filters.threshold_otsu(volume,nbins=6)
    volume[volume < t] = 0
    
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


# equalize the histogram of the image
def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


# enhance the image
def enhance(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = bias_field_correction(volume)
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

# enhance the image without bias field correction
def enhance_noN4(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

# get the resampled image
def get_resampled_sitk(data_sitk,target_spacing):
    new_spacing = target_spacing

    orig_spacing = data_sitk.GetSpacing()
    orig_size = data_sitk.GetSize()

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
              int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
              int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    res_filter = sitk.ResampleImageFilter()
    img_sitk = res_filter.Execute(data_sitk,
                                new_size,
                                sitk.Transform(),
                                sitk.sitkLinear,
                                data_sitk.GetOrigin(),
                                new_spacing,
                                data_sitk.GetDirection(),
                                0,
                                data_sitk.GetPixelIDValue())

    return img_sitk


# Z-enhance 
def z_enhance_wrapper(image_path,path_to):
    try:
        duck_line = "zscore-normalize "+image_path+"-o " + path_to
        subprocess.getoutput(duck_line)
    except:
        logging.warning('Failed zscore-normalize')
    

#register to a template        
def register_to_template(input_image_path, output_path, fixed_image_path):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            itk.imwrite(result_image, output_path+"/"+image_id)
                
            print("Registered ", image_id)
        except:
            print("Cannot transform", input_image_path.split("/")[-1])
        

    
    