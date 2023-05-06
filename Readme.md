# MRI toolkit
This is a collection of handy helper functions for MRI data analysis. Link to medium post: https://zapaishchykova.medium.com/preprocessing-mri-in-python-4d67c291b8f3

## In summary, your pipeline will look something like this:
- (optional) Converting from DICOM to nifty ( in case you got "raw" data)
- Image registration
- Image rescaling
- (optional) Skull stripping if you are working with brains
- (optional) N4 Bias field correction
- Image normalization, filtering, binning
- (option) converting from 3D to 2D slices

## Quick start
1. Clone repo `git clone`

2. To create an enviroment, run: 
`conda env create -f environment.yml`

3. To activate the enviroment:
`conda activate mri_toolkit`

4. To run the example, open notebook example.ipynb:
`jupyter notebook`


### Data example from Pixar OpenfMRI
MRI data of 3-12 year old children and adults during viewing of a short animated film. The data is available at https://openfmri.org/dataset/ds000228/
