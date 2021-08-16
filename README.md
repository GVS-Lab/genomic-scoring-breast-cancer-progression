# Chromatin as a biomarker for breast cancer progression

Tumor initiation and progression is driven by both intracellular changes within the tumor cells as well as the biophysical properties of the tumor microenvironment. Here we measure nuclear morphology and chromatin organization to characterize the spatial chromatin organization at the single cell level during tumor progression. To gain further insights into the alterations in the local tissue microenvironments, we also compute the nuclear orientations to identify mechanically coupled spatial neighbourhoods that have been posited to drive tumor progression.

### Image derived features
Below is a graphical summary of the features used to measure chromatin architecture. 

<br/> 
<p align="center">
<img src='/nuclear_feat.png' height='250' width='600'>
<br/>

Below is a graphical summary of the tissue/ neighborhhood features computed.

<br/> 
<p align="center">
<img src='/tissue_feat.png' height='400' width='600'>
<br/>

Please refer to the notes_on_feature_extraction for more information on the features and their computation.

### Cell heath score
The aim of this project is to compute a metric, based on the chromatin structures that can distinguish between a normal and pathogenic cell in breast tissues. The developement of one such score, we term mechano-genomic score is explaind in _R_notebooks/Constructing_Mechano_genomic_score.Rmd_. To demonstrate our work we have provided a few sample images [here](https://www.dropbox.com/sh/4cv8u3zoma8xth0/AAB-sUA_sG8Y1q0GHZ1JqSHGa?dl=0) and one can run the notebook _python_notebooks/score_nuclei_given_an_image.ipynb_. For reference, we include the DNA labelled image and the protein expression of standard biomarkers.
