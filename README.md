# An automated approach to optimize working windows for VCG features 


This archive contains all computational steps used to produce results in the upcoming paper: 
**Automatically Optimized Vectorcardiographic Features are Associated with Recurrence of Atrial Fibrillation after Electrical Cardioversion** (under revision now).

Although we made codes publically available, we are not authorized to publish source ECG and clinical data (provided codes will not produce results without them).

However, all computational code is included and we added results of several steps, if these results did not contain single subject-specific data.
Therefore, sub-folders like **figs_maps** contain all generated maps, including raw and smoothed version as well. 
It also contains final KDE maps and raw form of generated tables, before being formatted in the MS Excell (folder **results_data**).
Subfolder **results** contain other raw images used in the paper (usually after processing in Adobe Illustrator).

We had to change few lines of code in three files in places containing sensitive information - these places usually fixed data - miss-spelled name from one table did not fit name in other table etc. These places are commented and contain keyword "ANONYMIZED" in the commented block.  

The following table shows all computation sub-steps and their description. These sub-steps should be run consecutively because most of them produce results
for a following step. Essential steps (topic of the paper) are **bold**, steps build just for our further insight are _italic_. 
When description refers to a Figure, it means it was exported as an SVG/PNG file, improved for visual appearance in Adobe illustrator, and exported for manuscript as a raster image.  

| Script name                                                                            | Description                                                                                                                                                                        |
|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _Preprocessing and conversion:_                                                        |                                                                                                                                                                                    |
| [step_0_read_and_convert_data.py](step_0_read_and_convert_data.py)                     | Build single dataset from available patient ECGs and Outcome files                                                                                                                 |
| [step_1_detectQRS_buildAVG_shape.py](step_1_detectQRS_buildAVG_shape.py)               | Detects QRS and builds averaged QRS for each file. The code refers to detection model in ONNX format which is not provided since it was exclusively licensed to a private company. | 
| [step_2_convert_to_XYZ.py](step_2_convert_to_XYZ.py)                                   | Converts averaged QRS into VCG using Korrs matrix                                                                                                                                  |  
| [step_3_dur_area_features_UPG.py](step_3_dur_area_features_UPG.py)                     | Computes basic features as QRS Area etc.                                                                                                                                           |
| _[step_4-5_basic_analysis.py](step_4-5_basic_analysis.py)_                             | _Basic analysis during development (not essential)_                                                                                                                                |
| _Core methods for **feature windows optimization:**_                                   |                                                                                                                                                                                    |
| **[step_6_build_matrices.py](step_6_build_matrices.py)**                               | **Builds 3D matrices for every feature as separate .npy files**                                                                                                                    |
| **[step_7_analyze_using_CV.py](step_7_analyze_using_CV.py)**                           | **Generates feature window maps as PNG files and detects optimal points in 5-fold CV** (Fig. 3)  ![](figs_maps\CV_SMOOTH_dXMean_1_AUCData_85_cases_SMOOTH_median10.png)            |
| **[step_8_KDE_from_CV.py](step_8_KDE_from_CV.py)**                                     | **Generates aka Kernel-Density-Estimates to find optimal window** (Fig. 3, Fig. 6)  ![](figs_maps\KDE_dYMean.png)                                                                  |
| _Methods to generate **result tables** and aditional images:_                          |                                                                                                                                                                                    |
| [step_9_gen_AVG_VCG_images.py](step_9_gen_AVG_VCG_images.py)                           | Renders average ECG with found optimal windows (Fig. 1, Fig. 5)   ![](figs\COG_Averaged_VCG_RN_f_84_cases.svg)                                                                     |
| [step_10_boxplots.py](step_10_boxplots.py)                                             | Renders boxplots and AUC graphs (Fig. 4, Fig .5)   ![](results\AUC_dYMean.svg)                                                                                                     |
| _[step_11_draw_ECG_images.py](step_11_draw_ECG_images.py)_                             | _Renders standard ECG graphs for each patient (not essential)_                                                                                                                     |
| [step_12_compare_clin_feas_and_computed.py](step_12_compare_clin_feas_and_computed.py) | Generates results comparing performance of computed and clinical features                                                                                                          |

Contact: fplesinger at isibrno.cz