# Attention to fires: multi-channel deep-learning models forwildfire severity prediction

This repository provides the code of the paper "Attention to fires: multi-channel deep-learning models forwildfire severity prediction" (under review).

**Autors:** Monaco, S.; Greco, S.; Farasin, A.; Colomba, L.; Apiletti, D.; Garza, P.; Cerquitelli, T.; Baralis, E.

## Abstract
Wildfires are one of the natural hazards that the European Union is actively monitoring through the Copernicus EMS Earth observation program and continuously releases public information related to such catastrophic events. Such occurrences are the cause of both short and long terms damages. Thus, to limit their impact and plan the restoration process, a rapid intervention by authorities is needed, which can be enhanced by the use of satellite imagery and automatic burned area delineation methodologies, accelerating the response and the decision-making processes. In this context, we analyze the burned area severity estimation problem by exploiting a state-of-the-art deep-learning framework. Experimental results compare different model architectures and loss functions on a very large real-world Sentinel2 satellite dataset. Furthermore, a novel multi-channel attention-based analysis is presented to uncover the prediction behaviour and provide model interpretability. A perturbation mechanism is applied to an attention-based DS-UNet to evaluate the contribution of different domain-driven groups of channels to the severity estimation problem.

## Architecture
![Double step Framework](http://dbdmg.polito.it/dbdmg_web/wp-content/uploads/2021/10/dsf.png)

The Double-Step Framework splits the task between a first wildfire binary detection and a consequent severity prediction. Such a framework allows complete customization of both training loss functions and backbone deep-learning architectures.
The main building blocks of the DSF can be summarized in the following:
 - Binary-class backbone
 - Binary loss
 - Binary threshold
 - Regression backbone
 - Regression loss
## Dataset
The proposed dataset consists of 73 Areas of Interest (AoI) gathered from different European Regions by Copernicus EMS, which provides severity prediction and wildfire delineation. For each of them, the service provides the four geographical coordinates which define the AoI and the corresponding reference date. Such information has been used to retrieve and select the most suitable satellite acquisitions. Accepted regions are selected if the following constraints are satisfied: (i) the satellite acquisition must stay within the accepted range of the reference date, (ii) the data acquisition must be available for at least the 90\% of the desired AoI, and (iii) cloud coverage must not exceed 10\% of the AoI. Furthermore, an extra analysis of coherence between the acquisition and the delineation map has been performed by calculating the delta Normalized Burnt Ratio (dNBR) between the pre-fire and post-fire images.

Each AoI, identified by a conventional product name, is associated with pre and post fire images collected form Sentinel-2 acquisitions. In this work we only focus on post-fire images. Then, each AoI is present with a terrain areas represented with a matix of size W×H×D, where W and H are respectively width and height (with dimensions up to 5000×5000), and D = 12 represents the 12 channels acquired by satellite sensors.
Finally, each sample presents the pixel-wise ground-truth grading map with 5 severity levels, corresponding to the damage intensity caused by the wildfire, ranging from 0 (no damage) to 4 (completely destroyed).

For reproducibility purpose, we provide localization points of the AoI and the date of the acquisition we considered for the analysis in the file ```dset.csv```. All satellite acquisition are downloaded on https://www.sentinel-hub.com/, while the grading maps are available on https://emergency.copernicus.eu/.

The full dataset is available upon request.
## Usage
The train-test pipeline can be performed by running ```python run_double --model_name=<MOD> --losses=<LOSS>```, Where <MOD> can be any of 'unet', 'nestedunet', 'segnet', 'attentionunet' and <LOSS> can be any of 'bcemse', 'dicemse', 'bdmse', 'bsmse', 'siousiou', 'sioumse', 'bcef1mse'. Further information on the possible flags can be obtaining running ```python run_double --help```. 
  
The code automatically log the progress to [Weights & Biases](https://wandb.ai/). An account is needed for this procedure. If you don't want this, simply add the ```--unlog=True``` flag.
  
Folder structure, having at least the Sentinel-2 post-fire acquisition and the gruond truth grading map, should have the following structure:

```
rescue
├── <The scripts from this repo>
data
│   ├── sentinel-hub
|   │   ├── EMSR214_05LELAVANDOU_02GRADING_MAP_v1_vector
|   |   │   ├── sentinel2_2017-07-29.tiff
|   │   |   └── EMSR214_05LELAVANDOU_02GRADING_MAP_v1_vector_mask.tiff
|   │   ├── EMSR207_04AVELAR_02GRADING_MAP_v2_vector
|   |   │   ├── sentinel2_2017-07-04.tiff
|   │   |   └── EMSR207_04AVELAR_02GRADING_MAP_v2_vector_mask.tiff
|   │   ├── EMSR207_08CERNACHEDOBONJARDIM_02GRADING_MAP_v2_vector
|   |   │   ├── sentinel2_2017-07-04.tiff
|   │   |   └── EMSR207_08CERNACHEDOBONJARDIM_02GRADING_MAP_v2_vector_mask.tiff
|   │   ├── EMSR209_01MOGUER_02GRADING_MAP_v2_vector
  ...
```


## Citation
The paper is under review. Citation reference will be provided here.

## Contacts

Please contact [simone.monaco@polito.it](mailto:simone.monaco@polito.it) for any further questions.
