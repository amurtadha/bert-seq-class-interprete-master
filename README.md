 
 An implementation to intrepret BERT sequence classification using Captum Library 
 

 

# Data

The data should be in json file with this format {'text':'', 'label':''}. We have processed MR as an example, [download](https://drive.google.com/drive/folders/1D-HnEWps6NsajM3x-DIn560atKFlw2-I?usp=sharing) and place it datasets/MR.

# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# How to use

*  Train a baseline         
*  Run the following code to train a baseline:
```
python train_baseline.py --dataset='MR' 
```

Note that we have already trained a baseline on MR can be downloaded from this [link](https://drive.google.com/drive/folders/1D-HnEWps6NsajM3x-DIn560atKFlw2-I?usp=sharing), simply put in state_dict, or you can use your custom model, no additional configuration is needed.


*  Interprete the baseline prediction         
*  Run the following code:
```
python run_interpretation.py --dataset='MR' 
```
The results will be written to outputs/MR.html
An example of ten sentence are shown in outputs/MR.html




