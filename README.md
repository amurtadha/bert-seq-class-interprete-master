 
 An implementation to intrepret BERT sequence classification using Captum Library 
 

 

# Data

The data should be in json file with this format {'text':'', 'label':''}. We have pre-processed MR dataset as an example, [download it](https://drive.google.com/drive/folders/1D-HnEWps6NsajM3x-DIn560atKFlw2-I?usp=sharing) and place it in datasets/MR.

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

<img src="https://github.com/amurtadha/bert-seq-class-interprete-master/tree/main/outputs/MR.jpg" alt="Alt text" title="Optional title">


![Alt text](https://github.com/amurtadha/bert-seq-class-interprete-master/tree/main/outputs/MR.jpg?raw=true "Title")
