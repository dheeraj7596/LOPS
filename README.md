# ConWea: Contextualized Weak Supervision for Text Classification

- [Model](#model)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Commands](#commands)
	- [Requirements](#requirements)
- [Citation](#citation)

## Model


## Training

### Required inputs
Each Dataset should contain following files:
- **DataFrame pickle file**
  - Example: ```data/nyt-fine/df.pkl```
    - This dataset should contain two columns named ```text```, ```label```
    - ```text``` contains text and ```label``` contains its corresponding label.
    - Must be named as ```df.pkl```
- **Seed Words Json file**
  - Example: ```data/nyt-fine/seedwords.json```
    - This json file contains seed words list for each label.
    - Must be named as ```seedwords.json```

### Commands

The ```main.py``` requires five arguments: 
- ```dataset_path```, which is a path to dataset containing  required DataFrame and seedwords
- ```dataset``` is the name of the dataset. Ex: ```nyt-fine```
- ```clf``` is the text classifier, currently only accepts ```gpt2``` for GPT-2, ```bert``` for BERT, ```roberta``` for RoBERTa, and ```xlnet``` for XL-Net.
- ```gpu_id``` refers to the id of the gpu. If not mentioned, the process runs on cpu.
- ```lops``` is the flag if set to 1 enables LOPS and if set to 0 disables LOPS

Example command to run:
```shell script
$ python main.py --dataset_path dataset_path --dataset dataset --clf bert --gpu_id 3 --lops 1
```

### Requirements

This project is based on ```python==3.7```. The dependencies are as follow:
```
keras-contrib==2.0.8
scikit-learn==0.21.3
torch==1.9.1
argparse
transformers==4.3.3
nltk
scipy=1.3.1
numpy==1.17.2
```

## Citation

```
```