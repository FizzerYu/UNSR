# UNSR

A Single Image Super Resolution Framework

- Easy use, easy train
- End to End notes
- Liunx system support

## To-do
- [  ] visualization support


## Usage
### 1. dataset preparation
#### 1. Train Dataset
- [DIV2K_dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 

#### 2. Benchmark Datasets
- [LapSRN](http://vllab.ucmerced.edu/wlai24/LapSRN/)

    - 2.1 Download and extract the ```SR_testing_datasets.zip```.
    - 2.2 Choose where you want to save the ```test dataset``` and put ```Prepare_TestData_HR_LR.m``` in this folder.
    - 2.3 Specific the path_original in ```Prepare_TestData_HR_LR.m```
    - 2.4 Start the test dataset process with ```matlab -nodesktop -nosplash -r Prepare_TestData_HR_LR```

- final test dataset structure

```
|-- SR_testing_datasets.zip
|-- path_original  # <-- this is path_original
    |-- BSD100
    |-- Set5
    |-- Set14
    |-- Urban100
```

```
|-- your_test_data_file   #<-- will be used in model test
    |-- Prepare_TestData_HR_LR.m
    |-- HR
        |-- BSD100
        |-- Set5
        |-- Set14
        |-- Urban100
    |-- LR   
        |-- LRBI
            |-- BSD100
            |-- Set5
            |-- Set14
            |-- Urban100
```



#### 3. Note
- You should have a matlab in your computer
- It would be a good choice to link files rather than put these dataset in your main folder.
- We don't process ```historical``` since it is rarely used as benchmark.

### 2. Design Your Model
1. Design your model in ```train/model```
2. Your model will be import from ```make_model``` function, so you must define this function in your custom model file.
3. If you have some custom setting in your model like the ```block_number```, you can transfer the parameters throuh ```--model_choose```.

```python
# a sample make_model function
def make_model(args, parent=False):
    if args.model_choose == 'RCAN':   # default model
        return mymodel(scale=args.scale[0])
    elif args.model_choose.startswith('custom'): 
         # --model_choose custom_12_12_True 
        custom_args = args.model_choose.split('_')[1:]  # remove the first "custom"
        custom_args = [True if x=='True' for x in custom_args]
        custom_args = [False if x=='Flase' for x in custom_args]
        return mymodel(scale=args.scale[0], block_number1 = int(custom_args[0]), block_number2 = int(custom_args[1]), args3 = custom_args[2] )
```


### 3. Train Your Model
1. cd to the train folder
2. run the following code

> CUDA_VISIBLE_DEVICES=0 mypython main.py --model wsr --save wsr --scale 4 --decay_type cosine --lr 2e-4 --reset --ext bin --chop --save_results --print_model --patch_size 192 2>&1 | tee ./../experiment/wsr-`date +%Y-%m-%d-%H-%M-%S`.txt

### 4. Test Your Model
1. cd to the test folder
2. Copy the model folder from train folder(it would be convenient if you use link files)
3. run the following code
> CUDA_VISIBLE_DEVICES=0 python main.py --scale 4 --model wsr --pre_train ../experiment/wsr/model/model_best.pt --test_only --save_results --chop --save ../experiment/wsr/ --dir_data your_test_data_file/ --ext img --data_test B100

4. since there are many test dataset, it would be easier to write a ```.sh``` file. (Example is in ```run_test.sh```, run it with ```source run_test.sh```)

### 5. Evaluate Your Model




## Package Usage
torch >= 1.6.0
torch-summary


## Reference
This code was original from [RCAN](https://github.com/yulunzhang/RCAN) with modification

