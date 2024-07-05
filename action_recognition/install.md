# Create Environment
```
conda create -n fiboar python=3.8 

conda activate fiboar
```
# Install dependencies for action recognition task

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- matlotlib : `pip install matplotlib`


# DataSet preparation
The dataset could be structured as follows:
```
├── data
    ├── Action_01
        ├── Video_01.mp4
        ├── Video_02.mp4
        ├── …
```
After all the data is prepared, resize and crop the video to person-centric to get rid of background noise. Then, prepare the CSV files for the training, validation, and testing sets as `train.csv`, `val.csv`, and `test.csv`. The format of the CSV file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```



# Training 

We provide configs for training fibottention for action recognition  on Smarthome, NTU and NUCLA datasets  in [action_recognition/configs/](configs/). Please update the paths in the config to match the paths in your machine before using.

For example to train  on Smarthome using 8 GPUs run the following command:

`python action_recognition/tools/run_net.py --cfg configs/SMARTHOME.yaml NUM_GPUS 8`












