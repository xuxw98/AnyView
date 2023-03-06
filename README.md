This is the pytorch implementation for the paper: *Revisiting Indoor 3D Object Detection: A New Practical Setting and Framework*, which is in submission to RAL.


## Demo
To be uploaded.

## Installation

Our code is tested with PyTorch 1.9.0, CUDA 10.2 and Python 3.6. It may work with other versions.



You will need to install `pointnet2` layers by running



```
cd third_party/pointnet2 && python setup.py install
```



You will also need Python dependencies (either `conda install` or `pip install`)



```
matplotlib

opencv-python

plyfile

'trimesh>=2.35.39,<2.35.40'

'networkx>=2.2,<2.3'

scipy

imageio

scikit-image

opencv

numpy
```


## Dataset preparation

The instructions for preprocessing ScanNet are [here](https://github.com/xuxw98/AnyView/tree/main/scannet).


## Training


Run the following command:

```
python main.py --dataset_name scannetAV --model_name 3detr_sepview --checkpoint_dir <path to store outputs> --num_views <number of frames>
```

## Testing

Run the following command:
```
python main.py --dataset_name scannetAV --model_name 3detr_sepview --test_ckpt <path_to_checkpoint> --test_only 
```


