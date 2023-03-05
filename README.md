This is the pytorch implementation for the paper: *Revisiting Indoor 3D Object Detection: A New Practical Setting and Framework*, which is in submission to RAL.



# Running AnyView

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


# Benchmarking

## Dataset preparation



We follow the 3detr codebase for preprocessing our data.

The instructions for preprocessing ScanNet are [here](https://github.com/xuxw98/AnyView/tree/main/scannet).



## Training



The model can be simply trained by running `main.py`.

```
python main.py --dataset_name scannetAV --model_name 3detr_sepview --checkpoint_dir <path to store outputs> --num_views <number of frames>
```



To reproduce the results in the paper, we provide the arguments in the [`scripts`](scripts/) folder.

A variance of 1% AP25 across different training runs can be expected.



## Testing



Once you have the datasets prepared, you can test pretrained models as



```
python main.py --dataset_name scannetAV --model_name 3detr_sepview --test_ckpt <path_to_checkpoint> --test_only 
```



For testing on uniform dataset settings, You can edit the num_views in  [`scannet/datasets/__init__.py`](scannet/datasets/__init__.py#L57) to choose to test on different number of frames. Correspondingly, you need to edit the MAX_FRAMES and PPF in  [`models/model_3detr_sepview.py`](models/model_3detr_sepview.py#L24-L25) , which follows the rule that 5, 10, 15, 20, 30, 40, 50 frames correspond to 200, 200, 100, 100, 70, 50, and 40 PPF.



For testing on continuous dataset settings, You can keep the num_views in  [`scannet/datasets/__init__.py`](scannet/datasets/__init__.py#L57) be 50 and use the argument `--test_anyview x00` to choose to test on different number of frames, which follows the rule that 100, 200, 300, 400, 500, 600, 700 correspond to 5, 10, 15, 20, 30, 40, and 50 frames. Correspondingly, you need to edit the MAX_FRAMES and PPF in  [`models/model_3detr_sepview.py`](models/model_3detr_sepview.py#L24-L25).

