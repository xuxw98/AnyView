### Prepare ScanNetAV Data for AnyView



We follow the procedure in [3detr](https://github.com/facebookresearch/3detr/) and [3DMV](https://github.com/angeladai/3DMV/).



1.Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory. 



2.In this directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`. Add the `--max_num_point 50000` flag if you only use the ScanNet data for the detection task. It will downsample the scenes to less points.



3.In this directory, 2D train images can be processed from the ScanNet dataset using the 2d data preparation script in [prepare_data](prepare_data). It depends on the [sens file reader from ScanNet](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py) and [ScanNet util](https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py), which should be placed in the same directory.

Then generate 2D train data by running:

```
python prepare_2d_data.py --scannet_path scans --output_path 2D --export_label_images
```

2D data is expected to be in the following file structure:

```
scene0000_00/
|--color/
   |--[framenum].jpg
       ⋮
|--depth/
   |--[framenum].png   (16-bit pngs)
       ⋮
|--pose/
   |--[framenum].txt   (4x4 rigid transform as txt file)
       ⋮
|--label/    (if applicable)
   |--[framenum].png   (8-bit pngs)
       ⋮
scene0000_01/
⋮
```



4.In this directory, extract point cloud per frame and gather other information  by running `python generate_anyview_data.py`. Then anyview_2d_data folder holds the final data.