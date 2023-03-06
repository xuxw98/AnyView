### Prepare ScanNetAV Data for AnyView

1.Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this directory. 



2.Extract point clouds and annotations by running `python batch_load_scannet_data.py`.


3.Follow [3DMV](https://github.com/angeladai/3DMV/) to generate 2D data.

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



4.Running `python generate_anyview_data.py` to generate compressed files (`.npz`) for each scene.
