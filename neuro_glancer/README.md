# Neuro Glancer

Visualize the data volumes in the Neuro-Glancer.

## Data Volumes

* `em_img_merged_dorsal_crop_volume_small`
`dorsal_crop_volume_small` merged to a single 3D volume

* `em_seg_merged_dorsal_crop_volume_small`
The to `dorsal_crop_volume_small` corresponding segmentation masks merged to a single 3D volume

* `exm_img_dorsal_crop_volume_small`
The to the exm domain transferred `dorsal_crop_volume_small` volume merged to a single 3D volume

* `em_img_merged_dorsal_crop_volume_mid`
`dorsal_crop_volume_mid` merged to a single 3D volume

* `em_seg_merged_dorsal_crop_volume_mid`
The to `dorsal_crop_volume_mid` corresponding segmentation masks merged to a single 3D volume

* `exm_img_merged_dorsal_ccgan__dorsal_crop_dense__vanilla__20E_15DE`
The `dorsal_crop_volume_small` volume transferred to the ExM domain using a vanilla CycleGAN trained on the `dorsal_crop_dense` dataset for 20 normal epochs and 15 additional epochs during which the LR linearly converts to zero

* `exm_img_merged_dorsal_ccgan__dorsal_crop_dense__vanilla__25E_20DE`
The `dorsal_crop_volume_small` volume transferred to the ExM domain using a vanilla CycleGAN trained on the `dorsal_crop_dense` dataset for 20 normal epochs and 15 additional epochs during which the LR linearly converts to zero

## Use
```bash
salloc -p cox -t 0-01:00 -c 8 --gres gpu:8 --mem 100000
source activate py3_torch
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
pip install --upgrade pip
pip install neuroglancer imageio h5py cloud-volume

echo "Commands to start the four most important volume representations:"
if [ $volume == "x" ]
then
python3 -i neuro_glancer.py --port 9011 --imgs "/n/pfister_lab2/Lab/leander/em2exm/neuro_glancer/volumes/em_img_merged_dorsal_crop_volume_small.h5" --segs "/n/pfister_lab2/Lab/leander/em2exm/neuro_glancer/volumes/em_seg_merged_dorsal_crop_volume_small.h5"
elif [ $volume == "y" ]
python3 -i neuro_glancer.py --port 9012 --imgs "/n/pfister_lab2/Lab/leander/em2exm/neuro_glancer/volumes/exm_img_merged_dorsal_ccgan__dorsal_crop_dense__vanilla__20E_15DE.h5" --segs "/n/pfister_lab2/Lab/leander/em2exm/neuro_glancer/volumes/em_seg_merged_dorsal_crop_volume_small.h5"
elif [ $volume == "z" ]
python3 -i neuro_glancer.py --port 9013 --imgs '/n/pfister_lab2/Lab/leander/em2exm/pytorch_connectomics/datasets/inference/merged_256_not_filtered_dorsal_b.h5' --segs '/n/pfister_lab2/Lab/leander/em2exm/slurm_jobs/ccgan_dorsal_crop_volume_small__train__segmodel/segment/dorsal_b/bcd_watershed_result.h5'
elif [ $volume == "a" ]
python3 -i neuro_glancer.py --port 9014 --imgs '/n/pfister_lab2/Lab/leander/em2exm/pytorch_connectomics/datasets/inference/merged_256_not_filtered_dorsal_b.h5' --segs '/n/pfister_lab2/Lab/leander/em2exm/slurm_jobs/seg___volume_small__from__dens_van_20E_15DE__train_segmodel/segment/dorsal_b/bcd_watershed_result.h5'
if
```
