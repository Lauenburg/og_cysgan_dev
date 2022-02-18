from connectomics.data.augmentation import Compose
from connectomics.data.utils import *
from connectomics.data.dataset.dataset_volume import VolumeDataset 
import torch
from torch.utils.data import Dataset
from glob import glob
import argparse
import time
import os
import h5py

class TestBatch:
    def __init__(self, batch):
        pos, out_input = zip(*batch)
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.out_input = self.out_input.pin_memory()
        return self

class CustomImageDataset(Dataset):
    def __init__(self, output):
        self.output = output
    def __len__(self):
        return len(output)

    def __getitem__(self, idx):
        return self.output[idx][0], self.output[idx][1]

def collate_fn_test(batch):
    return TestBatch(batch)

def merge(dataset, 
         volume_size, # size of the merged volume 
         output_dir, # path where to save the merged output volume
         test_filename, # name of the merged output volume file
         stride=(8, 32, 32), # sampling stride for inference
         output_size = [17, 65, 65], # size of the sub-volumes
         output_scale = [1.0, 1.0, 1.0], # sacale of the output merged volume
         out_planes = 1): # number of out put channels
    r"""Inference function of the trainer class.
    """

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_test,
        sampler=None, num_workers=4, pin_memory=True)

                
    output_scale = output_scale
    spatial_size = list(np.ceil(
        np.array(output_size) *
        np.array(output_scale)).astype(int))
    channel_size = out_planes

    sz = tuple([channel_size] + spatial_size)
    ww = build_blending_matrix(spatial_size, 'gaussian')

    #output_size = [tuple(np.ceil(np.array(x) * np.array(output_scale)).astype(int))
    #               for x in dataloader._dataset.volume_size]
    output_size = [tuple(np.ceil(np.array(volume_size[0]) * np.array(output_scale)).astype(int))]
    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(channel_size)]) for x in output_size]
    weight = [np.zeros(x, dtype=np.float32) for x in output_size]
    print("Total number of batches: ", len(dataloader))

    start = time.perf_counter()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            print('progress: %d/%d batches, total time %.2fs' %
                  (i+1, len(dataloader), time.perf_counter()-start))

            pos, volume = sample.pos, sample.out_input
            output = np.asarray(volume)

            for idx in range(output.shape[0]):
                st = pos[idx]
                st = (np.array(st) *
                      np.array([1]+output_scale)).astype(int).tolist()
                out_block = output[idx]
                if result[st[0]].ndim - out_block.ndim == 1:  # 2d model
                    out_block = out_block[:, np.newaxis, :]
                result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                              st[3]:st[3]+sz[3]] += out_block * ww[np.newaxis, :]
                weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                              st[3]:st[3]+sz[3]] += ww

    end = time.perf_counter()
    print("Prediction time: %.2fs" % (end-start))

    for vol_id in range(len(result)):
        if result[vol_id].ndim > weight[vol_id].ndim:
            weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
        result[vol_id] /= weight[vol_id]  # in-place to save memory
        result[vol_id] *= 255
        result[vol_id] = result[vol_id].astype(np.uint8)


    if output_dir is None:
        return result
    else:
        print('Final prediction shapes are:')
        for k in range(len(result)):
            print(result[k].shape)
        writeh5(os.path.join(output_dir, test_filename), result,
                ['vol%d' % (x) for x in range(len(result))])
        print('Prediction saved as: ', test_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit the segmentation configs")
    parser.add_argument('--output_dir', required=True, type = str, help = "Path to the root dir of the inferenced subvolume")
    parser.add_argument('--save_vol_A', type=str, default="0000000", help= "Volumes are saved if a 1 stands at there index. Order of volumes: fake_B, seg_syn_mask_A, seg_syn_contours_A, seg_rec_mask_A, seg_rec_contours_A, seg_syn_distance_A, seg_rec_distance_A" )
    parser.add_argument('--save_vol_B', type=str, default="0000000", help= "Volumes are saved if a 1 stands at there index. Order of volumes: fake_A, seg_syn_mask_B, seg_syn_contours_B, seg_rec_mask_B, seg_rec_contours_B, seg_syn_distance_B, seg_rec_distance_B" )

    parser.add_argument('--mvs_A_z', type=int, default=33, help='z dimension of the orginal merged A volume')
    parser.add_argument('--mvs_A_x', type=int, default=65, help='x dimension of the orginal merged A volume')
    parser.add_argument('--mvs_A_y', type=int, default=65, help='y dimension of the orginal merged A volume')

    parser.add_argument('--mvs_B_z', type=int, default=33, help='z dimension of the orginal merged B volume')
    parser.add_argument('--mvs_B_x', type=int, default=65, help='x dimension of the orginal merged B volume')
    parser.add_argument('--mvs_B_y', type=int, default=65, help='y dimension of the orginal merged B volume')

    parser.add_argument('--vs_z', type=int, default=33, help='when using volume_dataset; size of the z dimension for a subvolume: 9 | 17 | 33 | 65')
    parser.add_argument('--vs_x', type=int, default=65, help='when using volume_dataset; size of the x dimension for a subvolume: 33 | 65 | 129 | 257')
    parser.add_argument('--vs_y', type=int, default=65, help='when using volume_dataset; size of the y dimension for a subvolume: 33 | 65 | 129 | 257')

    parser.add_argument('--ss_z', type=int, default=16, help='when using volume_dataset; stride size for dim z: 4 | 8 | 16 | 32')
    parser.add_argument('--ss_x', type=int, default=32, help='when using volume_dataset; stride size for dim x: 16 | 32 | 64 | 128')
    parser.add_argument('--ss_y', type=int, default=32, help='when using volume_dataset; stride size for dim y: 16 | 32 | 64 | 128')

    args = parser.parse_args()

    merged_volume_shape_A = [[args.mvs_A_z, args.mvs_A_x, args.mvs_A_y]]
    merged_volume_shape_B = [[args.mvs_B_z, args.mvs_B_x, args.mvs_B_y]]

    sub_volume_shape = [args.vs_z, args.vs_x, args.vs_y]
    stride = [args.ss_z, args.ss_x, args.ss_y]

    # determine which sub-volumes to join for A
    save_vols_A = [True if s == "1" else False for s in str(args.save_vol_A)]
    names_A = ["fake_B", "seg_syn_mask_A", "seg_syn_contours_A", "seg_rec_mask_A", "seg_rec_contours_A", "seg_syn_distance_A", "seg_rec_distance_A"]
    names_A = [b for a, b in zip(save_vols_A, names_A) if a]

    # determine which sub-volumes to join for B
    save_vols_B = [True if s == "1" else False for s in str(args.save_vol_B)]
    names_B = ["fake_A", "seg_syn_mask_B", "seg_syn_contours_B", "seg_rec_mask_B", "seg_rec_contours_B", "seg_syn_distance_B", "seg_rec_distance_B"]
    names_B = [b for a, b in zip(save_vols_B, names_B) if a]

    # merger all indicated A subvolumes
    for name in names_A:
        print(f"Merging the {name} sub-volumes")
        load_path = os.path.join(args.output_dir, name)
        # retrive the data
        out_vols = glob(os.path.join(load_path,"*"))
        output = []
        for vol_name in out_vols:
            pos = np.asarray([ int(p) for p in vol_name.split("/")[-1].split(".")[0].split("_")])
            output.append([pos,np.load(vol_name)])
        # merge the data
        merge(output, merged_volume_shape_A, args.output_dir, name+".h5", stride, sub_volume_shape)

    # merge all indicated B subvolumes
    for name in names_B:
        print(f"Merging the {name} sub-volumes")
        load_path = os.path.join(args.output_dir, name)
        # retrive the data
        out_vols = glob(os.path.join(load_path,"*"))
        output = []
        for vol_name in out_vols:
            pos = np.asarray([ int(p) for p in vol_name.split("/")[-1].split(".")[0].split("_")])
            output.append([pos,np.load(vol_name)])

        merge(output, merged_volume_shape_B, args.output_dir, name+".h5", stride, sub_volume_shape)


    