from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import random
import torch
import os
import tifffile
from data.base_dataset import BaseDataset
import torchio as tio

# save the locations for joining the sub-volumes
import csv

from connectomics.data.utils import *


class CerberusDataset(BaseDataset):
    """
    Dataset class for volumetric image datasets. At training time, subvolumes are randomly sampled from all the large 
    input volumes with (optional) rejection sampling to increase the frequency of foreground regions in a batch. At inference 
    time, subvolumes are yielded in a sliding-window manner with overlap to counter border artifacts. 

    Args:
        volume (list): list of image volumes.
        label (list, optional): list of label volumes. Default: None
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        mode (str): ``'train'`` or ``'test'``. Default: ``'train'``
        target_opt (list): list the model targets generated from segmentation labels.
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        reject_size_thres (int, optional): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_diversity (int, optional): threshold to decide if a sampled volumes contains multiple objects. Default: 0
        reject_p (float, optional): probability of rejecting non-foreground volumes. Default: 0.95

    Note: 
        For relatively small volumes, the total number of possible subvolumes can be smaller than the total number 
        of samples required in training (the product of total iterations and mini-natch size), which raises *StopIteration*. 
        Therefore the dataset length is also decided by the training settings.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Adding dataset-specific options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. 

        Returns:
            the modified parser.
        """
        parser.add_argument('--tif_A_file_name', type=str, required=True, help='name of the tiff file for the data of domain A')
        parser.add_argument('--tif_A_file_name_label', type=str, required=True, help='name of the tiff label file for the data of domain A')
        parser.add_argument('--tif_B_file_name', type=str, required=True, help='name of the tiff file for the data of domain B')

        parser.set_defaults(max_dataset_size=1000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.opt = opt # experiment options
        self.sample_volume_size = (opt.vs_z, opt.vs_x, opt.vs_y) # size for the image subvolumes
        self.sample_label_size = (opt.vs_z, opt.vs_x, opt.vs_y) # size for the label subvolumes
        self.sample_stride = (opt.ss_z,opt.ss_x,opt.ss_y) # stride size

        self.mode = self.opt.phase # mode: train | test
        assert self.mode in ['train', 'test']


        # setting contour dilation to two since contours of thickness one are to thin for the L1 loss 
        if self.opt.bcd:
            self.target_opt = ['0','4-1-1', '6'] # option for the label output: binary mask | binary mask + contours | ...
        else:
            self.target_opt = ['0','4-1-1'] # option for the label output: binary mask | binary mask + contours | ...

        self.erosion_rates = None # erosion rate for the binary masks
        self.dilation_rates = None # dilation rate for the binary masks
        self.iter_num = -1 # total number of training iterations (-1 for inference)

        # setup csv documentation to merge sub-volumes after inferencing
        self.header = ["index", "A_z1", "A_z2", "A_x1", "A_x2", "A_y1", "A_y2", "B_z1", "B_z2", "B_x1", "B_x2", "B_y1", "B_y2"]
        # save the csv file in the experiments checkpoint directory
        self.csv_path = os.path.join(self.opt.checkpoints_dir, "vs_"+"_".join(list(map(str,self.sample_volume_size)))+"_sz_"+"_".join(list(map(str,self.sample_stride)))+".csv")
        # initialize the csv file with the header
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(self.header)

        # get the paths to the dataset;
        self.dir_A = os.path.join(self.opt.dataroot, self.opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_A_label = os.path.join(self.opt.dataroot, 'gt_seg_mask')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(self.opt.dataroot, self.opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        # load the datasets from the tiff files
        # the image data should be of type uint8 
        # labels should be of type uint8, uint16, uint32, or uint64 depending on the number of objects in the data
        with tifffile.TiffFile(os.path.join(self.dir_A, self.opt.tif_A_file_name)) as tif_A, \
                     tifffile.TiffFile(os.path.join(self.dir_A_label, self.opt.tif_A_file_name_label)) as tif_A_label, \
                     tifffile.TiffFile(os.path.join(self.dir_B, self.opt.tif_B_file_name)) as tif_B: 

                # convert to array
                # no need for `key` because all pages are in same series 
                volume_A = tif_A.asarray()
                volume_B = tif_B.asarray()
                label = tif_A_label.asarray()

                # assert data formats
                assert(volume_A.dtype == "uint8"), "The input image volume should be of type uint8"
                assert(volume_B.dtype == "uint8"), "The input image volume should be of type uint8"
                assert(label.dtype in ["uint8" ,"uint16", "uint32", "uint64"]), "The input label volume should be of type uint8 | uint16 | uint32 | uint64"

                # volume dataset can handle lists of different volumes
                self.volume_A = [volume_A] 
                self.volume_B = [volume_B]
                self.label = [label]

        # print out the volume shapes
        print("self.volume_A.shape ", np.asarray(self.volume_A).shape)
        print("self.label.shape ", np.asarray(self.label).shape)
        print("self.volume_B.shape ", np.asarray(self.volume_B).shape)

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.volume_size_A = [np.array(x.shape) for x in self.volume_A]
        self.volume_size_B = [np.array(x.shape) for x in self.volume_B]
        

        # convert the image sub-volume size to a numpy array
        self.sample_volume_size = np.array(
            self.sample_volume_size).astype(int)  

        # convert the label sub-volume size to a numpy array
        # calculate the scaling difference bet
        if self.label is not None:
            self.sample_label_size = np.array(
                self.sample_label_size).astype(int)  # model label size
            self.label_vol_ratio = self.sample_label_size / self.sample_volume_size
           
        # assert the subvolume sizes are not larger then the volume sizes
        self._assert_valid_shape()

        # convert the stride size / stride size to a numpy array
        self.sample_stride = np.array(self.sample_stride).astype(int)
        
        # compute number of samples for each dataset (multi-volume input)
        # based on the possible positions of a sliding window of shape 'sample_volume_size' for a stride of 'sample_stride'
        # similar to the positions of a convolution kernel
        self.sample_size_A = [count_volume(self.volume_size_A[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size_A))]
        self.sample_size_B = [count_volume(self.volume_size_B[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size_B))]

        # total number of possible inputs for each volume
        # sample_size_* is a 3D tuple holding the number of possible postions in the x,y,z direction
        # the number of all possible postions is x*y*z
        self.sample_num_A = np.array([np.prod(x) for x in self.sample_size_A])
        self.sample_num_B = np.array([np.prod(x) for x in self.sample_size_B])
        
        # if we have multiple volumes the number of possible sub-volumes is the sum 
        # of possible sub-volumes of the singel volumes
        self.sample_num_a_A = np.sum(self.sample_num_A)
        self.sample_num_a_B = np.sum(self.sample_num_B)

        # number of possible subvolumes = [0, #vol1, #vol1+#vol2, #vol1+#vol2+#vol1, ...]
        self.sample_num_c_A = np.cumsum([0] + list(self.sample_num_A))
        self.sample_num_c_B = np.cumsum([0] + list(self.sample_num_B))

        # for validation and test -> [y*x, x]
        # used the calculate the postion of a sample and not to run out of the image
        if self.mode in ['test']: 
            self.sample_size_test_A = [
                np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size_A]
            self.sample_size_test_B = [
                np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size_B]

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        self.iter_num_A = max(
            self.iter_num, self.sample_num_a_A) if self.mode == 'train' else self.sample_num_a_A
        self.iter_num_B = max(
            self.iter_num, self.sample_num_a_B) if self.mode == 'train' else self.sample_num_a_B
        print('Total number of samples to be generated for A: ', self.iter_num_A)
        print('Total number of samples to be generated for B: ', self.iter_num_B)

    def __len__(self):
        # total number of possible samples
        return max(self.iter_num_A, self.iter_num_B)

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        vol_size = self.sample_volume_size
        if self.mode == 'train':

            # random sample the sub-volume A
            pos_A, out_volume_A, out_label = self._random_sampling(vol_size, "A")
            
            # normalize the image sub-volume to -1 to 1 and convert to float32 
            out_volume_A = self._normalize(out_volume_A)
            
            # retrive the binary masks and the contour masks from the label
            label = seg_to_targets(out_label, self.target_opt, self.erosion_rates, self.dilation_rates)
            label = np.squeeze(np.asarray(label))

            # random sample the sub-volume A - we only have labels for A
            pos_B, out_volume_B, _  = self._random_sampling(vol_size, "B")
            out_volume_B = self._normalize(out_volume_B)

            # assert normalizations
            assert(np.min(out_volume_A) >= -1.0 and np.max(out_volume_A) <= 1.0), "The image sub-volumes have to be normalized between -1.0 and 1.0"
            assert(np.min(out_volume_B) >= -1.0 and np.max(out_volume_B) <= 1.0), "The image sub-volumes have to be normalized between -1.0 and 1.0"

            # the binary mask and contour mask should normalized to a range of 0 to 1 while the distance map should be between -1 to 1 
            if len(self.target_opt) == 2:
                assert(np.min(label) >= 0.0 and np.max(label) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
            elif len(self.target_opt) == 3:
                assert(np.min(label[:2,:,:,:]) >= 0.0 and np.max(label[:2,:,:,:]) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
                assert(np.min(label[2:3,:,:,:]) >= -1.0 and np.max(label[2:3,:,:,:]) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
                  
            # assert data formats
            assert(out_volume_A.dtype == "float32"), "The input to a PyTorch network should always be of type float32"
            assert(out_volume_B.dtype  == "float32"), "The input to a PyTorch network should always be of type float32"
            assert(label.dtype == "float32"), "The input to a PyTorch network should always be of type float32"

            # convert the image sub-volume to tensor and add a batch dimension
            out_volume_A = torch.Tensor(out_volume_A).reshape((1,out_volume_A.shape[0],out_volume_A.shape[1],out_volume_A.shape[2]))
            out_volume_B = torch.Tensor(out_volume_B).reshape((1,out_volume_B.shape[0],out_volume_B.shape[1],out_volume_B.shape[2]))
            label = torch.Tensor(label)

            return {'A': out_volume_A, 'B': out_volume_B, 'Label': label, 'A_pos': pos_A, 'B_pos': pos_B, 'A_paths': "z_"+str(pos_A[1])+"-"+str(pos_A[1]+vol_size[0]) +"_x_"+ str(pos_A[2])+"-"+str(pos_A[2]+vol_size[1]) +"_y_"+ str(pos_A[3])+"-"+str(pos_A[3]+vol_size[2]), 'B_paths': "z_"+str(pos_B[1])+"-"+str(pos_B[1]+vol_size[0]) +"_x_"+ str(pos_B[2])+"-"+str(pos_B[2]+vol_size[1]) +"_y_"+ str(pos_B[3])+"-"+str(pos_B[3]+vol_size[2])}

        elif self.mode == 'test':
            pos_A = self._get_pos_test(index, "A")
            pos_B = self._get_pos_test(index, "B")

            # sub-volume A
            out_volume_A = crop_volume(
                self.volume_A[pos_A[0]], vol_size, pos_A[1:])
            
            #normalize sub-volume A
            out_volume_A = self._normalize(out_volume_A)

            # sub-volume B
            out_volume_B = crop_volume(
                self.volume_B[pos_B[0]], vol_size, pos_B[1:])

            # normalize sub-volume B
            out_volume_B = self._normalize(out_volume_B)

            # label
            out_label = crop_volume(self.label[pos_A[0]], vol_size, pos_A[1:])

            # retrive the binary masks and the contour masks from the label
            label = seg_to_targets(out_label, self.target_opt, self.erosion_rates, self.dilation_rates)
            label = np.squeeze(np.asarray(label))

            # convert the image sub-volume to tensor and add a batch dimension
            assert(np.min(out_volume_A) >= -1.0 and np.max(out_volume_A) <= 1.0), "The image sub-volumes have to be normalized between -1.0 and 1.0"
            assert(np.min(out_volume_B) >= -1.0 and np.max(out_volume_B) <= 1.0), f"The image sub-volumes have to be normalized {np.min(label)} and{np.max(label)}"

            # the binary mask and contour mask should normalized to a range of 0 to 1 while the distance map should be between -1 to 1 
            if len(self.target_opt) == 2:
                assert(np.min(label) >= 0.0 and np.max(label) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
            elif len(self.target_opt) == 3:
                assert(np.min(label[:2,:,:,:]) >= 0.0 and np.max(label[:2,:,:,:]) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
                assert(np.min(label[2:3,:,:,:]) >= -1.0 and np.max(label[2:3,:,:,:]) <= 1.0), f"The binary and contour mask have to be normalized between 0.0 and 1.0 and not {np.min(label)} and{np.max(label)}"
                
            # assert data formats
            assert(out_volume_A.dtype == "float32"), "The input to a PyTorch network should always be of type float32"
            assert(label.dtype == "float32"), "The input to a PyTorch network should always be of type float32"
            assert(out_volume_B.dtype  == "float32"), "The input to a PyTorch network should always be of type float32"

            # add batch dimension and convert to tensor
            out_volume_A = torch.Tensor(out_volume_A).reshape((1,out_volume_A.shape[0],out_volume_A.shape[1],out_volume_A.shape[2]))
            out_volume_B = torch.Tensor(out_volume_B).reshape((1,out_volume_B.shape[0],out_volume_B.shape[1],out_volume_B.shape[2]))
            label = torch.Tensor(label)

            # write line to existing csv file
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([str(index), str(pos_A[1]), str(pos_A[1]+vol_size[0]), str(pos_A[2]), str(pos_A[2]+vol_size[1]), str(pos_A[3]), str(pos_A[3]+vol_size[2]),str(pos_B[1]),str(pos_B[1]+vol_size[0]), str(pos_B[2]),str(pos_B[2]+vol_size[1]), str(pos_B[3]),str(pos_B[3]+vol_size[2])])

            return {'A': out_volume_A, 'B':  out_volume_B, 'Label': label, 'A_pos': pos_A, 'B_pos': pos_B, 'A_paths': "z_"+str(pos_A[1])+"-"+str(pos_A[1]+vol_size[0]) +"_x_"+ str(pos_A[2])+"-"+str(pos_A[2]+vol_size[1]) +"_y_"+ str(pos_A[3])+"-"+str(pos_A[3]+vol_size[2]), 'B_paths': "z_"+str(pos_B[1])+"-"+str(pos_B[1]+vol_size[0]) +"_x_"+ str(pos_B[2])+"-"+str(pos_B[2]+vol_size[1]) +"_y_"+ str(pos_B[3])+"-"+str(pos_B[3]+vol_size[2])}

    #######################################################
    # Position Calculator
    #######################################################

    def _index_to_dataset(self, index, AorB):
        """retrive the index of the dataset -> in case only a single volume was provided
            the output will alsways be equal to zero
        """
        if AorB == 'A':
            return np.argmax(index < self.sample_num_c_A) - 1  # which dataset
        elif AorB == 'B':
            return np.argmax(index < self.sample_num_c_B) - 1  # which dataset

    def _index_to_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = int(np.floor(pz_r/sz[1]))
        pos[2] = pz_r % sz[1]
        return pos

    def _get_pos_test(self, index, AorB):
        "Get the position of a sample based on the current index"
        pos = [0, 0, 0, 0]
        did = self._index_to_dataset(index, AorB)
        # set the dataset index -> allways zero in case of single volume
        pos[0] = did

        if AorB == 'A':
            index2 = index - self.sample_num_c_A[did]
            pos[1:] = self._index_to_location(index2, self.sample_size_test_A[did])
            # if out-of-bound, tuck in
            for i in range(1, 4):
                if pos[i] != self.sample_size_A[pos[0]][i-1]-1:
                    pos[i] = int(pos[i] * self.sample_stride[i-1])
                else:
                    pos[i] = int(self.volume_size_A[pos[0]][i-1] -
                                self.sample_volume_size[i-1])
        elif AorB == 'B':
            index2 = index - self.sample_num_c_B[did]
            pos[1:] = self._index_to_location(index2, self.sample_size_test_B[did])
            # if out-of-bound, tuck in
            for i in range(1, 4):
                if pos[i] != self.sample_size_B[pos[0]][i-1]-1:
                    pos[i] = int(pos[i] * self.sample_stride[i-1])
                else:
                    pos[i] = int(self.volume_size_B[pos[0]][i-1] -
                                self.sample_volume_size[i-1])

        return pos

    def _get_pos_train(self, vol_size, AorB):
        """Retrives the position of a random sample"""
        # random: multithread
        # np.random: same seed
        pos = [0, 0, 0, 0]
        # pick a dataset
        if AorB == 'A':
            did = self._index_to_dataset(random.randint(0, self.sample_num_a_A-1), AorB)
            pos[0] = did
            # pick a position
            tmp_size = count_volume(
                self.volume_size_A[did], vol_size, self.sample_stride)
            tmp_pos = [random.randint(0, tmp_size[x]-1) * self.sample_stride[x]
                    for x in range(len(tmp_size))]
        elif AorB == 'B':
            did = self._index_to_dataset(random.randint(0, self.sample_num_a_B-1), AorB)
            pos[0] = did
            # pick a position
            tmp_size = count_volume(
                self.volume_size_B[did], vol_size, self.sample_stride)
            tmp_pos = [random.randint(0, tmp_size[x]-1) * self.sample_stride[x]
                    for x in range(len(tmp_size))]
        pos[1:] = tmp_pos
        return pos

    #######################################################
    # Volume Sampler
    #######################################################
    def _random_sampling(self, vol_size, AorB):
        """Randomly sample a subvolume from all the volumes. 
        """
        pos = self._get_pos_train(vol_size, AorB) # retrive position of random sample
        return self._crop_with_pos(pos, vol_size, AorB) # crop's out the sample based on the given position

    def _crop_with_pos(self, pos, vol_size, AorB):
        "Cropes out the sub-volume at position 'pos' that expends by size 'vol_size'"
        out_label = None
        if AorB == 'A':
            out_volume = crop_volume(
                self.volume_A[pos[0]], vol_size, pos[1:])
                    # position in the label and valid mask
            out_label = None
            if self.label is not None:
                pos_l = np.round(pos[1:]*self.label_vol_ratio)
                out_label = crop_volume(
                    self.label[pos[0]], self.sample_label_size, pos_l)
                # For warping: cv2.remap requires input to be float32.
                # Make labels index smaller. Otherwise uint32 and float32 are not
                # the same for some values.
                out_label = self._relabel(out_label.copy()).astype(np.float32)
        elif AorB == 'B':
             out_volume = crop_volume(
                self.volume_B[pos[0]], vol_size, pos[1:])
        return pos, out_volume, out_label
    #######################################################
    # Utils
    #######################################################
    def _assert_valid_shape(self):
        assert all(
            [(self.sample_volume_size <= x).all()
             for x in self.volume_size_A]
        ), "Input size should be smaller than volume size."

        assert all(
            [(self.sample_volume_size <= x).all()
             for x in self.volume_size_B]
        ), "Input size should be smaller than volume size."

        if self.label is not None:
            assert all(
                [(self.sample_label_size <= x).all()
                 for x in self.volume_size_A]
            ), "Label size should be smaller than volume size."

    
    def _relabel(self, seg, do_type=False):
        # get the unique labels
        uid = np.unique(seg)
        # ignore all-background samples
        if len(uid) == 1 and uid[0] == 0:
            return seg

        uid = uid[uid > 0]
        mid = int(uid.max()) + 1  # get the maximum label for the segment

        # create an array from original segment id to reduced id
        m_type = seg.dtype
        if do_type:
            m_type = getSegType(mid)
        mapping = np.zeros(mid, dtype=m_type)
        mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
        return mapping[seg.astype(np.int64)]

    def _normalize(self, vol):
        # normalize intput to -1 to 1 and convert to float32 
        return ((vol / 255.0) * 2.0 - 1.0).astype(np.float32)