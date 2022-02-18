"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import single_html
import wandb

import numpy as np
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = single_html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if opt.cerberus:
            image_dir = webpage.get_image_dir()
            # create data folder if not exist
            if i == 0:
                for name in visuals:
                    folder_name = os.path.join(image_dir, name)
                    try:
                        os.makedirs(folder_name, exist_ok=False)
                    except FileExistsError as e:
                        print(e)
                    try:
                        os.makedirs(folder_name, exist_ok=False)
                    except FileExistsError as e:
                        print(e)

            poses = model.get_current_poses()  # get image results

            # A coordinate frame
            pos_A = poses["pos_A"]
            pos_name_A = "_".join([str(a.cpu().detach().numpy()[0]) for a in pos_A]) + '.npy' 
            print(pos_name_A)
            img_A = [visuals["fake_B"], visuals["seg_syn_mask_A"], visuals["seg_syn_contours_A"], visuals["seg_rec_mask_A"], visuals["seg_rec_contours_A"]]
            names_A = ["fake_B", "seg_syn_mask_A", "seg_syn_contours_A", "seg_rec_mask_A", "seg_rec_contours_A"]

            for i, (name, img) in enumerate(zip(names_A, img_A)):
                print(os.path.join(os.path.join(image_dir, name), pos_name_A))
                image_tensor = img.data
                image_numpy = image_tensor[0].cpu().float().numpy()
                save_path = os.path.join(os.path.join(image_dir, name), pos_name_A)   
                if name == "fake_B":
                    image_numpy = (image_numpy + 1) / 2.0
                    np.save(save_path, image_numpy.astype(np.uint8))
                np.save(save_path, image_numpy)

            # B coordinate frame
            pos_B = poses["pos_B"]
            pos_name_B = "_".join([str(b.cpu().detach().numpy()[0]) for b in pos_B]) + '.npy' 
            img_B = [visuals["fake_A"], visuals["seg_syn_mask_B"], visuals["seg_syn_contours_B"], visuals["seg_rec_mask_B"], visuals["seg_rec_contours_B"]]
            names_B = ["fake_A", "seg_syn_mask_B", "seg_syn_contours_B", "seg_rec_mask_B", "seg_rec_contours_B"]

            for i, (name, img) in enumerate(zip(names_B, img_B)):
                print(os.path.join(os.path.join(image_dir, name), pos_name_A))
                image_tensor = img.data
                image_numpy = image_tensor[0].cpu().float().numpy()
                save_path = os.path.join(os.path.join(image_dir, name), pos_name_B)   
                if name == "fake_A":
                    image_numpy = (image_numpy + 1) / 2.0
                    np.save(save_path, image_numpy.astype(np.uint8))
                np.save(save_path, image_numpy)

        else:
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
            webpage.save()  # save the HTML
