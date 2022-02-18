import yaml
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit the segmentation configs")
    parser.add_argument("--project_name", required=True, type = str, help = "Name of the current project/test")
    parser.add_argument("--segmenter_type", required=True, type = str, help = "Type of the segmentation model: bs | bcd")
    parser.add_argument("--slurm_dir", required=True, type = str, help = "Path to the slurm_job directory")
    parser.add_argument("--train_imgs", default= "seg_em2exm_f0_img_patches.hdf5", type = str, help = "Name of the h5 image file")
    parser.add_argument("--train_labels", default= "seg_em2exm_f0_seg_patches.hdf5", type = str, help = "Name of the h5 gt segmentation file")
    parser.add_argument("--inference_imgs", default= "seg_em2exm_f0_source_patches.hdf5", type = str, help = "Name of the h5 image file for inferencing")
    parser.add_argument("--results_dir", default="test", type = str, help = "Folder name where to save the inferencing results")
    
    args = parser.parse_args()

    if args.segmenter_type.lower() == "bc":
        unet_conf_file = args.project_name+"-UNet-BC.yaml"
    elif args.segmenter_type.lower() == "bcd":
         unet_conf_file = args.project_name+"-UNet-BCD.yaml"
    else:
        raise(AttributeError, "The segmentation type has to be either \"bc\" or \"bcd\"")

    project_folder = os.path.join(args.slurm_dir, args.project_name)

    # retrive the files
    with open(os.path.join(project_folder, args.project_name+"-Base.yaml"), "r") as f:
        base_config = yaml.safe_load(f)


    with open(os.path.join(project_folder, unet_conf_file), "r") as f:
            unet_config = yaml.safe_load(f)

    # update the base config values
    base_config["DATASET"]["IMAGE_NAME"] = os.path.join(project_folder, args.train_imgs)
    base_config["DATASET"]["LABEL_NAME"] = os.path.join(project_folder, args.train_labels)
    base_config["DATASET"]["OUTPUT_PATH"] = os.path.join(project_folder,"segment")
    base_config["INFERENCE"]["IMAGE_NAME"] = os.path.join(project_folder, args.inference_imgs)
    base_config["INFERENCE"]["OUTPUT_PATH"] = os.path.join(project_folder, "segment/"+args.results_dir)

    # set the number of iterations
    #base_config["SOLVER"]["ITERATION_SAVE"] = 500
    #base_config["SOLVER"]["ITERATION_TOTAL"] = 1000

    # update4 the unet config file
    unet_config["INFERENCE"]["OUTPUT_PATH"] = os.path.join(project_folder, "segment/"+args.results_dir)
    unet_config["DATASET"]["OUTPUT_PATH"] = os.path.join(project_folder, "segment")

    # save the changes
    with open(os.path.join(project_folder, args.project_name+"-Base.yaml"), "w") as f:
        yaml.safe_dump(base_config,f)

    with open(os.path.join(project_folder, unet_conf_file), "w") as f:
        yaml.safe_dump(unet_config,f)