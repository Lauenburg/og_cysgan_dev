import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.exceptions import ConfigException
import numpy as np


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

            # cerberus configs
            parser.add_argument('--cerberus_contour_weight', type=float, default=1.0, help='weight factor for the segmentation loss')
            parser.add_argument('--cerberus_mask_weight', type=float, default=1.0, help='weight factor for the segmentation loss')
            parser.add_argument('--cerberus_distance_weight', type=float, default=1.0, help='weight factor for the segmentation loss')
            parser.add_argument('--cerberus_D_weight_syn', type=float, default=1.0, help='weight factor for the adverserial loss, synthesized')
            parser.add_argument('--cerberus_D_weight_rec', type=float, default=1.0, help='weight factor for the adverserial loss, reconstructed')
            parser.add_argument('--lambda_B_BCD', type=float, default=2.0, help='weight for (cycle) loss  A_C ≈ G_A(B_C) with (B,B_M,B_C)-> (A,A_M,A_C) -> (B,B_M,B_C )')

        # configure then CycleGAN mode - has to be accessable during training and testing
        parser.add_argument('--cerberus',  action='store_true', help='adds loss based on difference between segmentation and GT mask')
        parser.add_argument('--detach_advers',  action='store_true', help='if set to true creates a version of B_{B,r}, B_{C,r}, B_{D,r} that is detached from G_B and uses them for the pseudo cycle consistency')
        parser.add_argument('--detach_pcycle',  action='store_true', help='if set to true creates a version of B_{B,r}, B_{C,r}, B_{D,r} that is detached from G_B and uses them for the additional adverserial loss')
        parser.add_argument('--detach_seg',  action='store_true', help='if set to true creates a version of A_{B,r}, A_{C,r}, A_{D,r} that is detached from G_A and uses them for the segmentation loss')
        parser.add_argument('--reduce_visuals',  action='store_true', help='only depict real_A, fake_B, real_B, fake_A, seg_syn_mask_A, seg_syn_mask_B, seg_syn_contours_A, seg_syn_contours_B, cerberus_label_mask, cerberus_label_contours')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class. 

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.cerberus:
            if self.opt.detach_seg:
                self.loss_names = ['D_A', 'G_A', 'cycle_A','idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'seg_syn_logits_mask', 'seg_syn_logits_contour', 'seg_rec_logits_mask_detached', 'seg_rec_logits_contour_detached']
                # add the weights for the distance map pseudo cycle consistency if bcd is True
                if self.opt.bcd:
                    self.loss_names = self.loss_names +  ["seg_syn_logits_distance", "seg_rec_logits_distance_detached"]
            else:
                self.loss_names = ['D_A', 'G_A', 'cycle_A','idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'seg_syn_logits_mask', 'seg_syn_logits_contour', 'seg_rec_logits_mask', 'seg_rec_logits_contour']
                # add the weights for the distance map pseudo cycle consistency if bcd is True
                if self.opt.bcd:
                    self.loss_names = self.loss_names +  ["seg_syn_logits_distance", "seg_rec_logits_distance"]            
            # only add the following losses in case their accociated weight is greater zero
            if self.isTrain:
                if self.opt.cerberus_D_weight_syn > 0:
                    self.loss_names = self.loss_names + ['G_B_BCD_syn']
                if self.opt.cerberus_D_weight_rec > 0:
                    self.loss_names = self.loss_names + ['G_B_BCD_rec']
                if self.opt.cerberus_D_weight_rec > 0.0 or self.opt.detach_advers:
                    self.loss_names = self.loss_names + ['D_B_BCD']
                if self.opt.lambda_B_BCD > 0.0:
                    self.loss_names = self.loss_names + ['cycle_B_BCD']
        else:
            # default losses
            self.loss_names = ['D_A', 'G_A', 'cycle_A','idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if not self.opt.reduce_visuals:
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
        else:
            visual_names_A = ['real_A', 'fake_B']
            visual_names_B = ['real_B', 'fake_A']
        
        if not self.opt.reduce_visuals:
            if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
                visual_names_A.append('idt_B')
                visual_names_B.append('idt_A')

        if self.opt.cerberus:
            # adding the label components to the visualization
            visual_names_A.append('seg_syn_mask_A')
            visual_names_B.append('seg_syn_mask_B')

            visual_names_A.append('seg_syn_contours_A')
            visual_names_B.append('seg_syn_contours_B')

            if not self.opt.reduce_visuals:
                visual_names_A.append('seg_rec_mask_A')            
                if self.opt.detach_advers and self.opt.detach_pcycle:
                    visual_names_B.append('seg_rec_mask_B_detached')
                else:
                    visual_names_B.append('seg_rec_mask_B')
            
            if not self.opt.reduce_visuals:
                visual_names_A.append('seg_rec_contours_A')           
                if self.opt.detach_advers and self.opt.detach_pcycle:
                    visual_names_B.append('seg_rec_contours_B_detached')  
                else:
                    visual_names_B.append('seg_rec_contours_B')  

            if not self.opt.reduce_visuals:
                if self.opt.bcd:
                    visual_names_B.append('seg_syn_distance_B')
                    visual_names_A.append('seg_syn_distance_A')   
                    if self.opt.detach_advers and self.opt.detach_pcycle:
                        visual_names_A.append('seg_rec_distance_B_detached')     
                    else:
                        visual_names_A.append('seg_rec_distance_B')     
                    visual_names_A.append('seg_rec_distance_A')          

            if self.isTrain:
                visual_names_B.append('cerberus_label_mask')
                visual_names_B.append('cerberus_label_contours')  
                if self.opt.bcd and not self.opt.reduce_visuals:
                    visual_names_B.append('cerberus_label_distance') 
                      
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        # added by leander
        if self.opt.cerberus:
            self.pos_names = ['pos_A', 'pos_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            if self.opt.cerberus:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_B_BCD']
            else:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
            
            # added by Leander
            if self.opt.cerberus:
                # discriminator used to evaluate the masks, contour, and distance map generated from real B and reconstructed from fake_A
                if self.opt.cerberus_D_weight_rec > 0.0 or self.opt.cerberus_D_weight_syn > 0.0: 
                    if self.opt.bcd:
                        self.netD_B_BCD = networks.define_D(3, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
                    else:
                        self.netD_B_BCD = networks.define_D(2, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        if self.isTrain:
            if self.opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                if self.opt.cerberus:
                    pass
                else:
                    assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # added by Leander
            if self.opt.cerberus:
                self.real_GT_M_pool = ImagePool(opt.pool_size)
                if self.opt.cerberus_D_weight_syn > 0.0: 
                    self.fake_B_BCD_syn_pool = ImagePool(opt.pool_size)
                if self.opt.cerberus_D_weight_rec > 0.0: 
                    self.fake_B_BCD_rec_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # added by leander
            if self.opt.cerberus:
                # mask and countour loss
                # using BCEWithLogitsLoss since GANLoss also relies on it and we can not apply sigmoids!
                self.criterionMask = torch.nn.BCEWithLogitsLoss()
                self.criterionCont = torch.nn.BCEWithLogitsLoss()
                # distance loss
                self.criterionDist = torch.nn.L1Loss()
                # pseudo cycle consistency
                self.criterionPseudoCycle = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            if self.opt.cerberus:
                if self.opt.cerberus_D_weight_rec > 0.0 or self.opt.cerberus_D_weight_syn > 0.0: 
                    self.optimizer_D_BCD = torch.optim.Adam(itertools.chain(self.netD_B_BCD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D_BCD)

        if self.isTrain:
            if self.opt.cerberus_D_weight_rec == 0.0 and self.opt.detach_advers:
                ConfigException("These configurations do not match", f"If opt.cerberus_D_weight_rec is {self.opt.cerberus_D_weight_rec} there is no point in setting opt.detach_advers to {self.opt.detach_advers} since S_{{B,r}} is never used by the discriminator")
            if self.opt.lambda_B_BCD == 0.0 and self.opt.detach_pcycle:
                ConfigException("These configurations do not match", f"If opt.lambda_B_BCD is {self.opt.lambda_B_BCD} there is no point in setting opt.detach_pcycle to {self.opt.detach_pcycle} since S_{{B,r}} is never used by the pseudo cycle loss")


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
    
        if self.opt.cerberus:
            # the pose is only used by the testing function to label the output volumes
            self.pos_A = input['A_pos']
            self.pos_B = input['B_pos']

            # process the label components
            if self.isTrain:
                self.cerberus_label = input['Label'].to(self.device)
                self.cerberus_label_mask = self.cerberus_label[:,:1,:,:,:]
                self.cerberus_label_contours = self.cerberus_label[:,1:2,:,:,:]

                if self.cerberus_label.shape[1]==3 and self.opt.bcd:
                    self.cerberus_label_distance = self.cerberus_label[:,2:3,:,:,:]
                elif self.cerberus_label.shape[1]==3 and not self.opt.bcd:
                    raise ConfigException("These configurations do not match", f"opt.bcd is {self.opt.bcd} while cerberus_label has {self.cerberus_label.shape[1]} channels")
                else:
                    pass
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        # forward pass A -> B
        self.fake_B = self.netG_A(self.real_A)  # G_A(A) = B_fake
        if self.opt.cerberus:
            # backward pass B_syn -> A_rec
            # only feed the image and not the label components to the G_B
            # the model expects input that is normalized between -1 and 1
            # G_B and G_A do not have a tanh output due to the multi channel nature of the data
            self.rec_A = self.netG_B(torch.tanh(self.fake_B[:,:1,:,:,:]))   # G_B(G_A(A)) = A_rec

            if self.opt.detach_seg:
                # detach version so that D_B_BCD can be used to update G_A with out updating G_B
                # only feed the image and not the label components to the G_A
                # the model expects input that is normalized between -1 and 1 
                # G_B and G_A do not have a tanh output due to the multi channel nature of the data
                self.fake_B_detached = self.fake_B[:,:1,:,:,:].detach().clone()
                self.rec_A_detached = self.netG_B(torch.tanh(self.fake_B_detached))  # G_A(G_B(B)) = B_rec_detached
        else:
            # backward pass B_syn -> A_rec
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        
        # forward pass B -> A
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) = A_fake
        if self.opt.cerberus:
            # backward pass A_syn -> B_rec
            # only feed the image and not the label components to the G_A
            # the model expects input that is normalized between -1 and 1 
            # G_B and G_A do not have a tanh output due to the multi channel nature of the data
            self.rec_B = self.netG_A(torch.tanh(self.fake_A[:,:1,:,:,:]))  # G_A(G_B(B)) = B_rec

            if self.opt.detach_advers or self.opt.detach_pcycle:
                # detach version so that D_B_BCD can be used to update G_A with out updating G_B
                # only feed the image and not the label components to the G_A
                # the model expects input that is normalized between -1 and 1 
                # G_B and G_A do not have a tanh output due to the multi channel nature of the data
                self.fake_A_detached = self.fake_A[:,:1,:,:,:].detach().clone()
                self.rec_B_detached = self.netG_A(torch.tanh(self.fake_A_detached))  # G_A(G_B(B)) = B_rec_detached
        else:
            # backward pass A_syn -> B_rec
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # fot cerberus it is required that we split the image and label components
        # the different components need different normalizations
        if self.opt.cerberus:
            # forward pass A->B
            self.seg_syn_mask_A = self.fake_B[:,1:2,:,:,:] # no sigmoid as we use BCEwithLogits when comparing to real_mask_A
            self.seg_syn_contours_A = self.fake_B[:,2:3,:,:,:]  # no sigmoid as we use BCEwithLogits when comparing to real_contour_A
            # if the generator output has four channels and bcd is set to true process the distance map
            if  self.opt.bcd and self.fake_B.shape[1] == 4:
                self.seg_syn_distance_A = torch.tanh(self.fake_B[:,3:4,:,:,:]) # normalize with tanh as we us an L1 loss
            # D_B evaluates real_B vs fake_B, since real_B is normalized between -1 and 1 we have to apply the same normalization to fake_B
            self.fake_B = torch.tanh(self.fake_B[:,:1,:,:,:]) 

            # backward pass B_syn -> A_rec
            self.seg_rec_mask_A = self.rec_A[:,1:2,:,:,:]  # no sigmoid as we use BCEwithLogits when comparing to real_mask_A
            self.seg_rec_contours_A = self.rec_A[:,2:3,:,:,:]  # no sigmoid as we use BCEwithLogits when comparing to real_contour_A
            if  self.opt.bcd and self.rec_A.shape[1] == 4:
                self.seg_rec_distance_A = torch.tanh(self.rec_A[:,3:4,:,:,:]) # normalize with tanh as we us an L1 loss
            # rec_A is used to ensure cycle consistency to real_B using an L1 loss, since real_A is normalized between -1 and 1 we have to apply the same normalization to rec_A
            self.rec_A = torch.tanh(self.rec_A[:,:1,:,:,:]) # L1 loss in cycle consitency

            if self.opt.detach_seg:
                self.seg_rec_mask_A_detached = self.rec_A_detached[:,1:2,:,:,:] # no sigmoid as we use BCEwithLogits when comparing to real_mask_A
                self.seg_rec_contours_A_detached = self.rec_A_detached[:,2:3,:,:,:] # no sigmoid as we use BCEwithLogits when comparing to real_contour_A
                if  self.opt.bcd and self.rec_A_detached.shape[1] == 4:
                    self.seg_rec_distance_A_detached = torch.tanh(self.rec_A_detached[:,3:4,:,:,:]) # D_B_BCD evaluates real_distance_A vs seg_rec_distance_B_detached, since real_distance_A is normalized between -1 and 1 we have to apply the same normalization to seg_rec_distance_B_detached

            # forward pass B->A
            # we do not have GT label components for B
            self.seg_syn_mask_B = torch.sigmoid(self.fake_A[:,1:2,:,:,:]) # normalize with sigmoid as we apply a L1 loss between seg_syn_mask_B and seg_rec_mask_B_detached 
            self.seg_syn_contours_B = torch.sigmoid(self.fake_A[:,2:3,:,:,:]) # normalize with sigmoid as we apply a L1 loss between seg_syn_contours_B and seg_rec_contours_B_detached 
            # if the generator output has four channels and bcd is set to true process the distance map
            if  self.opt.bcd and self.fake_A.shape[1] == 4:
                self.seg_syn_distance_B = torch.tanh(self.fake_A[:,3:4,:,:,:]) # normalize with tanh as we us an L1 loss
            # D_A evaluates real_A vs fake_A, since real_A is normalized between -1 and 1 we have to apply the same normalization to fake_A
            self.fake_A = torch.tanh(self.fake_A[:,:1,:,:,:])

            # recreation of B
            # we do not have GT label components for B
            # if adverserial and pseudo cycle are detached we have not use for seg_rec_mask_B, seg_rec_contours_B, and seg_rec_distance_, 
            if not (self.opt.detach_advers and self.opt.detach_pcycle):
                # normalize with sigmoid as we apply a L1 loss between seg_syn_mask_B and seg_rec_mask_B 
                # additionally D_B_BCD evaluates real_mask_A vs seg_rec_mask_B, since real_mask_A is normalized between 0 and 1 we have to apply the same normalization to seg_rec_mask_B
                self.seg_rec_mask_B = torch.sigmoid(self.rec_B[:,1:2,:,:,:]) 
                # normalize with sigmoid as we apply a L1 loss between seg_syn_contours_B and seg_rec_contours_B
                # additionally D_B_BCD evaluates real_contours_A vs seg_rec_contours_B, since real_mask_A is normalized between 0 and 1 we have to apply the same normalization to seg_rec_contours_B 
                self.seg_rec_contours_B = torch.sigmoid(self.rec_B[:,2:3,:,:,:]) 
                # if the generator output has four channels and bcd is set to true process the distance map
                if  self.opt.bcd and self.rec_B.shape[1] == 4:
                    self.seg_rec_distance_B = torch.tanh(self.rec_B[:,3:4,:,:,:]) # D_B_BCD evaluates real_distance_A vs seg_rec_distance_B, since real_distance_A is normalized between -1 and 1 we have to apply the same normalization to seg_rec_distance_B
                # rec_B is used to ensure cycle consistency to real_B using an L1 loss, since real_A is normalized between -1 and 1 we have to apply the same normalization to rec_A
                # since rec_B is passed through the whole cycle while real_B never enters the cycle we require the attached version
            self.rec_B = torch.tanh(self.rec_B[:,:1,:,:,:])

            if self.opt.detach_advers or self.opt.detach_pcycle:
                # we have two additional adverserial losses, one for the synthesized B label components vs real A label components, and on for recreadted B label components vs real A label components
                # the adverserial loss based on syn_label_B vs real_label A updates the generator B
                # we nee a detached version of rec_label_B (this means that it is detached from G_B and attached to G_A) for the adverserial loss based on rec_label_B vs real_label A to only update generator A and not also generator A (which would end in updating A twice)
                # additionally we require the detached version for the pseudo cycle consistency. For normal cycle consistency rec_A and rec_B are passed through the whole cycle while real_A and real_B never enter the cycle. 
                # For the pseudo cycle consistency B_{B,r}, B_{C,r}, B_{D,r} run through the whole cycle while A_{B,s}, A_{C,s}, A_{D,s} run through half of the cycle.
                # Accordingly to update G_B only once we nee the a detached version of B_{B,r}, B_{C,r}, B_{D,r} that ran only through the second half of the cycle

                # normalize with sigmoid as we apply a L1 loss between seg_syn_mask_B and seg_rec_mask_B_detached 
                # additionally D_B_BCD evaluates real_mask_A vs seg_rec_mask_B_detached, since real_mask_A is normalized between 0 and 1 we have to apply the same normalization to seg_rec_mask_B_detached
                self.seg_rec_mask_B_detached = torch.sigmoid(self.rec_B_detached[:,1:2,:,:,:]) 
                # normalize with sigmoid as we apply a L1 loss between seg_syn_contours_B and seg_rec_contours_B_detached
                # additionally D_B_BCD evaluates real_contours_A vs seg_rec_contours_B_detached, since real_mask_A is normalized between 0 and 1 we have to apply the same normalization to seg_rec_contours_B_detached 
                self.seg_rec_contours_B_detached = torch.sigmoid(self.rec_B_detached[:,2:3,:,:,:]) 
                # if the generator output has four channels and bcd is set to true process the distance map
                if  self.opt.bcd and self.rec_B_detached.shape[1] == 4:
                    self.seg_rec_distance_B_detached = torch.tanh(self.rec_B_detached[:,3:4,:,:,:]) # D_B_BCD evaluates real_distance_A vs seg_rec_distance_B_detached, since real_distance_A is normalized between -1 and 1 we have to apply the same normalization to seg_rec_distance_B_detached


    def backward_D_basic(self, netD, real, fake, fake2=None):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            fake2 (tensor array) -- images generated by a generator (only not None in cerberus mode)

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        if fake2 == None:
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D
        else:
            # Fake
            # detach both images as they where already applied to D_B_BCD when updating G_B and G_A
            pred_fake = netD(fake.detach())
            pred_fake2 = netD(fake2.detach())

            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D_fake2 = self.criterionGAN(pred_fake2, False)

            # Combined loss and calculate gradients
            loss_D = (loss_D_real + (loss_D_fake + loss_D_fake2 ) * 0.5 ) * 0.5
            loss_D.backward()
            return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # added by Leander
    # update the discriminator that evaluates the generated masks, contours, and distance map
    def backward_D_B_BCD(self):
        """Calculate GAN loss for discriminator D_B_BCD"""
        # query a GT mask sample
        real_GT_M = self.real_GT_M_pool.query(self.cerberus_label)

        # merge the processed mask, contours, and distance map
        if  self.opt.bcd:
            if self.opt.cerberus_D_weight_syn > 0.0: 
                    concat_label_syn = torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B, self.seg_syn_distance_B), axis=1)
            if self.opt.cerberus_D_weight_rec > 0.0:
                if self.opt.detach_advers:
                    concat_label_rec = torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached, self.seg_rec_distance_B_detached), axis=1)
                else:
                    concat_label_rec = torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B, self.seg_rec_distance_B), axis=1)
        else:
            if self.opt.cerberus_D_weight_syn > 0.0: 
                concat_label_syn = torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B), axis=1)
            if self.opt.cerberus_D_weight_rec > 0.0:
                if self.opt.detach_advers:
                    concat_label_rec = torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached), axis=1)
                else:
                    concat_label_rec = torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B), axis=1)
        
        if self.opt.cerberus_D_weight_rec > 0.0 and self.opt.cerberus_D_weight_syn > 0.0: 
            fake_B_BCD_syn = self.fake_B_BCD_syn_pool.query(concat_label_syn)
            fake_B_BCD_rec = self.fake_B_BCD_rec_pool.query(concat_label_rec)
            self.loss_D_B_BCD = self.backward_D_basic(self.netD_B_BCD, real_GT_M, fake_B_BCD_syn, fake_B_BCD_rec)
        elif self.opt.cerberus_D_weight_syn > 0.0:
            fake_B_BCD_syn = self.fake_B_BCD_syn_pool.query(concat_label_syn)
            self.loss_D_B_BCD = self.backward_D_basic(self.netD_B_BCD, real_GT_M, fake_B_BCD_syn)
        elif self.opt.cerberus_D_weight_rec > 0.0:
            fake_B_BCD_rec = self.fake_B_BCD_rec_pool.query(concat_label_rec)
            self.loss_D_B_BCD = self.backward_D_basic(self.netD_B_BCD, real_GT_M, fake_B_BCD_rec)
        else:
            ConfigException("These configurations do not match", f"Why is D_B_BCD used if opt.cerberus_D_weight_rec is {self.opt.cerberus_D_weight_rec} and opt.cerberus_D_weight_syn is {self.opt.cerberus_D_weight_syn}?")
        

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # G_B and G_A do not have a tanh output due to the multi channel nature of the data   
            # real_B is normalized between -1 and 1
            if self.opt.cerberus:
                self.idt_A = torch.tanh(self.netG_A(self.real_B)[:,:1,:,:,:])
            else:
                self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            if self.opt.cerberus:
                self.idt_B = torch.tanh(self.netG_B(self.real_A)[:,:1,:,:,:])
            else:
                self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        if self.opt.cerberus:
            # calculate the cerberus losses
            # forward pass A -> B
            # mask difference
            self.loss_seg_syn_logits_mask = self.criterionMask(self.seg_syn_mask_A, self.cerberus_label_mask) * self.opt.cerberus_mask_weight
            # contour difference
            self.loss_seg_syn_logits_contour = self.criterionCont(self.seg_syn_contours_A, self.cerberus_label_contours) * self.opt.cerberus_contour_weight
            if self.opt.bcd:
                # distance map difference
                self.loss_seg_syn_logits_distance = self.criterionDist(self.seg_syn_distance_A, self.cerberus_label_distance) * self.opt.cerberus_distance_weight

                
            # backward pass (reconstruction) B_syn -> A
            # Cylce consistency loss: A_M ≈ G_B(G_A(A_M))
            # mask difference
            if self.opt.detach_seg:
                self.loss_seg_rec_logits_mask_detached = self.criterionMask(self.seg_rec_mask_A_detached, self.cerberus_label_mask) * self.opt.cerberus_mask_weight
                # contour difference
                self.loss_seg_rec_logits_contour_detached = self.criterionCont(self.seg_rec_contours_A_detached, self.cerberus_label_contours) * self.opt.cerberus_contour_weight
                if self.opt.bcd:
                    # distance map difference
                    self.loss_seg_rec_logits_distance_detached = self.criterionDist(self.seg_rec_distance_A_detached, self.cerberus_label_distance) * self.opt.cerberus_distance_weight
            else:
                self.loss_seg_rec_logits_mask = self.criterionMask(self.seg_rec_mask_A, self.cerberus_label_mask) * self.opt.cerberus_mask_weight
                # contour difference
                self.loss_seg_rec_logits_contour = self.criterionCont(self.seg_rec_contours_A, self.cerberus_label_contours) * self.opt.cerberus_contour_weight
                if self.opt.bcd:
                    # distance map difference
                    self.loss_seg_rec_logits_distance = self.criterionDist(self.seg_rec_distance_A, self.cerberus_label_distance) * self.opt.cerberus_distance_weight

            # (Pseudo-Cycle) Consistency loss: A_C ≈ G_A(B_C)
            lambda_B_BCD = self.opt.lambda_B_BCD
            if lambda_B_BCD > 0.0:
                if self.opt.bcd:
                    if self.opt.detach_pcycle:
                        self.loss_cycle_B_BCD = self.criterionPseudoCycle(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B, self.seg_syn_distance_B), axis=1), 
                                            torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached, self.seg_rec_distance_B_detached), axis=1)) * lambda_B_BCD          
                    else:
                        self.loss_cycle_B_BCD = self.criterionPseudoCycle(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B, self.seg_syn_distance_B), axis=1), 
                                            torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B, self.seg_rec_distance_B), axis=1)) * lambda_B_BCD
                else:
                    if self.opt.detach_pcycle:
                        self.loss_cycle_B_BCD = self.criterionPseudoCycle(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B), axis=1), 
                                            torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached), axis=1)) * lambda_B_BCD
                    else:
                        self.loss_cycle_B_BCD = self.criterionPseudoCycle(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B), axis=1), 
                                            torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B), axis=1)) * lambda_B_BCD
                
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        if self.opt.cerberus:
            # added by leander
            # GAN loss D_B_BCD(G_B(B))
            if self.opt.bcd:
                if self.opt.cerberus_D_weight_syn > 0.0:
                    self.loss_G_B_BCD_syn = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B, self.seg_syn_distance_B), axis=1)), True) * self.opt.cerberus_D_weight_syn
                # also update G_A by applying the discriminator also to S_{B,r}
                if self.opt.cerberus_D_weight_rec > 0.0:
                    if self.opt.detach_advers:
                        self.loss_G_B_BCD_rec = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached, self.seg_rec_distance_B_detached), axis=1)), True) * self.opt.cerberus_D_weight_rec
                    else:
                        self.loss_G_B_BCD_rec = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B, self.seg_rec_distance_B), axis=1)), True) * self.opt.cerberus_D_weight_rec
            else:
                if self.opt.cerberus_D_weight_syn > 0.0:
                    self.loss_G_B_BCD_syn = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_syn_mask_B, self.seg_syn_contours_B), axis=1)), True) * self.opt.cerberus_D_weight_syn
                 # also update G_A by applying the discriminator also to S_{B,r}
                if self.opt.cerberus_D_weight_rec > 0.0:
                    if self.opt.detach_advers:
                        self.loss_G_B_BCD_rec = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_rec_mask_B_detached, self.seg_rec_contours_B_detached), axis=1)), True) * self.opt.cerberus_D_weight_rec
                    else:
                        self.loss_G_B_BCD_rec = self.criterionGAN(self.netD_B_BCD(torch.cat((self.seg_rec_mask_B, self.seg_rec_contours_B), axis=1)), True) * self.opt.cerberus_D_weight_rec



          
        if self.opt.cerberus:
            # add the mask and contour losses for the A->B->A loop
            if self.opt.detach_seg:
                cerberus_loss = self.loss_seg_syn_logits_mask + self.loss_seg_syn_logits_contour + self.loss_seg_rec_logits_mask_detached + self.loss_seg_rec_logits_contour_detached
                # add the distance based losses for the A->B->A loop
                if self.opt.bcd:
                    cerberus_loss = cerberus_loss + self.loss_seg_syn_logits_distance + self.loss_seg_rec_logits_distance_detached
            else:
                cerberus_loss = self.loss_seg_syn_logits_mask + self.loss_seg_syn_logits_contour + self.loss_seg_rec_logits_mask + self.loss_seg_rec_logits_contour
                # add the distance based losses for the A->B->A loop
                if self.opt.bcd:
                    cerberus_loss = cerberus_loss + self.loss_seg_syn_logits_distance + self.loss_seg_rec_logits_distance
            # add the adverserial losses
            if self.opt.cerberus_D_weight_syn > 0.0:
                cerberus_loss = cerberus_loss + self.loss_G_B_BCD_syn
            if self.opt.cerberus_D_weight_rec > 0.0:
                cerberus_loss = cerberus_loss + self.loss_G_B_BCD_rec
            if lambda_B_BCD > 0.0:
                cerberus_loss = cerberus_loss + self.loss_cycle_B_BCD

               

            self.loss_G = cerberus_loss + self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        else:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        # added by Leander
        # D_B_BCD
        if self.opt.cerberus_D_weight_rec > 0.0 or self.opt.cerberus_D_weight_syn > 0.0: 
            self.set_requires_grad([self.netD_B_BCD], True)
            self.optimizer_D_BCD.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_B_BCD()      # calculate graidents for D_B
            self.optimizer_D_BCD.step()  # update D_A and D_B's weights
