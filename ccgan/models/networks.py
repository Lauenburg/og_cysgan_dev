import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from connectomics.model.arch import unet



###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance', dim=2):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
        dim (int)       -- the dimension of the data (2D or 3D): 2 | 3

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    assert(dim==2 or dim==3), "The dimension of the data needs to be 2 for 2D or 3 for 3D"

    if norm_type == 'batch':
        if dim == 2:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif dim == 3:
            norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        if dim == 2:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif dim == 3:
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        opt -- config options

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dim=opt.data_dim)

    if netG == 'resnet_9blocks':
        if opt.data_dim == 2:
            net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, opt=opt)
        elif opt.data_dim == 3:
            net = ResnetGenerator3D(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, no_convT=opt.no_convT)
    elif netG == 'resnet_6blocks':
        if opt.data_dim == 2:
            net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, opt=opt)
        elif opt.data_dim == 3:
            net = ResnetGenerator3D(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, no_convT=opt.no_convT)
    elif netG == 'unet_32':
        if opt.data_dim == 2:
            raise NotImplementedError("For 2D image data you have to use resnet_9blocks, resnet_6blocks, unet_128 or unet_256")        
        elif opt.data_dim == 3:
            net = UnetGenerator(input_nc, output_nc, 1, ngf, norm_layer=norm_layer, use_dropout=use_dropout, data_dim=3, not_iso=opt.not_iso, no_convT=opt.no_convT, opt=opt)
    elif netG == 'pyc_unet':
        if opt.data_dim == 2:
            raise NotImplementedError("For 3D volume data you have to use Resnet or unet_32")
        elif opt.data_dim == 3:
            net = unet.UNet3D(in_channel=input_nc, out_channel=output_nc)
    elif netG == 'cerberus':
        if opt.data_dim == 2:
            raise NotImplementedError("For 3D volume data you have to use Resnet or unet_32")
        elif opt.data_dim == 3:
            if opt.bcd:
                # image, mask, contour, distance map
                net = unet.UNet3D(in_channel=1, out_channel=4, block_type='residual_se', filters=[32, 64, 96, 128, 160], norm_mode='gn')
            else:
                # image, mask, contour
                net = unet.UNet3D(in_channel=1, out_channel=3, block_type='residual_se', filters=[32, 64, 96, 128, 160], norm_mode='gn')
    elif netG == 'unet_64':
        if opt.data_dim == 2:
            raise NotImplementedError("For 2D image data you have to use resnet_9blocks, resnet_6blocks, unet_128 or unet_256")        
        elif opt.data_dim == 3:
            net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, data_dim=3, not_iso=opt.not_iso, no_convT=opt.no_convT)
    elif netG == 'unet_128':
        if opt.data_dim == 2:
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        elif opt.data_dim == 3:
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, data_dim=3, not_iso=opt.not_iso, no_convT=opt.no_convT)
    elif netG == 'unet_256':
        if opt.data_dim == 2:
            net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        elif opt.data_dim == 3:
            raise NotImplementedError("For 3D volume data you have to use Resnet or unet_32")
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm, dim=opt.data_dim)

    if netD == 'basic':  # default PatchGAN classifier
        if opt.data_dim == 2:
            net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif opt.data_dim == 3:
            net = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
            pass
    elif netD == 'n_layers':  # more options
        if opt.data_dim == 2:
            net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif opt.data_dim == 3:
            net = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
            pass
    elif netD == 'pixel':     # classify if each pixel is real or fake
        if opt.data_dim == 2:
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        elif opt.data_dim == 3:
            raise NotImplementedError("For 3D volume data you have to use the NLayerDiscriminator")
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            opt                 -- config options
        """
        # retrive the training config options from the cyclegan module
        self.opt = opt

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]

        if self.opt is not None and self.opt.seg_match:
            print("SEG MATCH config")
            gen_head = []
            seg_head = []
            # generation head
            
            gen_head += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            gen_head += [nn.Tanh()]

            # segmentation head
            seg_head += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
            seg_head += [nn.Conv2d(output_nc, 1, kernel_size=1, padding=0)]

            self.model = nn.Sequential(*model)
            self.gen_head = nn.Sequential(*gen_head)
            self.seg_head = nn.Sequential(*seg_head)

        elif False:
            # TODO: Forward pass for disc configuration
            pass
        else:
            print("VANILLA config")
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]

            self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.opt is not None and self.opt.seg_match:
            """Seg match forward"""
            out = self.model(input)
            gen = self.gen_head(out)
            seg = self.seg_head(out)
            return gen, seg
        elif False:
            # TODO: Forward pass for disc configuration
            pass
        else:
            """Standard forward"""
            return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetGenerator3D(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', p2=False, no_convT=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            p2                  -- if the dimensions of the input are equal 2^n set True, if equal 2^n+1 set false
            convT               -- If set to True the upsampling is achieved via transpose convolution, else NN upsampling and convolution is used
            
            The convT option is based on the observations by Zhang, Z.  et al. mentioned in  Translating and Segmenting Multimodal Medical Volumes with Cycle- andShape-Consistency Generative Adversarial Network:
            “It is also observed in [25] that transpose-convolution can cause checkerboard artifacts due to the uneven overlapping of convolutional kernels. 
            Actually, this effect is even severer for 3D transpose-convolutions as [. . .]” (Zhang, Z.  et al. (2018), p. 5)
        """
        print("ResnetGenerator3D")
        assert(n_blocks >= 0)
        super(ResnetGenerator3D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ConstantPad3d(3,0),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if not no_convT:
                print("transpose convolution")
                if not p2:
                    model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                                kernel_size=3, stride=2,
                                                padding=1, output_padding=0,
                                                bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]
                else:
                    model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                                kernel_size=3, stride=2,
                                                padding=1, output_padding=1,
                                                bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]                   
            else:
                print("NN upsampling and convolution")
                model += [nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=(5-i),
                                            padding=(i+1)),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        model += [nn.ConstantPad3d(3,0)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, data_dim=2, not_iso=False, no_convT=False, opt=None):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            not_iso  (bool)   -- we only use 3D convolutions for the innermost layers where so data is close to isotropic


        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.not_iso=not_iso
        self.opt=opt
        # construct unet structure
        if data_dim == 2:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
            for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            # gradually reduce the number of filters from ngf * 8 to ngf
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        elif data_dim == 3:
            if no_convT:
                # construct unet structure
                unet_block = UnetSkipConnectionBlock3D_no_tConv(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, block_count=0, num_downs=num_downs, not_iso=self.not_iso, opt=self.opt)  # add the innermost layer
                counter = 0
                for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
                    counter += 1
                    unet_block = UnetSkipConnectionBlock3D_no_tConv(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, not_iso=self.not_iso, num_downs=num_downs, block_count=counter, opt=self.opt)
                # gradually reduce the number of filters from ngf * 8 to ngf
                unet_block = UnetSkipConnectionBlock3D_no_tConv(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,block_count=counter+1, num_downs=num_downs, opt=self.opt)
                unet_block = UnetSkipConnectionBlock3D_no_tConv(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,block_count=counter+2, num_downs=num_downs, opt=self.opt)
                unet_block = UnetSkipConnectionBlock3D_no_tConv(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,block_count=counter+3, num_downs=num_downs, opt=self.opt)
                self.model = UnetSkipConnectionBlock3D_no_tConv(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, block_count=counter+4, num_downs=num_downs, opt=self.opt)
            else:   
                unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, not_iso=self.not_iso)  # add the innermost layer
                for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
                    unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, not_iso=self.not_iso)
                # gradually reduce the number of filters from ngf * 8 to ngf
                unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
                unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
                unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
                self.model = UnetSkipConnectionBlock3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        print(self.model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        # normalization layer dependent bias in convoluational layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # if the now input channel number is given, input channel number is set to outer channel number
        if input_nc is None:
            input_nc = outer_nc
        
        # define layers for encoding block (downsampling)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # define layers for decoding block (upsampling)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # assamble blocks and add to model
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # add a sinlge outer downsampling conv layer with input_nc = number of channels in input images
            down = [downconv]
            # add outer decoder block with Tanh() activation function
            up = [uprelu, upconv, nn.Tanh()]
            # add down and up block left and right to the existing submodule
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)            
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock3D(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, not_iso=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            not_iso  (bool)   -- we only use 3D convolutions for the innermost layers where so data is close to isotropic
        """
        print(not_iso)
        super(UnetSkipConnectionBlock3D, self).__init__()
        self.outermost = outermost
        self.not_iso = not_iso

        # normalization layer dependent bias in convoluational layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # if the now input channel number is given, input channel number is set to outer channel number
        if input_nc is None:
            input_nc = outer_nc
        
        # define layers for encoding block (downsampling)
        downconv2D = nn.Conv3d(input_nc, inner_nc, kernel_size=(1,4,4), stride=(1,2,2),
                                padding=(0,1,1), bias=use_bias)
        
        downconv3D = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                     stride=2, padding=1, bias=use_bias)
        
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # define layers for decoding block (upsampling)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # assamble blocks and add to model
        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(1,4,4), stride=(1,2,2),
                                        padding=(0,1,1))
            # add a sinlge outer downsampling conv layer with input_nc = number of channels in input images
            down = [downconv2D]
            # add outer decoder block with Tanh() activation function
            up = [uprelu, upconv, nn.Tanh()]
            # add down and up block left and right to the existing submodule
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv3D]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if self.not_iso:
                upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv3D, downnorm]
                up = [uprelu, upconv, upnorm]
            else:
                upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=(1,4,4), stride=(1,2,2),
                                            padding=(0,1,1), bias=use_bias)
                down = [downrelu, downconv2D, downnorm]
                up = [uprelu, upconv, upnorm]
            

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)            
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock3D_no_tConv(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, block_count=0, not_iso=False, num_downs=5, opt=None):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            not_iso  (bool)   -- we only use 3D convolutions for the innermost layers where so datat is close to not_iso
        """
        super(UnetSkipConnectionBlock3D_no_tConv, self).__init__()
        self.outermost = outermost
        self.not_iso = not_iso

        # normalization layer dependent bias in convoluational layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # if the now input channel number is given, input channel number is set to outer channel number
        if input_nc is None:
            input_nc = outer_nc
        
        # define layers for encoding block (downsampling)
        downconv2D = nn.Conv3d(input_nc, inner_nc, kernel_size=(1,4,4), stride=(1,2,2),
                                padding=(0,1,1), bias=use_bias)
        
        downconv3D = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                     stride=2, padding=1, bias=use_bias)
        
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # define layers for decoding block (upsampling)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # assamble blocks and add to model
        if outermost:
            upconv = [nn.Conv3d(inner_nc * 2, outer_nc,
                                        kernel_size=(1,4,4), stride=(1,1,1),
                                        padding=(0,1,1)),
                     Interpolate((opt.vs_z,opt.vs_x,opt.vs_y))]
            # add a sinlge outer downsampling conv layer with input_nc = number of channels in input images
            down = [downconv2D]
            # add outer decoder block with Tanh() activation function
            up = [uprelu, *upconv, nn.Tanh()]
            # add down and up block left and right to the existing submodule
            model = down + [submodule] + up
        elif innermost:
            upconv = [nn.Conv3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=1,
                                        padding=1, bias=use_bias),
                     Interpolate((int(opt.vs_z/(2**(num_downs-5))),int(opt.vs_x/2**(num_downs-1)),int(opt.vs_y/2**(num_downs-1))))]
            down = [downrelu, downconv3D]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            if self.not_iso:
                upconv = [nn.Conv3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=1,
                                        padding=1, bias=use_bias),
                         Interpolate((int(opt.vs_z/(2**(num_downs - 5 - block_count))),int(opt.vs_x/2**(num_downs-1-block_count)),int(opt.vs_y/2**(num_downs-1-block_count))))]
                down = [downrelu, downconv3D, downnorm]
                up = [uprelu, *upconv, upnorm]
            else:
                upconv = [nn.Conv3d(inner_nc * 2, outer_nc,
                                            kernel_size=(1,4,4), stride=(1,1,1),
                                            padding=(0,1,1), bias=use_bias),
                         Interpolate((opt.vs_z,int(opt.vs_x/2**(num_downs-1-block_count)),int(opt.vs_y/2**(num_downs-1-block_count))))]
                down = [downrelu, downconv2D, downnorm]
                up = [uprelu, *upconv, upnorm]
            

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)   
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.size = size
        
    def forward(self, x):
        x = self.interpolate(x, self.size, mode='trilinear', align_corners=True)
        return x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
        
class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        # downsample fewer times in z direction since the data is not isotropic
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,padw,padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
