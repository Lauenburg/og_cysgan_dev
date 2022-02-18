from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    __test__ = False


    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--one_way', type=bool, default='False', help='Only inference testA and not testB')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False

        # define the image volumes that should be saved in cerberus mode
        parser.add_argument('--save_vol_A', type=str, default="0000000", help= "Volumes are saved if a 1 stands at there index. Order of volumes: fake_B, seg_syn_mask_A, seg_syn_contours_A, seg_rec_mask_A, seg_rec_contours_A, seg_syn_distance_A, seg_rec_distance_A" )
        parser.add_argument('--save_vol_B', type=str, default="0000000", help= "Volumes are saved if a 1 stands at there index. Order of volumes: fake_A, seg_syn_mask_B, seg_syn_contours_B, seg_rec_mask_B, seg_rec_contours_B, seg_syn_distance_B, seg_rec_distance_B" )
        return parser
