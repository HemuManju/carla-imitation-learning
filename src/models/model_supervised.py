import torch
import torchvision.models as models
from torch import nn


def create_resnet_basic_block(
    width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out
):
    basic_block = nn.Sequential(
        nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode="nearest"),
        nn.Conv2d(
            nb_channel_in,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            nb_channel_out,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
    )
    return basic_block


class Model_Segmentation_Traffic_Light_Supervised(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super().__init__()
        if hparams['crop_sky']:
            self.size_state_RL = 6144
        else:
            self.size_state_RL = 8192
        if hparams['rnn_ae']:
            self.size_state_RL = 1024       # for RNN training
        resnet18 = models.resnet18(pretrained=hparams['pretrained'])
        
        # See https://arxiv.org/abs/1606.02147v1 section 4: Information-preserving
        # dimensionality changes
        #
        # "When downsampling, the first 1x1 projection of the convolutional branch is performed
        # with a stride of 2 in both dimensions, which effectively discards 75% of the input.
        # Increasing the filter size to 2x2 allows to take the full input into consideration,
        # and thus improves the information flow and accuracy."

        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        resnet18.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer4[0].downsample[0].kernel_size = (2, 2)

        assert resnet18.layer2[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer3[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer4[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        self.example_input_array = [torch.randn((32, 3, 188, 288)), torch.randn((32, 3))]
        self.semseg = hparams['semseg']
        self.traffic_status = hparams['traffic_status']
        self.traffic_dist = hparams['traffic_dist']
        self.dist_car = hparams['dist_car']
        self.action = hparams['action']

        self.use_aux =  hparams['use_aux']
        self.use_sensor = hparams['use_sensor']
        self.use_hlcmd = hparams['use_hlcmd']      # high-level command

        nb_images_input = hparams['nb_images_input']
        nb_images_output = hparams['nb_images_output']
        hidden_size = hparams['hidden_size']
        nb_class_segmentation = hparams['nb_class_segmentation']


        ### to adapt to the number of input
        # if nb_images_input != 1:
        # new_conv1 = nn.Conv2d(
        #     nb_images_input * 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )
        # resnet18.conv1 = new_conv1

        self.encoder = nn.Sequential(
            *(list(resnet18.children())[:-2])
        )  # resnet18_no_fc_no_avgpool
        self.last_conv_downsample = nn.Sequential(
            # nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False),
            # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.AdaptiveAvgPool2d((1,2)),   # for reducdd latent size for RNN training
        )

        if self.traffic_status:
            # Classification red, green, No traffic light
            self.fc1_traffic_light_inters = nn.Linear(self.size_state_RL, hidden_size)
            self.fc2_traffic_light_state = nn.Linear(hidden_size, 3)
        
        if self.traffic_dist:
            # classification on the distance to traffic_light  
            self.fc2_distance_to_tl = nn.Linear(hidden_size, 4)

        if self.dist_car:
            # classification on the distance to the front car
            self.fc1_dist_to_frontcar = nn.Linear(self.size_state_RL, hidden_size)
            self.fc2_dist_to_frontcar = nn.Linear(hidden_size, 4)


        if self.semseg: 
            # We will upsample image with nearest neightboord interpolation between each umsample block
            # https://distill.pub/2016/deconv-checkerboard/
            self.up_sampled_block_0 = create_resnet_basic_block(6, 8, 512, 512)
            self.up_sampled_block_1 = create_resnet_basic_block(12, 16, 512, 256)
            self.up_sampled_block_2 = create_resnet_basic_block(24, 32, 256, 128)
            self.up_sampled_block_3 = create_resnet_basic_block(48, 64, 128, 64)
            self.up_sampled_block_4 = create_resnet_basic_block(74, 128, 64, 32)   # for semseg
            # self.up_sampled_block_4 = create_resnet_basic_block(188, 288, 64, 32)    # for image reconstruction  

            self.last_conv_segmentation = nn.Conv2d(
                32,
                nb_class_segmentation * nb_images_output,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            )
            self.last_bn = nn.BatchNorm2d(
                nb_class_segmentation * nb_images_output,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

        aux_len = int(self.use_aux) * (3 + 4 + 4)
        sensor_len = int(self.use_sensor) * 2

        if self.action:
            num_actions = 20
            if not self.use_hlcmd:
                self.fc_action = nn.Sequential(
                    nn.Linear(2 * self.size_state_RL + aux_len + sensor_len, hidden_size),  # 2X size for RNN encoding
                    nn.LeakyReLU(),
                    nn.Linear(hidden_size, int(hidden_size/2)),
                    nn.LeakyReLU(),
                    nn.Linear(int(hidden_size/2), num_actions)
                )
            else:
                self.fc_action_base = nn.Sequential(
                    nn.Linear(self.size_state_RL + aux_len + sensor_len, hidden_size), nn.LeakyReLU(),
                )

                self.fc_action1 = nn.Sequential(
                    nn.Linear(hidden_size, int(hidden_size/2)), nn.LeakyReLU(),
                    nn.Linear(int(hidden_size/2), num_actions)
                )

                self.fc_action2 = nn.Sequential(
                    nn.Linear(hidden_size, int(hidden_size/2)), nn.LeakyReLU(),
                    nn.Linear(int(hidden_size/2), num_actions)
                )

                self.fc_action3 = nn.Sequential(
                    nn.Linear(hidden_size, int(hidden_size/2)), nn.LeakyReLU(),
                    nn.Linear(int(hidden_size/2), num_actions)
                )

                self.fc_action4 = nn.Sequential(
                   nn.Linear(hidden_size, int(hidden_size/2)), nn.LeakyReLU(),
                    nn.Linear(int(hidden_size/2), num_actions)
                )

        
        numparams = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('encoder params:', numparams)


    def freezeLayers(self, selected_subnet=['encoder','decoder'], exclude_mode=False):
        '''Freezes selected subnet layers (or everything else if exclude_mode is true)
        '''
        print('Frezing layers ...\nPutting modules in eval() mode:')
        for name, param in self.named_children():
            for subnet in selected_subnet: 
                if not exclude_mode and subnet in name: 
                    param.eval()
                    print('\t',name)
                elif exclude_mode and subnet not in name:
                    param.eval()
                    print('\t',name)
        print('Finshed putting modules in eval mode\n')

        for name, param in self.named_parameters():
            for subnet in selected_subnet: 
                if not exclude_mode and subnet in name: 
                    param.requires_grad = False
                elif exclude_mode and subnet not in name:
                    param.requires_grad = False

        print('\nFozen selected layers. Trainable weights are:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('\t',name)
        return

    def loadWeights(self, ckpt_path, selected_subnet=['encoder', 'decoder'], exclude_mode=False):
        '''loads selected subnet weights (or everything else if exclude_mode is true)
        '''
        
        print('Loading weights.........................')
        checkpoint = torch.load(ckpt_path)
        trained_params = checkpoint['state_dict']
        for name, param in self.state_dict().items():
            for subnet in selected_subnet:
                if not exclude_mode and subnet in name:
                    tr_param = trained_params['net.' + name]
                    param.copy_(tr_param)
                    print('\t',subnet)
                elif exclude_mode and subnet not in name: 
                    tr_param = trained_params['net.' + name]
                    param.copy_(tr_param)
                    print('\t',name)
        print('Loaded selected weights complete!***************************\n')
        return


    def forwardAction(self, classif_state_net=None, x=None, tl_state_output=None,
     dist_to_tl_output=None, dist_to_frontcar=None):
        '''Forwad through the action network
        '''

        if self.use_sensor and self.use_aux:
            aux_out = torch.cat((tl_state_output, dist_to_tl_output, dist_to_frontcar), dim=1)
            act_input = torch.cat((aux_out, x[1]), dim=1)               # append sensor data
            act_input = torch.cat((classif_state_net, act_input), dim=1)
        elif self.use_aux:
            aux_out = torch.cat((tl_state_output, dist_to_tl_output, dist_to_frontcar), dim=1)
            act_input = torch.cat((classif_state_net, aux_out), dim=1)
        elif self.use_sensor:
            act_input = torch.cat((classif_state_net, x[1]), dim=1)          # append sensor data
        else:
            act_input = classif_state_net

        if self.use_hlcmd:
            act_input = self.fc_action_base(act_input)
            act_out = []
            for cmd in range(1,5):
                ind = torch.where(x[1][:,0]==cmd)
                act_inp = act_input[ind]

                # forward pass
                if cmd == 1:
                    act_out.append(self.fc_action1(act_inp))
                elif cmd == 2:
                    act_out.append(self.fc_action2(act_inp))
                elif cmd == 3:
                    act_out.append(self.fc_action3(act_inp))
                elif cmd == 4:
                    act_out.append(self.fc_action4(act_inp))    
            act = torch.cat(act_out)
        else:
            act = self.fc_action(act_input)
        
        return act


    def forward(self, x):
        """[summary]
        NOTE: Please do the following:
            -remove no_grad() when training a particular sub-module.
            -change action network sizes to appropirate if this is a part of RNN AE.

        Args:
            x (list of Tensors): input to the network. shape (modality, (batch, ...)))

        Returns:
            [Tensors]: Different prediction of the network
        """        

        with torch.no_grad():
            # Encoder first, resnet18 without last fc and abg pooling
            encoding = self.encoder(x[0])  # 512*4*4 or 512*4*3 (crop sky)

            encoding = self.last_conv_downsample(encoding)

        out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar, act = None, None,  None, None, None
        if self.semseg:
            # Segmentation branch
            upsample0 = self.up_sampled_block_0(encoding)  # 512*8*8 or 512*6*8 (crop sky)
            upsample1 = self.up_sampled_block_1(upsample0)  # 256*16*16 or 256*12*16 (crop sky)
            upsample2 = self.up_sampled_block_2(upsample1)  # 128*32*32 or 128*24*32 (crop sky)
            upsample3 = self.up_sampled_block_3(upsample2)  # 64*64*64 or 64*48*64 (crop sky)
            upsample4 = self.up_sampled_block_4(upsample3)  # 32*128*128 or 32*74*128 (crop sky)
            
            ### for segmentation
            out_seg = self.last_bn(
                self.last_conv_segmentation(upsample4)
            )  # nb_class_segmentation*128*128

            ### For image reconstruction
            # out_seg = nn.functional.relu(self.last_conv_segmentation(upsample4))


        # Classification branch, traffic_light (+ state), intersection or none
        classif_state_net = encoding.view(-1, self.size_state_RL)
       
        if self.traffic_status:
            traffic_light_state_net = self.fc1_traffic_light_inters(classif_state_net)
            traffic_light_state_net = nn.functional.relu(traffic_light_state_net)
            tl_state_output = self.fc2_traffic_light_state(traffic_light_state_net)

        if self.traffic_dist:
            dist_to_tl_output = self.fc2_distance_to_tl(traffic_light_state_net)
        
        if self.dist_car:
            # dist_to_frontcar = self.fc_dist_to_frontcar(classif_state_net)
            dist_to_frontcar = self.fc1_dist_to_frontcar(classif_state_net)
            dist_to_frontcar = nn.functional.relu(dist_to_frontcar)
            dist_to_frontcar = self.fc2_dist_to_frontcar(dist_to_frontcar)

        if self.action:
            self.forwardAction(classif_state_net, x, tl_state_output, dist_to_tl_output, dist_to_frontcar)


        return out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar, act
