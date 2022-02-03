import torch
import torchvision.models as models
from torch import nn




class Model_Segmentation_Traffic_Light_Supervised_RNN(nn.Module):
    def __init__(
        self,
        hparams,
        autoencoder,
    ):
        super().__init__()
        

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
        
        ####################################################
        ### Loading the modules from the autoencoder
        self.encoder = autoencoder.encoder
        self.last_conv_downsample = autoencoder.last_conv_downsample
        
        # TODO: better be done in the AE model later
        
        if self.semseg:
            self.decoder = nn.Sequential(
                autoencoder.up_sampled_block_0,  # 512*8*8 or 512*6*8 (crop sky)
                autoencoder.up_sampled_block_1,  # 256*16*16 or 256*12*16 (crop sky)
                autoencoder.up_sampled_block_2,  # 128*32*32 or 128*24*32 (crop sky)
                autoencoder.up_sampled_block_3,  # 64*64*64 or 64*48*64 (crop sky)
                autoencoder.up_sampled_block_4,  # 32*128*128 or 32*74*128 (crop sky)
                
                ### for segmentation  
                autoencoder.last_conv_segmentation,
                autoencoder.last_bn         # nb_class_segmentation*128*128
            )
        if self.traffic_status:
            self.fc1_traffic_light_inters = autoencoder.fc1_traffic_light_inters
            self.fc2_traffic_light_state = autoencoder.fc2_traffic_light_state
        if self.traffic_dist:
            self.fc2_distance_to_tl = autoencoder.fc2_distance_to_tl
        if self.dist_car:
            self.fc1_dist_to_frontcar = autoencoder.fc1_dist_to_frontcar
            self.fc2_dist_to_frontcar = autoencoder.fc2_dist_to_frontcar
        
        if self.action:
            self.forwardAction = autoencoder.forwardAction

        ####################################################
        ### The RNN
        self.device = torch.device('cuda:0')
        self.hidden_size = 1024
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        numparams = sum(p.numel() for p in self.rnn.parameters() if p.requires_grad)
        print('RNN params:', numparams)

        self.example_input_array = [torch.randn((32, 4, 3, 188, 288)), torch.randn((32, 4, 3))]


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

    def encodetoRNNInput(self, x):
        """[summary]

        Args:
            x ([type]): input to the network

        Returns:
            [type]: latent vector from the encoder before the rnn. shape: (batch, seq, hidden)
        """        
        batch_sz, seq_len, num_ch, h, w = x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3], x[0].shape[4] 

        encoder_input = x[0].contiguous().view(batch_sz*seq_len, num_ch, h, w)
        encoding = self.encoder(encoder_input)  # 512*4*4 or 512*4*3 (crop sky)
        encoding = self.last_conv_downsample(encoding)

        rnn_input = encoding.view(batch_sz, seq_len, -1)        # (batch_sz, seq, hidden)

        return rnn_input

    def decodeRNNtoImage(self, latent):
        """[summary]

        Args:
            latent ([type]): rnn output. shape: (batch, seq, hidden)

        Returns:
            [type]: predicted images reconstructed/segmented. shape: (batch*seq, image_dim..).
                - image_dim = (num_semseg_cls, 74, 128) for semseg or (3, 188, 288) for RGB
        """
        out_seg = None
        last_ch, last_h, last_w = 512, 3, 4
        print(latent.shape)
        batch_sz, seq_len, _ = latent.shape
        decoder_input = latent.contiguous().view(batch_sz*seq_len, last_ch, last_h, last_w)
        out_seg = self.decoder(decoder_input)        
        
        return out_seg

    
    def decodeRNNtoAux(self, x, latent):
        """[summary]

        Args:
            x ([type]): input to the network
            latent ([type]): rnn output of shape: (batch, seq, hidden)

        Returns:
            [type]: auxiliary predictions. shape: (numtasks, batch*seq, task_dim)
        """        

        #################################################
        ### auxiliary prediction
        tl_state_output, dist_to_tl_output, dist_to_frontcar = None, None, None

        latent = latent.contiguous().view(-1, self.hidden_size)       # (batch*seq, hidden_size)

        if self.traffic_status:
            traffic_light_state_net = self.fc1_traffic_light_inters(latent)
            traffic_light_state_net = nn.functional.relu(traffic_light_state_net)
            tl_state_output = self.fc2_traffic_light_state(traffic_light_state_net)

        if self.traffic_dist:
            dist_to_tl_output = self.fc2_distance_to_tl(traffic_light_state_net)
        
        if self.dist_car:
            dist_to_frontcar = self.fc1_dist_to_frontcar(latent)
            dist_to_frontcar = nn.functional.relu(dist_to_frontcar)
            dist_to_frontcar = self.fc2_dist_to_frontcar(dist_to_frontcar)

        return tl_state_output, dist_to_tl_output, dist_to_frontcar



    def forward(self, x):
        """[summary]

        Args:
            x (list of list of Tensors): input to the network. shape (modality, (batch, seqlen ...)))

        Returns:
            [type]: [description]
        """
        batch_sz = x[0].shape[0]

        ### Encode input images
        with torch.no_grad():  # if frozen CNN
            rnn_input = self.encodetoRNNInput(x)
        
        ### feed latent up to the last t-step to the RNN (t_final used for ground-truth loss calc.)
        h_0 = torch.zeros(1, batch_sz, self.hidden_size).to(self.device)
        rnn_out, h_out = self.rnn(rnn_input[:,:-1,...], h_0)       # (batch, seq-1, hidden) - skip last time-step

        ##################################
        ### Post RNN
        ##################################
        ### image decoding
        out_seg = None
        # with torch.no_grad():  # if forzen decoder
            # out_seg = self.decodeRNNtoImage(rnn_out)
        
        ##################################
        ### auxiliary prediction
        tl_state_output, dist_to_tl_output, dist_to_frontcar = None, None, None
        # with torch.no_grad():  # if frozen auxiliary MLPs
            # tl_state_output, dist_to_tl_output, dist_to_frontcar = self.decodeRNNtoAux(x, rnn_out)

        ##################################
        ### action
        act = None
        # act_latent = torch.cat((rnn_input[:,-2,...], h_out[:,-1,...]))  # l_t and h_t
        # act = self.forwardAction(act_latent, x, tl_state_output, dist_to_tl_output, dist_to_frontcar)

        return out_seg, tl_state_output, dist_to_tl_output, dist_to_frontcar, act, rnn_input, rnn_out
