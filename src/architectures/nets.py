from torch import nn, trace
import pytorch_lightning as pl
import torch


class ConvNet1(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet1, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, obs_size, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(obs_size, 16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU(),
                                nn.Linear(32, n_actions))

    def forward(self, x):
        x = self.cnn_base(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.fc(x)
        return q_values


class ConvNetRawSegment(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet1, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, obs_size, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(obs_size, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(256, 200), nn.ReLU(),
                                nn.Linear(200, 48), nn.ReLU(),
                                nn.Linear(48, n_actions))

    def forward(self, x, x_seg):
        out_1 = self.cnn_base(x)
        out_2 = self.cnn_base(x_seg)
        out_1 = torch.flatten(out_1, start_dim=1)
        out_2 = torch.flatten(out_2, start_dim=1)
        x = out_1 + out_2
        q_values = self.fc(x)
        return q_values


class CNNAutoEncoder(pl.LightningModule):
    """
    Simple auto-encoder with MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        self.hidden_size: size of hidden layers
    """
    def __init__(self, hparams, z_size: int = 32):
        super(CNNAutoEncoder, self).__init__()

        # Parameters
        image_size = hparams.image_size
        self.example_input_array = torch.randn((1, *image_size))

        self.encoder = nn.Sequential(
            nn.Conv2d(image_size[0], 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=6, stride=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=6, stride=3), nn.ReLU())

        self.hidden_size = self._get_flatten_size()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 128, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(128, 128, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(), nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_size[0], kernel_size=4, stride=2),
            nn.Sigmoid())
        self.to_mu = nn.Linear(self.hidden_size, z_size)
        self.to_log_var = nn.Linear(self.hidden_size, z_size)
        self.z_to_hidden = nn.Linear(z_size, self.hidden_size)

    @torch.no_grad()
    def _get_flatten_size(self):
        x = self.encoder(self.example_input_array)
        return x.shape[-1]

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.to_mu(h), self.to_log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        z, mu, log_var = self.bottleneck(h)
        z = self.z_to_hidden(z)

        z = z.view(z.size(0), self.hidden_size, 1, 1)  # Unflatten
        x_out = self.decoder(z)
        return x_out, mu, log_var


class CNNAuxNet(pl.LightningModule):
    '''
    Contains a a simple auto-encoder with decoder and MLPs for auxiliary tasks.
    '''
    def __init__(self, hparams, z_size: int = 32):
        super(CNNAuxNet, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = [torch.randn((1, obs_size, 256, 256)), torch.randn((1, 3))]          # image and sensor

        ####################################
        #### 1. CNN with BN
        ####################################
        # self.encoder = nn.Sequential(
        # nn.Conv2d(obs_size, 32, kernel_size=7, stride=3), nn.BatchNorm2d(32), nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.BatchNorm2d(64), nn.ReLU(),
        # nn.Conv2d(64, 128, kernel_size=5, stride=3), nn.BatchNorm2d(128), nn.ReLU(),
        # nn.Conv2d(128, 128, kernel_size=3, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
        # nn.Conv2d(128, 256, kernel_size=3, stride=2)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=0), nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(), 
        #     nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, obs_size, kernel_size=7, stride=3),
        #     nn.Sigmoid())


        ####################################
        #### 2. CNN without BN, (with dropout) 
        ####################################
        ### 128 latent size
        # self.encoder = nn.Sequential(
        # nn.Conv2d(obs_size, 32, kernel_size=7, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(32, 32, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(64, 128, kernel_size=3, stride=2)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=0), nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, obs_size, kernel_size=7, stride=3),
        #     nn.Sigmoid()
        # )

        ### 256 latent size
        self.encoder = nn.Sequential(
        nn.Conv2d(obs_size, 32, kernel_size=7, stride=3), nn.Dropout(.2), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.Dropout(.2), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(32, obs_size, kernel_size=7, stride=3),
            nn.Sigmoid()
        )

        ### 1024 latent size
        # self.encoder = nn.Sequential(
        # nn.Conv2d(obs_size, 32, kernel_size=7, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(64, 128, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(256, 1024, kernel_size=3, stride=2)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, output_padding=0), nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, obs_size, kernel_size=7, stride=3),
        #     nn.Sigmoid()
        # )

        ####################################
        #### 3. CNN without BN, with dropout, kernel_1 = 5,not 7 
        ####################################
        # self.encoder = nn.Sequential(
        # nn.Conv2d(obs_size, 32, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(32, 32, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.Dropout(.2), nn.ReLU(),
        # nn.Conv2d(64, 128, kernel_size=3, stride=2)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=0), nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3, output_padding=1), nn.ReLU(), 
        #     nn.ConvTranspose2d(32, obs_size, kernel_size=5, stride=3, output_padding=2),
        #     nn.Sigmoid()
        # )


        ####################################
        #### 3. CNN other
        ####################################
        # self.encoder = nn.Sequential(
        # nn.Conv2d(obs_size, 32, kernel_size=7, stride=3),  nn.ReLU(),
        # nn.Conv2d(32, 32, kernel_size=5, stride=3), nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=5, stride=3), nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
        # nn.Conv2d(64, 128, kernel_size=3, stride=2)
        # )


        # self.encoder = nn.Sequential(
        #     nn.Conv2d(obs_size, 32, kernel_size=4, stride=2), nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=6, stride=3), nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=6, stride=3), nn.ReLU())

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=6, stride=3, output_padding=1),
        #     nn.ReLU(), nn.ConvTranspose2d(128, 64, kernel_size=6, stride=3, output_padding=2),
        #     nn.ReLU(), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),
        #     nn.ReLU(), nn.ConvTranspose2d(32, obs_size, kernel_size=4, stride=2),
        #     nn.Sigmoid())



        numparams = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('encoder params:', numparams)
        print('encoder output shape:', self._get_flatten_size())
        ####################################
        ####################################
        #### Autopilot & Aux MLPs
        ####################################
        sensor_len = 3
        auxtask_len = 3
        self.autopilotAC = nn.Sequential(nn.Linear(2*128 + 0, 1*64), nn.ReLU(),
                        nn.Linear(1*64, 1*32), nn.ReLU(),
                        nn.Linear(1*32, n_actions))

        # self.trafficlight = nn.Sequential(nn.Linear(2*128 + 0, 64), nn.ReLU(),
        #                         nn.Linear(64, 32), nn.ReLU(),
        #                         nn.Linear(32, 3)) # red, green, none

        # self.trafficlight_feed =  nn.Sequential(nn.Linear(3, 18), nn.LeakyReLU(),
        #                                 nn.Linear(18, 18))

        # print(self.state_dict().items())
        # print('level00')
        # for  param in self.parameters():
        #     print(param.requires_grad)
        # print('level0')
        # # for name, param in self.state_dict().items():
        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)
        

    def freezeLayers(self, selected_subnet=['encoder','decoder']):
        '''Freezes selected subnet layers
        '''
        for name, param in self.named_parameters():
            for subnet in selected_subnet: 
                if subnet in name:
                    param.requires_grad = False
        print('\nFozen selected layers. Trainable weights are:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def loadWeights(self, ckpt_path, selected_subnet=['encoder', 'decoder']):
        '''loads selected subnet weights
        '''
        
        print('Loading weights.........................')
        checkpoint = torch.load(ckpt_path)
        trained_params = checkpoint['state_dict']
        for name, param in self.state_dict().items():
            for subnet in selected_subnet: 
                if subnet in name:
                    tr_param = trained_params['net.' + name]
                    param.copy_(tr_param)
                    print(subnet)
        print('Loaded selected weights complete')
        

    @torch.no_grad()
    def _get_flatten_size(self):
        x = self.encoder(self.example_input_array[0])
        return x.shape


    def forward(self, x):

        d_out, traffic_out, act_out = None, None, None
        ##########################################
        ### Image CNN feed & reconstruction
        ##########################################
        h = self.encoder(x[0])
        d_out = self.decoder(h)
        hflat = torch.flatten(h, start_dim=1)    

        # latent = torch.cat((hflat, x[1]), dim=1)                 # add sensor data
        
        ##########################################
        ### Traffic light
        ##########################################
        # traffic_out = self.trafficlight(latent)                  # with sensor data
        # traffic_out = self.trafficlight(hflat)                   # no sensor data
        # traffic_feed = self.trafficlight_feed(traffic_out)       # scaling inermidiate MLP

        ##########################################
        ### Autopilot Action
        ##########################################
        # final_latent = torch.cat((latent, traffic_out), dim=1)   # concatenate aux output + sensor data
        # final_latent = torch.cat((hflat, traffic_out), dim=1)    # concatenate aux output  
        # final_latent = torch.cat((hflat, traffic_feed), dim=1)   # use aux scaling MLP
        final_latent = hflat                                     # only hidden vector   
        
        act_out = self.autopilotAC(final_latent)
        
        ##########################################
        ### zero other outputs based on config
        ##########################################
        if d_out is None:       d_out = 0
        if traffic_out is None: traffic_out = 0
        if act_out is None:     act_out = 0

        
        out = [d_out, traffic_out, act_out]
        return out
    

