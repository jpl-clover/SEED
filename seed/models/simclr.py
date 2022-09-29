import torch.nn as nn


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(
        self,
        encoder,
        num_projection_layers,
        projection_dim,
        n_features,
        proj_out_layer=-1,
    ):
        super(SimCLR, self).__init__()
        # TODO: Can n_features be automatically detected by the SimCLR constructor?
        self.n_features = n_features
        self.encoder = encoder
        # Replace the fc layer with an Identity function
        self.encoder.fc = nn.Identity()

        self.projector = ContrastiveHead(
            self.n_features,
            out_dim=projection_dim,
            num_layers=num_projection_layers,
            out_layer=proj_out_layer,
        )

    # make a setter for proj_output_layer

    @property
    def proj_output_layer(self):
        return self.projector.out_layer

    @proj_output_layer.setter
    def proj_output_layer(self, proj_out_layer):
        """
        This changes the output layer returned by the ContrastiveHead.
        For example, if proj_out_layer = 0, then the output of the ContrastiveHead == the output of self.encoder
        NOTE: since 0 is the encoder output, proj_out_layer = 1 corresponds to the 0th layer of our ContrastiveHead,
        so if you want to use the middle layer (i.e., layer1 in [layer0, layer1, layer2]),
        then pass in proj_out_layer = 2, which corresponds to layer1
        -1 Also corresponds to the last layer
        """
        assert -1 <= proj_out_layer <= self.projector.num_layers
        if proj_out_layer == -1:
            proj_out_layer = self.projector.num_layers
        self.projector.out_layer = proj_out_layer

    def forward(self, x_i, x_j=None):
        h_i = self.encoder(x_i)
        z_i = self.projector(h_i)
        if self.training and x_j is not None:
            h_j = self.encoder(x_j)
            z_j = self.projector(h_j)
            return h_i, z_i, h_j, z_j
        else:
            # note that this is different from the jpl-clover/SimCLR repo
            return z_i


BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well


class ContrastiveHead(nn.Module):
    def __init__(
        self, channels_in, out_dim=128, num_layers=3, out_layer=3, use_relu=True
    ):
        super().__init__()
        self.num_layers = num_layers
        if out_layer == -1:
            self.out_layer = num_layers
        else:
            self.out_layer = out_layer
        self.layers = nn.ModuleList()
        self.layer_indices = []
        for _ in range(num_layers - 1):
            self.layers.extend(
                [
                    nn.Linear(channels_in, channels_in, bias=False),
                    nn.BatchNorm1d(channels_in, eps=BATCH_NORM_EPSILON, affine=True),
                    nn.ReLU() if use_relu else nn.Identity(),
                ]
            )
            self.layer_indices.append(len(self.layers) - 1)

        # last layer (no reLU even if use_relu=True)
        self.layers.append(nn.Linear(channels_in, out_dim, bias=False))
        bn = nn.BatchNorm1d(out_dim, eps=BATCH_NORM_EPSILON, affine=True)
        nn.init.zeros_(bn.bias)
        self.layers.append(bn)
        self.layer_indices.append(len(self.layers) - 1)

    def forward(self, x):
        # TODO: figure out how to obtain intermediate outputs for fine-tuning
        if self.out_layer == 0:
            # Do nothing; just return the input from SimCLR.encoder
            return x
        else:
            layer_outputs = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.layer_indices:
                    layer_outputs.append(x)
            return layer_outputs[self.out_layer - 1]
