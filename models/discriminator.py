from torch import nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims=256, hidden_dims=64, output_dims=1):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        # self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )
        # self.layer = nn.Sequential(
        #     nn.Linear(input_dims, hidden_dims),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims, output_dims)
        # )
        self.apply(weights_init)

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out