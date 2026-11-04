from models.st_gt import STGT_Encoder
from models.star.star import STAR
from models.gno import GNO_Decoder

class SeismicReconModel():
    def __init__(self):
        self.encoder = STAR()
        self.decoder = GNO_Decoder()

    def forward(self, x):
        latent = self.encoder.forward(x)
        pred = self.decoder.forward(latent)