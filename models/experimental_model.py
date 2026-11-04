from models.st_gt import STGT_Encoder
from models.gno import GNO_Decoder

class SeismicReconModel():
    def __init__(self):
        self.encoder = STGT_Encoder()
        self.decoder = GNO_Decoder()

    def forward(self, x):
        latent = self.encoder.forward(x)
        pred = self.decoder.forward(latent)