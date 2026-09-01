from torch import nn

from ..encoders.atomic_encoders import AtomicGraphEncoder as Encoder_atomic
from ..encoders.atomic_encoders import GenericFeatureEncoder as Encoder_generic
from ..processors.order_processor import OrderProcessor as Processor
from ..decoders.standard_decoders import ScalarDecoder as Decoder
from ..decoders.standard_decoders import PositiveScalarsDecoder
from ..decoders.standard_decoders import PositiveKChannelDecoder
from ..decoders.standard_decoders import PositiveFeatureReadout
from .gnn_builder import GenericGNN


class ALIGNN(GenericGNN):
    def __init__(self, encoder, processor, decoder):
        super().__init__(
            encoder=encoder,
            processor=processor,
            decoder=decoder,
            name="ALIGNN",
        )