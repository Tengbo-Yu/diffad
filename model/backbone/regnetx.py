from model.base_module import BaseModule
from model.backbone.pycls_model_zoo import regnetx
from model.builder import BACKBONE


class BaseRegNet(BaseModule):
    def __init__(self, backbone_config, trunk):
        super().__init__()
        self.backbone_config = backbone_config

        del trunk.head
        self.trunk = trunk

    def forward(self, x):
        layers = self.trunk.named_children()
        outputs = []
        y = x
        for name, layer in layers:
            y = layer(y)
            if name in ['s2', 's3', 's4']:
                outputs.append(y)
        return outputs


@BACKBONE.register()
class RegNetX(BaseRegNet):
    def __init__(self, backbone_config):
        backbone_sub_type = backbone_config['sub_type']
        trunk = regnetx(backbone_sub_type, pretrained=True)
        super().__init__(backbone_config, trunk)
