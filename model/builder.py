from util.registry import Registry

BACKBONE = Registry("backbone")
NECK = Registry("neck")
TEMPORAL = Registry("temporal")
VIEWTRANSFORMATION = Registry("view_transformation")
LDM_MODEL = Registry("ldm_model")


def build_backbone(backbone_type: str, backbone_config: dict):
    cls = BACKBONE.get(backbone_type)
    return cls(backbone_config)


def build_neck(neck_type: str, neck_config: dict):
    cls = NECK.get(neck_type)
    return cls(neck_config)


def build_temporal(temporal_type: str, temporal_config: dict):
    cls = TEMPORAL.get(temporal_type)
    return cls(temporal_config)


def build_view_transformation(view_transformation_type: str, view_transformation_config: dict):
    cls = VIEWTRANSFORMATION.get(view_transformation_type)
    return cls(view_transformation_config)


BACKBONE_OUT_CHANNELS_DICT = {
    'RegNetX200MF': [56, 152, 368],
    'RegNetX400MF': [64, 160, 384],
    'RegNetX600MF': [96, 240, 528],
    'RegNetX800MF': [128, 288, 672],
    'RegNetX1.6GF': [168, 408, 912]
}
