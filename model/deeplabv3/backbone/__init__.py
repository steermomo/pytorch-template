from model.deeplabv3.backbone import resnet, mobilenet  # xception, drn,


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        # return xception.AlignedXception(output_stride, BatchNorm)
        raise NotImplementedError
    elif backbone == 'drn':
        # return drn.drn_d_54(BatchNorm)
        raise NotImplementedError
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError