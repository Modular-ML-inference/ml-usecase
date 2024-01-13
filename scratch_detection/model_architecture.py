import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.rpn.nms_thresh = 0.6
    model.rpn.score_thresh = 0.01
    model.rpn.anchor_generator.aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 3.0, 4.0), (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), 
                                                (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), 
                                                (0.2, 0.5, 1.0, 2.0, 3.0, 4.0))
    model.rpn.anchor_generator.sizes = ((32,), (64,), (128,), (256,), (512,))
    model.rpn._pre_nms_top_n['training'] = 2000
    model.rpn._pre_nms_top_n['testing'] = 1000
    model.rpn._post_nms_top_n['training'] = 1000
    model.rpn._post_nms_top_n['testing'] = 500


    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 12 
    model.roi_heads.box_detections_per_img = 12
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model