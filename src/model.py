from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import roi_align
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from backbone import Backbone
from dataset import ConvertToOriginalSize
import torch.nn as nn
import torch

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = Backbone()
        self.anchorGenerator = AnchorGenerator(
          sizes=((4, 8, 16, 32), (16, 32, 64 ,128), (32, 64, 128, 256)),
          aspect_ratios=((0.2, 0.5, 1.0, 2.0, 5.0),) * 3,
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchorGenerator,
            head=RPNHead(
                in_channels=256,
                num_anchors=self.anchorGenerator.num_anchors_per_location()[0],
            ),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.5,
        )
        self.roi_heads = RoIHeads(
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=[
                    "0",
                    "1",
                    "2",
                ],
                output_size=14,
                sampling_ratio=2,
            ),
            box_head=TwoMLPHead(in_channels=256 * 14 * 14, representation_size=1024),
            box_predictor=FastRCNNPredictor(in_channels=1024, num_classes=num_classes),
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            score_thresh=0.3,
            nms_thresh=0.3,
            detections_per_img=50,
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
        )
        self.resize = ConvertToOriginalSize()

    def forward(self, images, targets=None):
        # Feature extraction
        image_sizes = [img.shape[-2:] for img in images]
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        features = self.backbone(images) 
        features_dict = features
        rpn_boxes, rpn_losses = self.rpn(
            ImageList(images, image_sizes), features_dict, targets
        )
        detections, detector_losses = self.roi_heads(
            features_dict, rpn_boxes, [tuple(item) for item in image_sizes], targets
        )
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
        else:
            return detections
