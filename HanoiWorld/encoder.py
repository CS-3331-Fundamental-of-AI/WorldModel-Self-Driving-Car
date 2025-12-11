import torch
import torch.nn as nn
import torchvision.models as tvm


class FrozenEncoder(nn.Module):
    """
    TorchVision backbone frozen in eval mode with a small projection
    head to a fixed embedding size (default 128). Suitable for feeding
    precomputed embeddings into the RSSM without training the encoder.

    This is just for mocking & testing the enviroment
    """

    def __init__(self, backbone="resnet18", out_dim=128, weights="IMAGENET1K_V1", device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        encoder, feat_dim = self._build_backbone(backbone, weights)
        self.encoder = encoder.to(self.device)
        self.proj = nn.Linear(feat_dim, out_dim, bias=True).to(self.device)
        # Cache normalization stats as buffers to avoid reallocating each call.
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        # Freeze everything, including the projection unless you unfreeze manually.
        self.out_dim = out_dim
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, C, H, W) in range [0, 1] or uint8.
        If input is (B, H, W, C), it will be permuted automatically.
        """
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            # Assume NHWC -> NCHW
            x = x.permute(0, 3, 1, 2)
        # Expand grayscale to 3 channels for pretrained weights
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Sanity check channels
        if x.dim() != 4 or x.shape[1] not in (3,):
            raise ValueError(f"Expected input as NCHW with 1 or 3 channels, got {x.shape}")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.to(self.device)
        # Standard ImageNet normalization
        x = (x - self._mean) / self._std
        with torch.no_grad():
            feats = self.encoder(x)
            feats = torch.flatten(feats, 1)
            emb = self.proj(feats)
        return emb

    def _build_backbone(self, name, weights):
        """
        Returns (encoder_without_head, feature_dim).
        Supported: resnet18, resnet34, resnet50.
        """
        if name == "resnet18":
            weights_enum = tvm.ResNet18_Weights if hasattr(tvm, "ResNet18_Weights") else None
            model = tvm.resnet18(weights=getattr(weights_enum, weights) if weights_enum and weights else None)
        elif name == "resnet34":
            weights_enum = tvm.ResNet34_Weights if hasattr(tvm, "ResNet34_Weights") else None
            model = tvm.resnet34(weights=getattr(weights_enum, weights) if weights_enum and weights else None)
        elif name == "resnet50":
            weights_enum = tvm.ResNet50_Weights if hasattr(tvm, "ResNet50_Weights") else None
            model = tvm.resnet50(weights=getattr(weights_enum, weights) if weights_enum and weights else None)
        else:
            raise ValueError(f"Unsupported backbone '{name}'")

        feat_dim = model.fc.in_features
        # Drop classification head; keep everything up to the penultimate pooling.
        encoder = nn.Sequential(*(list(model.children())[:-1]))
        return encoder, feat_dim
