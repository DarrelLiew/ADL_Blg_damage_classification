# ============================================================
# 5. Siamese ResNet U-Net model
# ============================================================

RESNET_CHANNELS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],
}


def build_backbone(name, pretrained=True):
    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
    }
    constructor = getattr(models, name)
    if not pretrained:
        return constructor(weights=None)
    try:
        return constructor(weights=weights_map[name])
    except Exception as exc:
        print(f"Could not load pretrained weights, using random init instead: {exc}")
        return constructor(weights=None)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.channels = RESNET_CHANNELS[backbone_name]

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = ConvBlock(channels * 3, channels)

    def forward(self, pre_feat, post_feat):
        diff = torch.abs(post_feat - pre_feat)
        x = torch.cat([pre_feat, post_feat, diff], dim=1)
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SiameseResUNet(nn.Module):
    def __init__(self, backbone="resnet34", num_classes=5, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(backbone, pretrained=pretrained)
        enc = self.encoder.channels
        dec = CONFIG["decoder_channels"]

        self.fuse0 = FusionBlock(enc[0])
        self.fuse1 = FusionBlock(enc[1])
        self.fuse2 = FusionBlock(enc[2])
        self.fuse3 = FusionBlock(enc[3])
        self.fuse4 = FusionBlock(enc[4])

        self.bottleneck = ConvBlock(enc[4], dec[0])
        self.dec4 = DecoderBlock(dec[0], enc[3], dec[0])
        self.dec3 = DecoderBlock(dec[0], enc[2], dec[1])
        self.dec2 = DecoderBlock(dec[1], enc[1], dec[2])
        self.dec1 = DecoderBlock(dec[2], enc[0], dec[3])
        self.head = nn.Conv2d(dec[3], num_classes, kernel_size=1)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.encoder.parameters())
        print(f"Encoder FROZEN ({n:,} params)")

    def unfreeze_encoder(self, n_layers=2):
        layers = [self.encoder.stem, self.encoder.layer1, self.encoder.layer2,
                  self.encoder.layer3, self.encoder.layer4]
        for layer in layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.encoder.stem[1].parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze last {n_layers} encoder layers — trainable: {trainable:,} params")

    def forward(self, pre_img, post_img):
        out_size = pre_img.shape[-2:]
        pre = self.encoder(pre_img)
        post = self.encoder(post_img)

        f0 = self.fuse0(pre[0], post[0])
        f1 = self.fuse1(pre[1], post[1])
        f2 = self.fuse2(pre[2], post[2])
        f3 = self.fuse3(pre[3], post[3])
        f4 = self.fuse4(pre[4], post[4])

        x = self.bottleneck(f4)
        x = self.dec4(x, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        x = self.head(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


model = SiameseResUNet(
    backbone=CONFIG["backbone"],
    num_classes=CONFIG["num_classes"],
    pretrained=CONFIG["pretrained"],
).to(DEVICE)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")


with torch.no_grad():
    test_logits = model(batch["pre_image"].to(DEVICE), batch["post_image"].to(DEVICE))
print("Model output:", tuple(test_logits.shape))
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
