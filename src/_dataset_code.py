# ============================================================
# 3. Dataset, masks, and augmentations
# ============================================================

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1,
}


def polygon_list(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        polygons = []
        for item in geometry.geoms:
            polygons.extend(polygon_list(item))
        return polygons
    return []


def rasterize_mask(label_path, image_shape, is_post):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.int64)
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data["features"]["xy"]:
        geometry = wkt.loads(obj["wkt"])
        for poly in polygon_list(geometry):
            if poly.is_empty or poly.area < 1:
                continue
            coords = np.asarray(poly.exterior.coords)
            if coords.shape[0] < 3:
                continue
            rr, cc = draw_polygon(coords[:, 1], coords[:, 0], shape=(h, w))
            if rr.size == 0:
                continue
            if is_post:
                subtype = obj["properties"].get("subtype", "un-classified")
                mask[rr, cc] = DAMAGE_MAP.get(subtype, 1)
            else:
                mask[rr, cc] = 1
    return mask


def color_jitter(img):
    img = img.astype(np.float32) / 255.0
    b = 1.0 + random.uniform(-CONFIG["aug_brightness"], CONFIG["aug_brightness"])
    c = 1.0 + random.uniform(-CONFIG["aug_contrast"], CONFIG["aug_contrast"])
    s = 1.0 + random.uniform(-CONFIG["aug_saturation"], CONFIG["aug_saturation"])
    img = np.clip(img * b, 0.0, 1.0)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = np.clip((img - mean) * c + mean, 0.0, 1.0)
    gray = img.mean(axis=2, keepdims=True)
    img = np.clip(gray + s * (img - gray), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def apply_train_augmentations(pre_img, post_img, pre_mask, post_mask):
    if random.random() < CONFIG["aug_hflip"]:
        pre_img = np.flip(pre_img, axis=1).copy()
        post_img = np.flip(post_img, axis=1).copy()
        pre_mask = np.flip(pre_mask, axis=1).copy()
        post_mask = np.flip(post_mask, axis=1).copy()

    if random.random() < CONFIG["aug_vflip"]:
        pre_img = np.flip(pre_img, axis=0).copy()
        post_img = np.flip(post_img, axis=0).copy()
        pre_mask = np.flip(pre_mask, axis=0).copy()
        post_mask = np.flip(post_mask, axis=0).copy()

    if random.random() < CONFIG["aug_rotate90"]:
        k = random.choice([1, 2, 3])
        pre_img = np.rot90(pre_img, k=k).copy()
        post_img = np.rot90(post_img, k=k).copy()
        pre_mask = np.rot90(pre_mask, k=k).copy()
        post_mask = np.rot90(post_mask, k=k).copy()

    pre_img = color_jitter(pre_img)
    post_img = color_jitter(post_img)

    if random.random() < CONFIG["aug_blur_prob"]:
        pre_img = cv2.GaussianBlur(pre_img, (3, 3), 0)
    if random.random() < CONFIG["aug_blur_prob"]:
        post_img = cv2.GaussianBlur(post_img, (3, 3), 0)

    if random.random() < CONFIG["aug_noise_prob"]:
        noise = np.random.normal(0, 8, size=pre_img.shape).astype(np.float32)
        pre_img = np.clip(pre_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < CONFIG["aug_noise_prob"]:
        noise = np.random.normal(0, 8, size=post_img.shape).astype(np.float32)
        post_img = np.clip(post_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return pre_img, post_img, pre_mask, post_mask


class XView2Dataset(Dataset):
    def __init__(self, data_root, sample_ids, train=False):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.sample_ids = list(sample_ids)
        self.train = train

    def __len__(self):
        return len(self.sample_ids)

    def load_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def resize_image(self, image):
        target_h, target_w = CONFIG["image_size"]
        if image.shape[:2] == (target_h, target_w):
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def resize_mask(self, mask):
        target_h, target_w = CONFIG["image_size"]
        if mask.shape[:2] == (target_h, target_w):
            return mask
        return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

    def normalize(self, image):
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        pre_img = self.load_image(self.images_dir / f"{sample_id}_pre_disaster.png")
        post_img = self.load_image(self.images_dir / f"{sample_id}_post_disaster.png")
        pre_mask = rasterize_mask(self.labels_dir / f"{sample_id}_pre_disaster.json", pre_img.shape, False)
        post_mask = rasterize_mask(self.labels_dir / f"{sample_id}_post_disaster.json", post_img.shape, True)

        pre_img = self.resize_image(pre_img)
        post_img = self.resize_image(post_img)
        pre_mask = self.resize_mask(pre_mask)
        post_mask = self.resize_mask(post_mask)

        if self.train:
            pre_img, post_img, pre_mask, post_mask = apply_train_augmentations(pre_img, post_img, pre_mask, post_mask)

        return {
            "sample_id": sample_id,
            "pre_image": self.normalize(pre_img),
            "post_image": self.normalize(post_img),
            "pre_mask": torch.from_numpy(pre_mask).long(),
            "post_mask": torch.from_numpy(post_mask).long(),
        }


train_dataset = XView2Dataset(CONFIG["data_root"], TRAIN_IDS, train=True)
val_dataset = XView2Dataset(CONFIG["data_root"], VAL_IDS, train=False)
test_dataset = XView2Dataset(CONFIG["data_root"], TEST_IDS, train=False)

_loader_kwargs = dict(
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
    persistent_workers=CONFIG["num_workers"] > 0,
    prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
)
train_loader = DataLoader(train_dataset, shuffle=True, **_loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **_loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **_loader_kwargs)
