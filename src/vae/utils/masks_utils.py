import os
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
import glob


class LIDCMasks(torch.utils.data.Dataset):
    def __init__(self, directory, mask_mode="none", num_classes=7, 
                 use_onehot=False, split="train", val_ratio=0.1, seed=42, sdf_flag=False, original_textures=False, sdf_truncation=20, spacing=None,
                 patch_size=None, patch_pos_ratio=0.7):
        """
        directory: expected to contain masks as .npy
        mask_mode: controls preprocessing of the mask
        num_classes: total number of segmentation classes
        use_onehot: if True -> return one-hot mask [C,D,H,W], else single-channel [1,D,H,W]
        split: "train" or "val"
        val_ratio: fraction of data used for validation
        mask_part: "nodule", "lung", or "all" 
        patch_size: [D,H,W] crop size, or None to use the full volume (256^3) as-is.
            Hardware-forced deviation from the paper's full-volume mask VAE training --
            see decision log. None preserves the original (pre-patch) behavior exactly.
            ONLY APPLIED WHEN split=="train" -- see __getitem__. Validation is always
            full-volume (256^3), uncropped, matching the paper authors' own validation
            protocol; this argument is accepted for val instances too (so the training
            script's call sites stay symmetric) but is a no-op there by design.
        patch_pos_ratio: only used when split=="train" and patch_size is set. Fraction of
            crops centered on a nodule voxel ("positive" sample) rather than a fully random
            location. Nodules are small relative to a 256^3 volume, so plain uniform random
            cropping risks many patches never containing one -- this counters that, the same
            way the earlier (v1) project's patch sampling did.
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.mask_mode = mask_mode
        self.num_classes = num_classes
        self.use_onehot = use_onehot
        self.sdf_flag = sdf_flag
        self.sdf_truncation = sdf_truncation
        self.spacing = spacing
        self.original_textures = original_textures
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        self.patch_pos_ratio = patch_pos_ratio
        self.is_train = (split == "train")

        # collect all mask paths
        all_paths = sorted(glob.glob(self.directory + "/**/mask/*.npy", recursive=True))
        self.n_images = len(all_paths)
        print("Number of matched masks:", len(all_paths))
        if self.n_images == 0:
            raise RuntimeError(f"No masks found in {self.directory}")

        # reproducible train/val split
        np.random.seed(seed)
        indices = np.arange(self.n_images)
        np.random.shuffle(indices)
        val_count = round(self.n_images * val_ratio)
        print(f"Using {self.n_images - val_count} images for training, {val_count} for validation")

        if split == "train":
            selected_indices = indices[val_count:]
        elif split == "val":
            selected_indices = indices[:val_count]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_paths = [all_paths[i] for i in selected_indices]
        self.n_images = len(self.image_paths)


    def _foreground_class_ids(self):
        """Class IDs that represent nodule presence (excludes plain lung tissue -- lung is
        common and gets sampled by chance anyway; nodules are the rare class patch training
        risks missing entirely)."""
        if self.mask_mode == "nodule":
            return [1]
        elif self.mask_mode == "nodule+lung":
            return [1]
        elif self.mask_mode == "nodule+lung+texture":
            return [1, 2, 3, 4, 5]
        else:
            return []

    def _crop_bounds(self, mask_arr):
        """Returns (start_d, start_h, start_w, pd, ph, pw) for cropping mask_arr to
        self.patch_size. Train: patch_pos_ratio fraction of crops are centered on a
        random nodule voxel (falls back to plain random if this volume has none), the
        rest are plain random. Val: always a deterministic center crop, no randomness,
        so val loss/dice is comparable across epochs rather than noisy per-epoch."""
        D, H, W = mask_arr.shape
        pd, ph, pw = self.patch_size
        pd, ph, pw = min(pd, D), min(ph, H), min(pw, W)
        max_d, max_h, max_w = D - pd, H - ph, W - pw

        if self.is_train:
            if np.random.rand() < self.patch_pos_ratio:
                fg_ids = self._foreground_class_ids()
                if fg_ids:
                    coords = np.argwhere(np.isin(mask_arr, fg_ids))
                    if len(coords) > 0:
                        cz, cy, cx = coords[np.random.randint(len(coords))]
                        start_d = int(np.clip(cz - pd // 2, 0, max_d))
                        start_h = int(np.clip(cy - ph // 2, 0, max_h))
                        start_w = int(np.clip(cx - pw // 2, 0, max_w))
                        return start_d, start_h, start_w, pd, ph, pw
            # negative sample, or this volume has no foreground voxels to center on
            start_d = np.random.randint(0, max_d + 1) if max_d > 0 else 0
            start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
            start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        else:
            start_d, start_h, start_w = max_d // 2, max_h // 2, max_w // 2

        return start_d, start_h, start_w, pd, ph, pw

    def compute_sdf(self, mask, truncation=20, spacing=None):
        dt_out = distance_transform_edt(mask == 0, sampling=spacing)
        dt_in  = distance_transform_edt(mask == 1, sampling=spacing)
        sdf = dt_out - dt_in
        sdf = np.clip(sdf, -truncation, truncation) / truncation
        return sdf.astype(np.float32)

    def get_foreground_classes(self, mask_mode):
        """Defines which labels to compute SDF for depending on mask_mode."""
        if mask_mode == "nodule":
            return [1]  # only nodule
        elif mask_mode == "lung":
            return [1]  # lung only (after remap)
        elif mask_mode == "nodule+lung":
            return [1, 2]
        elif mask_mode == "nodule+lung+texture":
            return [1, 2, 3, 4, 5, 6]  # textures + lung
        else:
            return []
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Mask path {image_path} not found!")

        mask_arr = np.load(image_path)  # shape [D,H,W]

        # remap depending on mask_mode
        if self.mask_mode != "none":
            mask_arr = mask_arr.copy()
            
            if self.mask_mode == "nodule":
                mask_arr = (mask_arr >= 1).astype(np.int64)  # 0=background, 1=nodule
            elif self.mask_mode == "nodule+lung":
                mask_arr[mask_arr >= 1] = 1  # nodules
                mask_arr[mask_arr == 0.5] = 2  # lungs
            elif self.mask_mode == "lung":
                mask_arr[mask_arr > 0.5] = 0 # background
                mask_arr[mask_arr < 0.5] = 0  # background
                mask_arr[mask_arr == 0.5] = 1  # lungs
            elif self.mask_mode=="nodule+lung+texture" and not self.original_textures:  # 
                #reassign nodule texture so it is balanced (check current value 1-5 and change it for a random between 1-5)
                # Convert lung values (0.5) to 6
                mask_arr[mask_arr == 0.5] = 6
                # Round the array to nearest integer and convert to int
                mask_arr = np.round(mask_arr).astype(int)
                # Find all unique nodule labels (1–5, or however many nodules you have)
                nodule_labels = np.unique(mask_arr)
                nodule_labels = nodule_labels[(nodule_labels >= 1) & (nodule_labels <= 5)]
                # Reassign each nodule label to a new random texture between 1–5
                for label in nodule_labels:
                    new_texture = np.random.randint(1, 6)
                    mask_arr[mask_arr == label] = new_texture
                    
            elif self.mask_mode=="nodule+lung+texture" and self.original_textures:  
                mask_arr[mask_arr == 0.5] = 6

            mask_arr = mask_arr.astype(np.int64)

        # === Patch cropping (hardware-forced deviation from full-volume training -- see
        # decision log) === applied after remap, before SDF/one-hot, so everything downstream
        # operates on the cropped patch consistently.
        #
        # TRAIN ONLY. Validation always stays full-volume (256^3), uncropped -- this is not
        # an optimization, it's a fidelity requirement: the paper's own authors validate on
        # complete volumes (their config_vae_masks_train.json trains AND validates at 256^3),
        # and validation is a forward-pass-only workload (no backward pass, no optimizer
        # state, no gradient storage), so it does not carry the same VRAM pressure that forced
        # patch-based TRAINING. Cropping validation too would have been a second, avoidable
        # deviation stacked on top of the first, and would have made nodule-class val Dice
        # measure "did this patient's nodule happen to fall inside a fixed center crop" rather
        # than actual reconstruction quality -- nodules are scattered through the lung, not
        # centered, so this would silently and systematically penalize/inflate val metrics for
        # reasons unrelated to the model. is_train is False for split="val" regardless of what
        # val_patch_size the config or CLI provides, so this check makes that config key inert
        # for validation, by design.
        if self.patch_size is not None and self.is_train:
            start_d, start_h, start_w, pd, ph, pw = self._crop_bounds(mask_arr)
            mask_arr = mask_arr[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]

        out_dict = {"filename": image_path}

        # === SDF generation ===
        if self.sdf_flag:
            class_list = self.get_foreground_classes(self.mask_mode)
            sdf_channels = []
            for cls_id in class_list:
                binary = (mask_arr == cls_id).astype(np.uint8)
                sdf_map = self.compute_sdf(binary, truncation=self.sdf_truncation, spacing=self.spacing)
                sdf_channels.append(sdf_map[None])  # [1,D,H,W]
            if len(sdf_channels) > 0:
                sdf_tensor = torch.from_numpy(np.concatenate(sdf_channels, axis=0))  # [C,D,H,W]
            else:
                # no foreground — return empty channel or zeros
                sdf_tensor = torch.zeros((1,) + mask_arr.shape, dtype=torch.float32)
            out_dict["mask_sdf"] = sdf_tensor

        # === Original mask outputs ===
        #if not self.sdf_flag :
        mask_index = torch.from_numpy(mask_arr).long()
        if self.use_onehot:
            # One-hot encoding for model input/output (high memory)
            mask_input = F.one_hot(mask_index, num_classes=self.num_classes).permute(3,0,1,2).float()  # [C,D,H,W]
        else:
            # Single channel input (low memory)
            mask_input = mask_index.unsqueeze(0).float()  # [1,D,H,W]
        out_dict["mask_input"] = mask_input
        out_dict["mask_index"] = mask_index

        return out_dict
    
    def __len__(self):
        return self.n_images



def generalized_dice_loss_ignore_background(probs, targets, epsilon=1e-6, ignore_index=0): 
    """
    probs: softmax probabilities, shape (B, C, D, H, W)
    targets: integer labels, shape (B, D, H, W)
    """
    num_classes = probs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0,4,1,2,3).float()

    dims = (0,2,3,4)

    # Exclude the background class
    valid_classes = [i for i in range(num_classes) if i != ignore_index]
    one_hot = one_hot[:, valid_classes, ...]
    probs = probs[:, valid_classes, ...]
    
    # Intersection and union
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)

    # Class weights (inverse squared of ground-truth volume)
    gt_sum = one_hot.sum(dims)
    weights = 1.0 / (gt_sum**2 + epsilon)

    numerator = 2 * (weights * intersection).sum()
    denominator = (weights * union).sum() + epsilon

    dice = numerator / denominator

    return 1 - dice



def generalized_dice_loss_from_logits(logits, targets, num_classes, epsilon=1e-6):
    """
    logits: (B, C, D, H, W)
    targets: (B, D, H, W) integer labels
    """
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3, 4)

    numerator = 0.0
    denominator = 0.0
    for c in range(num_classes):
        probs_c = probs[:, c, ...]
        target_c = (targets == c).float()
        intersection = (probs_c * target_c).sum(dims)
        union = probs_c.sum(dims) + target_c.sum(dims)

        # class weight (inverse squared volume)
        w_c = 1.0 / (target_c.sum(dims) ** 2 + epsilon)

        numerator += w_c * (2 * intersection)
        denominator += w_c * union

    dice = (numerator + epsilon) / (denominator + epsilon)
    return 1 - dice


def generalized_dice_loss(probs, targets, epsilon=1e-6):
    """
    probs: softmax probabilities, shape (B, C, D, H, W)
    targets: integer labels, shape (B, D, H, W)
    """
    num_classes = probs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)

    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)

    gt_sum = one_hot.sum(dims)
    weights = 1.0 / (gt_sum**2 + epsilon)

    # Mask out absent classes
    present_mask = gt_sum > 0
    weights = weights * present_mask

    numerator = 2 * (weights * intersection).sum()
    denominator = (weights * union).sum() + epsilon

    dice = numerator / denominator

    return 1 - dice

def vae_loss_segmentation(logits, targets, mu, sigma, num_classes, mask_part, beta=1e-7):

    # --- dice loss (generalized) ---
    probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1-1e-7)
    
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 to apply class weighting")
    if mask_part == "lung" and num_classes != 2:
        raise ValueError("For mask_part='lung', num_classes must be 2")
    if num_classes==2 and mask_part == "lung":
        class_weights = torch.tensor([0.5, 0.5], device=logits.device)  # Example weights for binary case
    elif mask_part == "nodule" and num_classes == 2:
        class_weights = torch.tensor([0.5, 10.0], device=logits.device)  # Example weights for binary case
    # Define class weights to handle class imbalance
    elif num_classes == 3:
        # Example: [background, nodule, lung] weights
        class_weights = torch.tensor([0.5, 10.0, 1.0], device=logits.device)
    elif num_classes == 7:
        # Example: [background, nodule texture 1, nodule texture 2, nodule texture 3, nodule texture 4, nodule texture 5, lung] weights
        class_weights = torch.tensor([0.5, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0], device=logits.device)
    else:
        raise ValueError(f"Class weights not defined for num_classes={num_classes}")
    
    # --- cross entropy ---
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    #ce = F.cross_entropy(logits, targets)

    dice = generalized_dice_loss(probs, targets)
    #dice = generalized_dice_loss_from_logits(logits, targets, num_classes)

    # --- reconstruction ---
    recon_loss = ce + dice

    # --- KL divergence ---
    # sigma = torch.clamp(sigma, min=1e-6)
    # kl = -0.5 * torch.mean(1 + 2*torch.log(sigma) - mu.pow(2) - sigma.pow(2))

    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(sigma.shape))),
    )
    kl= torch.sum(kl_loss) / kl_loss.shape[0]

    return recon_loss + beta * kl, recon_loss, kl, ce, dice

import torch
import torch.nn.functional as F

def per_class_dice(inputs, targets, epsilon=1e-6, is_logits=True):
    """
    Computes Dice score per class.
    If is_logits=True, applies argmax to get hard predictions.
    Absent classes in GT are set to NaN.
    """
    if is_logits:
        pred = torch.argmax(inputs, dim=1)  # (B, D, H, W)
        inputs = F.one_hot(pred, num_classes=inputs.shape[1]).permute(0, 4, 1, 2, 3).float()
    else:
        num_classes = inputs.max().item() + 1
        inputs = F.one_hot(inputs, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    num_classes = inputs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (inputs * one_hot).sum(dims)
    union = inputs.sum(dims) + one_hot.sum(dims)
    dice_per_class = (2 * intersection + epsilon) / (union + epsilon)

    # Identify classes absent in GT
    gt_sum = one_hot.sum(dims)
    absent_mask = gt_sum == 0

    # Assign NaN to absent classes
    dice_per_class = dice_per_class.masked_fill(absent_mask, float('nan'))

    print("Dice per class:", dice_per_class)

    return dice_per_class

def sdf_regression_loss_weighted(pred_sdf, target_sdf, weights=None):
    """
    pred_sdf, target_sdf: (B, C, D, H, W)
    weights: (C,) tensor or None
    """
    l1_per_channel = torch.mean(torch.abs(pred_sdf - target_sdf), dim=(0,2,3,4))  # mean over batch+voxels
    if weights is not None:
        weights = weights / (weights.sum() + 1e-8)
        loss = (l1_per_channel * weights).sum()
    else:
        loss = l1_per_channel.mean()
    return loss

def eikonal_loss(sdf, class_weights=None):
    """
    Computes the eikonal loss for multi-class SDFs in 3D.
    
    Args:
        sdf: torch.Tensor of shape [B, C, D, H, W], predicted SDF
        class_weights: optional torch.Tensor of shape [C] to weight each channel (Normally not used)
        
    Returns:
        eik_loss: scalar tensor
    """
    # Compute gradients along spatial dimensions
    # torch.gradient returns a tuple of gradients along each dimension
    dx, dy, dz = torch.gradient(sdf, dim=(2,3,4))  # shape [B, C, D, H, W]

    # Gradient magnitude
    grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)  # [B, C, D, H, W]

    # Squared deviation from 1
    loss_per_voxel = (grad_norm - 1)**2  # [B, C, D, H, W]

    # Apply optional class weights
    if class_weights is not None:
        # reshape to broadcast: [1, C, 1, 1, 1]
        weights = class_weights.view(1, -1, 1, 1, 1)
        loss_per_voxel = loss_per_voxel * weights

    # Final eikonal loss: mean over batch, channels, and spatial dims
    eik_loss = loss_per_voxel.mean()
    return eik_loss


def vae_loss_sdf(pred_sdf, target_sdf, mu, sigma, beta=1e-7, eikonal_weight=0.1):
    """
    pred_sdf: (B,C,D,H,W)
    target_sdf: (B,C,D,H,W)
    """
    # Reconstruction term
    class_weights = torch.tensor([10, 1], device=pred_sdf.device)
    sdf_l1 = sdf_regression_loss_weighted(pred_sdf, target_sdf, class_weights)

    # Optional: add Eikonal regularization
    eik_loss = eikonal_loss(pred_sdf) * eikonal_weight

    # KL divergence
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(sigma.shape))),
    )
    kl = torch.sum(kl_loss) / kl_loss.shape[0]

    total = sdf_l1 + eik_loss + beta * kl
    return total, sdf_l1, eik_loss, kl