# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiler used with ensemble of models."""

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor

from anomalib.data.utils.tiler import Tiler, compute_new_image_size, upscale_image

class EnsembleTiler(Tiler):
    """Tile Image into (non)overlapping Patches which are then used for ensemble training.

    Args:
        tile_size (int | Sequence): Tile dimension for each patch.
        stride (int | Sequence): Stride length between patches.
        image_size (int | Sequence): Size of input image that will be tiled.

    Examples:
        >>> import torch
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>>
        >>> # random images, shape:  [B, C, H, W]
        >>> images = torch.rand(32, 5, 512, 512)
        >>> # once tiled, the shape is [tile_count_H, tile_count_W, B, C, tile_H, tile_W]
        >>> tiled = tiler.tile(images)
        >>> tiled.shape
        torch.Size([3, 3, 32, 5, 256, 256])

        >>> # assemble the tiles back together
        >>> untiled = tiler.untile(tiled)
        >>> untiled.shape
        torch.Size([32, 5, 512, 512])
    """

    def __init__(self, tile_size: int | Sequence, stride: int | Sequence, image_size: int | Sequence) -> None:
        super().__init__(
            tile_size=tile_size,
            stride=stride,
        )

        # calculate final image size
        self.image_size = self.validate_size_type(image_size)
        self.input_h, self.input_w = self.image_size
        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        # get number of patches in both dimensions
        self.num_patches_h = int((self.resized_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((self.resized_w - self.tile_size_w) / self.stride_w) + 1
        self.num_tiles = self.num_patches_h * self.num_patches_w

    def tile(self, image: Tensor, use_random_tiling: bool = False) -> Tensor:
        """Tiles an input image to either overlapping or non-overlapping patches.

        Args:
            image (Tensor): Input images.
            use_random_tiling (bool): Random tiling, which is part of original tiler but is unused here.

        Returns:
            Tensor: Tiles generated from images.
                Returned shape: [num_h, num_w, batch, channel, tile_height, tile_width].
        """
        # tiles are returned in order [tile_count * batch, channels, tile_height, tile_width]
        combined_tiles = super().tile(image, use_random_tiling)

        # rearrange to [num_h, num_w, batch, channel, tile_height, tile_width]
        tiles = combined_tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.num_channels,
            self.tile_size_h,
            self.tile_size_w,
        )
        tiles = tiles.permute(1, 2, 0, 3, 4, 5)

        return tiles  # noqa: RET504

    def untile(self, tiles: Tensor) -> Tensor:
        """Reassemble the tiled tensor into image level representation.

        Args:
            tiles (Tensor): Tiles in shape: [num_h, num_w, batch, channel, tile_height, tile_width].

        Returns:
            Tensor: Image constructed from input tiles. Shape: [B, C, H, W].
        """
        # tiles have shape [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, batch, channels, tile_size_h, tile_size_w = tiles.shape

        # set tilers batch size as it might have been changed by previous tiling
        self.batch_size = batch

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = super().untile(tiles)

        return untiled  # noqa: RET504


class TileCollater:
    """Class serving as collate function to perform tiling on batch of images from Dataloader.

    Args:
        tiler (EnsembleTiler): Tiler used to split the images to tiles.
        tile_index (tuple[int, int]): Index of tile we want to return.
        default_collate_fn (callable): default collate function from dataset
    """

    def __init__(self, tiler: EnsembleTiler, tile_index: tuple[int, int], default_collate_fn: Callable) -> None:
        self.tiler = tiler
        self.tile_index = tile_index
        self.default_collate_fn = default_collate_fn

    def _crop_tile(self, tensor: "Tensor", tile_index: tuple[int, int]) -> "Tensor":
        """Resize tensor to tiler's target size and crop the single requested tile.

        Avoids allocating a tensor for all tiles (e.g. 16 × batch × 3 × 256 × 256
        for a 4096-wide image), since only one tile is needed per job.
        """
        t = self.tiler
        _, _, h, w = tensor.shape
        resized_h, resized_w = compute_new_image_size(
            image_size=(h, w),
            tile_size=(t.tile_size_h, t.tile_size_w),
            stride=(t.stride_h, t.stride_w),
        )
        resized = upscale_image(tensor, size=(resized_h, resized_w), mode=t.mode)
        h_idx, w_idx = tile_index
        loc_i = h_idx * t.stride_h
        loc_j = w_idx * t.stride_w
        return resized[:, :, loc_i : loc_i + t.tile_size_h, loc_j : loc_j + t.tile_size_w]

    def __call__(self, batch: list) -> dict[str, Any]:
        """Collate batch and tile images + masks from batch.

        Args:
            batch (list): Batch of elements from data, also including images.

        Returns:
            dict[str, Any]: Collated batch dictionary with tiled images.
        """
        coll_batch = self.default_collate_fn(batch)

        coll_batch.image = self._crop_tile(coll_batch.image, self.tile_index)

        if hasattr(coll_batch, "gt_mask") and coll_batch.gt_mask is not None:
            full_mask = coll_batch.gt_mask
            mask = full_mask.unsqueeze(1)
            coll_batch.gt_mask = self._crop_tile(mask, self.tile_index).squeeze(1)
            if getattr(coll_batch, "gt_label", None) is not None:
                coll_batch.gt_label = self._tile_labels(coll_batch.gt_label, full_mask, coll_batch.gt_mask)

        return coll_batch

    @staticmethod
    def _tile_labels(image_labels: "Tensor", full_mask: "Tensor", tile_mask: "Tensor") -> "Tensor":
        """The label of this tile of every image: anomalous only if the image's
        ground-truth mask marks something inside the tile. Without that, a tile
        showing nothing but good surface would count as anomalous whenever the
        defect is elsewhere in the image, and each tile model's validation (image
        AUROC, which EarlyStopping/ModelCheckpoint watch, and its thresholds)
        would score it against a label it cannot see. Images whose mask marks
        nothing (good images, or anomalous ones without a drawn mask) keep the
        image's label. AOIMergeJob (jobs.py) recovers the image label as "any
        tile anomalous"."""
        has_mask = full_mask.flatten(1).any(dim=1)
        in_tile = tile_mask.flatten(1).any(dim=1)
        labels = image_labels.bool().where(~has_mask, in_tile)
        return labels.to(image_labels.dtype)
