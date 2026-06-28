# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utility functions and helpers."""

from diagnosis_model.grod.detector.utilities import box_ops
from diagnosis_model.grod.detector.utilities.distributed import (
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from diagnosis_model.grod.detector.utilities.logger import get_logger
from diagnosis_model.grod.detector.utilities.package import get_sha, get_version
from diagnosis_model.grod.detector.utilities.reproducibility import seed_all
from diagnosis_model.grod.detector.utilities.state_dict import clean_state_dict, strip_checkpoint
from diagnosis_model.grod.detector.utilities.tensors import (
    NestedTensor,
    collate_fn,
    nested_tensor_from_tensor_list,
)

__all__ = [
    # distributed
    "all_gather",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "reduce_dict",
    "save_on_master",
    # tensors
    "NestedTensor",
    "collate_fn",
    "nested_tensor_from_tensor_list",
    # box_ops (submodule)
    "box_ops",
    # logger
    "get_logger",
    # package
    "get_sha",
    "get_version",
    # reproducibility
    "seed_all",
    # state_dict
    "clean_state_dict",
    "strip_checkpoint",
]
