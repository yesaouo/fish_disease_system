"""Archived Mamba case-encoder ablation.

Production Phase 4 uses DeepSets (in ``..models.case_encoder``); Mamba is kept
here only as an architecture-ablation reference. Importing this subpackage
requires the ``mamba_ssm`` package, which on this host is installed in the
``mamba3`` conda env and needs ``CC=/usr/bin/gcc-12`` for triton kernel JIT.
"""
