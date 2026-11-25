# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenization-cost collection: time a serving framework's own request-processing code on a host CPU.

The harness needs no GPU kernels. It runs inside the pinned framework image, on the host CPU that
serves requests, and times each step of turning a chat request body into engine input.
"""
