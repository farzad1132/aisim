# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the op-latency dataset pinned by the installed AISim wheel before an offline or timed benchmark.

Run with the interpreter of the environment that holds the revision under test, so the pin comes from that wheel.
Revisions that still package their operation tables have nothing to fetch.
"""

import importlib.util

if importlib.util.find_spec("aisim_core.data.default_systems") is None:
    print("op-latency data is packaged in this revision")
else:
    from aisim_core.data.default_systems import default_systems_path

    print(default_systems_path())
