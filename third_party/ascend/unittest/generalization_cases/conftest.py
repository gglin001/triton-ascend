# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch

import numbers
import os
import re


_DTYPE_ALIAS = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
}
_DTYPE_PATTERN = re.compile(r"^(u?int\d+|float\d+|bfloat16|bf16|fp\d+|bool)([a-z0-9_]*)$")


def _normalize_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        if value == torch.bool:
            return "bool"
        if value == torch.int8:
            return "int8"
        if value == torch.int16:
            return "int16"
        if value == torch.int32:
            return "int32"
        if value == torch.int64:
            return "int64"
        if value == torch.uint8:
            return "uint8"
        if hasattr(torch, "uint16") and value == torch.uint16:
            return "uint16"
        if hasattr(torch, "uint32") and value == torch.uint32:
            return "uint32"
        if hasattr(torch, "uint64") and value == torch.uint64:
            return "uint64"
        if value == torch.float16:
            return "float16"
        if value == torch.float32:
            return "float32"
        if hasattr(torch, "float64") and value == torch.float64:
            return "float64"
        if value == torch.bfloat16:
            return "bfloat16"
        return None
    if isinstance(value, str):
        norm = _DTYPE_ALIAS.get(value.strip().lower(), value.strip().lower())
        return norm if _DTYPE_PATTERN.match(norm) else None
    if hasattr(value, "name") and isinstance(value.name, str):
        norm = _DTYPE_ALIAS.get(value.name.strip().lower(), value.name.strip().lower())
        return norm if _DTYPE_PATTERN.match(norm) else None
    return None


def _normalize_shape(value):
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(x, numbers.Integral) for x in value):
            return tuple(int(x) for x in value)
    return None


_DTYPE_PARAM_NAMES = {
    "sigtype",
    "data_type",
    "para_type",
    "normal_type",
}
SKIP_TEST_FILES = {
    "test_cumprod.py",
    "test_cumsum.py",
    "test_associative_scan.py",
    "test_atan.py",
    "test_log1p.pytest_relu.py",
    "test_tan.py",
    "test_log1p.py",
    "test_relu.py",
    "test_general_log.py",
    "test_general_log2.py",
}
SUPPORTED_DTYPES = {"float32"}
SUPPORTED_SHAPES = {
    # 1d
    (1,),
    # (2,),
    (8,),
    # (64,),
    # (256,),
    # 2d
    # (1, 16),
    (8, 32),
    # 3d
    (4, 8, 256),
    # 4d
    (8, 4, 8, 8),
    # 5d
    (2, 8, 4, 8, 8),
}


def _is_dtype_param(name):
    return "dtype" in name or name in _DTYPE_PARAM_NAMES


def _is_shape_param(name):
    return "shape" in name or name == "shaape"


def _extract_dtypes(value):
    dtype = _normalize_dtype(value)
    if dtype:
        return [dtype]
    dtypes = []
    if isinstance(value, (list, tuple, set)):
        for elem in value:
            dtype = _normalize_dtype(elem)
            if dtype:
                dtypes.append(dtype)
    return dtypes


def _extract_shapes(value):
    shape = _normalize_shape(value)
    if shape is not None:
        return [shape]
    shapes = []
    if isinstance(value, (list, tuple, set)):
        for elem in value:
            shape = _normalize_shape(elem)
            if shape is not None:
                shapes.append(shape)
    return shapes


def _collect_item_filters(item: pytest.Item):
    dtypes = []
    shapes = []
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return dtypes, shapes
    for name, value in callspec.params.items():
        lname = name.lower()
        if _is_dtype_param(lname):
            dtypes.extend(_extract_dtypes(value))
        elif _is_shape_param(lname):
            shapes.extend(_extract_shapes(value))
        elif lname == "param_list":
            dtypes.extend(_extract_dtypes(value))
            shapes.extend(_extract_shapes(value))
    return dtypes, shapes


def pytest_collection_modifyitems(config, items: list[pytest.Item]):
    if not SUPPORTED_DTYPES and not SUPPORTED_SHAPES and not SKIP_TEST_FILES:
        return
    for item in items:
        if SKIP_TEST_FILES:
            filename = os.path.basename(str(item.fspath))
            if filename in SKIP_TEST_FILES:
                item.add_marker(pytest.mark.skip(reason=f"skipped by file filter: {filename}"))
                continue
        dtypes, shapes = _collect_item_filters(item)
        reasons = []
        if SUPPORTED_DTYPES and dtypes:
            unsupported = sorted({d for d in dtypes if d not in SUPPORTED_DTYPES})
            if unsupported:
                reasons.append(f"unsupported dtype(s): {', '.join(unsupported)}")
        if SUPPORTED_SHAPES and shapes:
            unsupported_shapes = [s for s in shapes if s not in SUPPORTED_SHAPES]
            if unsupported_shapes:
                reasons.append(f"unsupported shape(s): {', '.join(str(s) for s in unsupported_shapes)}")
        if reasons:
            # print(f"skip item: {item}")
            item.add_marker(pytest.mark.skip(reason="; ".join(reasons)))


def pytest_ignore_collect(collection_path, config):
    if not SKIP_TEST_FILES:
        return False
    filename = os.path.basename(str(collection_path))
    return filename in SKIP_TEST_FILES


def pytest_collection_finish(session: pytest.Session):
    print("\n=== Final Selected Tests (Will Run) ===")
    idx = 0
    for item in session.items:
        if item.get_closest_marker("skip") is None:
            print(item.nodeid)
            idx += 1
    print(f"len: {idx}")
    print("=======================================\n")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    import torch.cpu

    torch.cpu.set_device(0)


@pytest.fixture(scope="session", autouse=False)
def assign_npu(worker_id):
    npu_count = torch.npu.device_count()
    if worker_id == "master":
        npu_id = 0
    else:
        idx = int(worker_id.replace("gw", ""))
        npu_id = idx % npu_count
    torch.npu.set_device(npu_id)
