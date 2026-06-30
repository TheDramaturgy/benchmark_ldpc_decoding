#!/usr/bin/env python3
"""
Aerial LDPC benchmark for Agora/FlexRAN exported datasets.

Supports both dataset styles:

  Old / size-sweep:
    metadata.json
    header.bin              optional
    llrs_i8.bin
    ref_msg_u8.bin

  New direct-MCS-mother-rate:
    metadata.json
    llrs_i8.bin
    ref_msg_u8.bin
    encoded_codewords_u8.bin optional
    flexran_decoded_u8.bin   optional

Timing modes:

  gpu_decode_only:
    Prebuild grouped Aerial inputs on the GPU before timing.
    Timed section: decoder.decode(...) + CUDA stream synchronization.
    This avoids host->device bus transfer of input LLRs in the timed section.

  gpu_plus_bus:
    Prebuild grouped Aerial inputs on the CPU before timing.
    Timed section: Aerial wrapper CPU->GPU copy/conversion + decoder.decode(...)
                   + synchronization. Depending on Aerial version, CPU outputs may
                   also include GPU->CPU return-copy cost.

  both:
    Run both modes and put both rows in the CSV.

Grouped / decoder API modes:

  --decoder-api explicit (default) uses the new explicit codeblock-batch LDPC
  decoder API. It packs C exported FlexRAN codeblocks into one input matrix
  of shape [N, C] and passes BG/Zc/nRows/num_codeblocks directly. This is
  the correct mode for reducing kernel instances without fake TB segmentation.

  --decoder-api legacy keeps the old Aerial TB-level decoder.decode(...) path.
  Use it only for comparison or for the old one-CB-per-TB safe path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def find_run_dirs(path: Path) -> List[Path]:
    path = path.resolve()

    def is_run_dir(p: Path) -> bool:
        return (
            (p / "metadata.json").exists()
            and (p / "llrs_i8.bin").exists()
            and (p / "ref_msg_u8.bin").exists()
        )

    if is_run_dir(path):
        return [path]

    metas = sorted(path.rglob("metadata.json"))
    run_dirs = [p.parent for p in metas if is_run_dir(p.parent)]

    if not run_dirs:
        die(
            f"No valid run directories found under {path}. "
            "A run directory must contain metadata.json, llrs_i8.bin, and ref_msg_u8.bin."
        )

    return run_dirs


def get_any(meta: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in meta:
            return meta[key]
    return default


def get_int(meta: Dict[str, Any], keys: Sequence[str], default: Optional[int] = None) -> int:
    value = get_any(meta, keys, default)
    if value is None:
        raise KeyError(f"Missing metadata key. Tried: {keys}")
    return int(value)


def get_float(meta: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> float:
    value = get_any(meta, keys, default)
    if value is None:
        raise KeyError(f"Missing metadata key. Tried: {keys}")
    return float(value)


def bytes_to_bits_lsb(byte_array: np.ndarray, nbits: int) -> np.ndarray:
    bits = np.unpackbits(byte_array, axis=1, bitorder="little")
    return bits[:, :nbits].astype(np.uint8, copy=False)


def compute_ber_bler(decoded_bits: np.ndarray, ref_bits: np.ndarray) -> Dict[str, Any]:
    if decoded_bits.shape != ref_bits.shape:
        raise ValueError(
            f"decoded_bits shape {decoded_bits.shape} != ref_bits shape {ref_bits.shape}"
        )

    err = decoded_bits != ref_bits

    bit_errors = int(np.count_nonzero(err))
    total_bits = int(ref_bits.size)
    block_errors = int(np.count_nonzero(np.any(err, axis=1)))
    total_blocks = int(ref_bits.shape[0])

    return {
        "bit_errors": bit_errors,
        "total_bits": total_bits,
        "ber": bit_errors / total_bits if total_bits else float("nan"),
        "block_errors": block_errors,
        "total_blocks": total_blocks,
        "bler": block_errors / total_blocks if total_blocks else float("nan"),
    }


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return x.get()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def normalize_metadata(meta: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Normalize old/new exporter metadata into one schema used by this script."""
    out = dict(meta)

    out["num_codeblocks"] = get_int(
        out, ["num_codeblocks", "num_samples"], None
    )
    out["llr_len"] = get_int(
        out, ["llr_len", "num_channel_llrs"], None
    )
    out["msg_bits_per_cb"] = get_int(
        out, ["msg_bits_per_cb", "msg_bits", "num_msg_bits"], None
    )
    out["msg_bytes_per_cb"] = get_int(
        out,
        ["msg_bytes_per_cb", "msg_bytes"],
        (int(out["msg_bits_per_cb"]) + 7) // 8,
    )

    out["base_graph"] = get_int(
        out, ["base_graph", "generated_base_graph"], None
    )
    out["zc"] = get_int(
        out, ["zc", "generated_zc"], None
    )
    out["n_rows"] = get_int(
        out, ["n_rows", "num_rows"], None
    )

    out["mcs_index"] = get_int(
        out, ["mcs_index"], parse_index_from_path(run_dir, "mcs", 0)
    )
    out["snr_index"] = get_int(
        out, ["snr_index"], parse_index_from_path(run_dir, "snr", 0)
    )
    out["snr_db"] = get_any(out, ["snr_db"], "")
    out["noise_sigma"] = get_any(out, ["noise_sigma", "sigma"], "")

    if "mcs_code_rate_x1024" not in out:
        r = get_any(out, ["mcs_code_rate", "mother_code_rate", "effective_cb_code_rate"], None)
        if r is not None:
            out["mcs_code_rate_x1024"] = int(round(float(r) * 1024.0))
        else:
            out["mcs_code_rate_x1024"] = ""

    if "mcs_qm" not in out:
        out["mcs_qm"] = get_any(out, ["mod_order_bits", "qm"], "")

    # For the new direct-MCS-mother-rate dataset, rate matching is disabled.
    # In that case, the Aerial rate_match_len should be the same as the
    # FlexRAN decoder input length excluding the first 2*Zc punctured positions.
    if "rate_match_len" not in out:
        out["rate_match_len"] = int(out["llr_len"])

    if "rv" not in out:
        out["rv"] = 0

    if "aerial_prepend_punctured_llrs" not in out:
        out["aerial_prepend_punctured_llrs"] = 2 * int(out["zc"])

    if "aerial_input_len" not in out:
        out["aerial_input_len"] = (
            int(out["aerial_prepend_punctured_llrs"]) + int(out["llr_len"])
        )

    if "aerial_tb_size" not in out:
        # For a single-codeblock synthetic TB, Aerial expects TB size without
        # the TB CRC. The exporter's msg_bits_per_cb corresponds to K.
        out["aerial_tb_size"] = int(out["msg_bits_per_cb"]) - 24

    if "aerial_code_rate" not in out:
        # Do not use true low/medium MCS rates here. Aerial's Python API infers
        # BG from tb_size and code_rate. For BG1 datasets, 0.68 avoids accidental
        # BG2 selection for small TBs.
        out["aerial_code_rate"] = 0.68 if int(out["base_graph"]) == 1 else 0.50

    if "max_iterations" not in out:
        out["max_iterations"] = int(get_any(out, ["max_iter"], 5))

    return out


def parse_index_from_path(path: Path, prefix: str, default: int = 0) -> int:
    import re

    pattern = re.compile(rf"^{prefix}_(\d+)$")
    for part in path.parts:
        m = pattern.match(part)
        if m:
            return int(m.group(1))
    return default


def load_exported_dataset(
    run_dir: Path, limit_codeblocks: Optional[int]
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    meta = normalize_metadata(load_json(run_dir / "metadata.json"), run_dir)

    num_cb_total = int(meta["num_codeblocks"])
    num_cb = num_cb_total if limit_codeblocks is None else min(num_cb_total, int(limit_codeblocks))

    llr_len = int(meta["llr_len"])
    msg_bytes = int(meta["msg_bytes_per_cb"])

    llr_file = meta.get("llr_file", "llrs_i8.bin")
    ref_file = meta.get("ref_file", "ref_msg_u8.bin")

    llrs = np.fromfile(run_dir / llr_file, dtype=np.int8)
    expected_llrs = num_cb_total * llr_len
    if llrs.size != expected_llrs:
        die(f"{run_dir / llr_file} has {llrs.size} int8 values, expected {expected_llrs}")
    llrs = llrs.reshape(num_cb_total, llr_len)[:num_cb]

    ref_msg = np.fromfile(run_dir / ref_file, dtype=np.uint8)
    expected_ref = num_cb_total * msg_bytes
    if ref_msg.size != expected_ref:
        die(f"{run_dir / ref_file} has {ref_msg.size} uint8 values, expected {expected_ref}")
    ref_msg = ref_msg.reshape(num_cb_total, msg_bytes)[:num_cb]

    meta["num_codeblocks_loaded"] = num_cb
    return meta, llrs, ref_msg


# ---------------------------------------------------------------------------
# Aerial setup and GPU synchronization
# ---------------------------------------------------------------------------

def setup_aerial_decoder(gpu: int, throughput_mode: bool, initial_iterations: Optional[int]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Keep cudart only for cudaSetDevice and as a fallback synchronizer.
    # Newer Aerial versions expect cuda_stream to be an aerial CudaStream object,
    # not a raw cudaStream_t handle, so let LdpcDecoder create its own stream.
    import cuda.bindings.runtime as cudart
    from aerial.phy5g.ldpc import LdpcDecoder

    cudart.cudaSetDevice(0)

    kwargs = {"throughput_mode": bool(throughput_mode)}
    if initial_iterations is not None and initial_iterations > 0:
        kwargs["num_iterations"] = int(initial_iterations)

    decoder = LdpcDecoder(**kwargs)

    # New wrapper stores a CudaStream object in _cuda_stream. Older wrapper may
    # expose cuda_stream as a raw handle. synchronize() below supports both.
    stream = getattr(decoder, "_cuda_stream", None)
    if stream is None:
        stream = getattr(decoder, "cuda_stream", None)
    synchronize(cudart, stream)
    return decoder, stream, cudart


def synchronize(cudart: Any, stream: Any) -> None:
    if stream is None:
        return
    if hasattr(stream, "synchronize"):
        stream.synchronize()
    else:
        cudart.cudaStreamSynchronize(stream)


# ---------------------------------------------------------------------------
# Grouping and Aerial parameter construction
# ---------------------------------------------------------------------------

@dataclass
class DecodeGroup:
    start: int
    end: int
    input_llr: Optional[Any]
    tb_size: int
    code_rate: float
    rate_match_len: int
    rv: int

    @property
    def c(self) -> int:
        return self.end - self.start


def effective_k_per_cb(meta: Dict[str, Any]) -> int:
    return int(meta["msg_bits_per_cb"])


def grouped_params(meta: Dict[str, Any], c: int, args: argparse.Namespace) -> Tuple[int, float, int, int]:
    """Return tb_size, code_rate, rate_match_len, rv for one grouped synthetic TB."""
    k_per_cb = effective_k_per_cb(meta)
    rm_per_cb = int(meta.get("rate_match_len", meta["llr_len"]))
    bg = int(meta["base_graph"])

    if args.code_rate is not None:
        code_rate = float(args.code_rate)
    elif "aerial_code_rate" in meta:
        code_rate = float(meta["aerial_code_rate"])
    elif bg == 1:
        code_rate = 0.68
    else:
        code_rate = 0.50

    rv = int(args.rv) if args.rv is not None else int(meta.get("rv", 0))

    if c == 1:
        tb_size = int(args.tb_size) if args.tb_size is not None else int(meta.get("aerial_tb_size", k_per_cb - 24))
        rate_match_len = int(args.rate_match_len) if args.rate_match_len is not None else rm_per_cb
    else:
        if args.tb_size is not None:
            raise ValueError("--tb-size override is only supported with --group-cbs 1")
        if args.rate_match_len is not None:
            raise ValueError("--rate-match-len override is only supported with --group-cbs 1")

        # Synthetic multi-CB TB relation:
        #
        #   TB payload + TB CRC(24) + C * CB CRC(24) = C * K
        #
        # For C=1, there is usually no CB CRC in the Aerial segmentation path, so
        # the single-CB case above uses K - 24. For grouped C>1, use the relation
        # used by the previous grouped script.
        tb_size = c * k_per_cb - 24 * c - 24
        rate_match_len = c * rm_per_cb

    return tb_size, code_rate, rate_match_len, rv


def get_num_iterations(meta: Dict[str, Any], args: argparse.Namespace) -> Optional[int]:
    if args.num_iterations is not None:
        return int(args.num_iterations)
    value = int(meta.get("max_iterations", 0))
    return value if value > 0 else None


def make_grouped_input_array(
    llrs_slice: np.ndarray,
    meta: Dict[str, Any],
    llr_sign: float,
    llr_scale: float,
    input_location: str,
    cpu_input_dtype: str,
) -> Any:
    """Build one Aerial input matrix of shape [N, C]."""
    llr_len = int(meta["llr_len"])
    punctured = int(meta.get("aerial_prepend_punctured_llrs", 2 * int(meta["zc"])))
    aerial_input_len = int(meta.get("aerial_input_len", punctured + llr_len))
    c = int(llrs_slice.shape[0])

    if c <= 0:
        raise ValueError("empty grouped input")

    cpu_dtype = np.float16 if cpu_input_dtype == "float16" else np.float32

    x = np.zeros((aerial_input_len, c), dtype=cpu_dtype, order="F")
    x[punctured : punctured + llr_len, :] = (
        llrs_slice.astype(cpu_dtype, copy=False).T * float(llr_sign) * float(llr_scale)
    )
    x = np.asfortranarray(x)

    if input_location == "gpu":
        import cupy as cp  # type: ignore
        return cp.asarray(x, dtype=cp.float16, order="F")

    if input_location == "cpu":
        return x

    raise ValueError(f"Unknown input_location={input_location}")


def build_decode_groups(
    llrs_i8: np.ndarray,
    meta: Dict[str, Any],
    group_cbs: int,
    args: argparse.Namespace,
    input_location: str,
    prebuild_inputs: bool,
) -> List[DecodeGroup]:
    groups: List[DecodeGroup] = []

    if args.grouping_mode == "explicit_codeblocks":
        # Correct batched codeblock mode for the new decoder API:
        # C exported FlexRAN codeblocks are columns of one Aerial input [N,C].
        # No synthetic TB size is used by the explicit decoder; BG/Zc/nRows/C
        # are passed directly to decode_codeblocks_explicit().
        for start in range(0, llrs_i8.shape[0], group_cbs):
            end = min(start + group_cbs, llrs_i8.shape[0])
            c = end - start

            # These are informational only for explicit mode / CSV; the new API
            # does not use tb_size, code_rate, rate_match_len, or rv.
            rm_per_cb = int(meta.get("rate_match_len", meta["llr_len"]))
            rv = int(args.rv) if args.rv is not None else int(meta.get("rv", 0))

            input_llr = None
            if prebuild_inputs:
                input_llr = make_grouped_input_array(
                    llrs_i8[start:end],
                    meta,
                    llr_sign=args.llr_sign,
                    llr_scale=args.llr_scale,
                    input_location=input_location,
                    cpu_input_dtype=args.cpu_input_dtype,
                )

            groups.append(
                DecodeGroup(
                    start=start,
                    end=end,
                    input_llr=input_llr,
                    tb_size=0,
                    code_rate=0.0,
                    rate_match_len=c * rm_per_cb,
                    rv=rv,
                )
            )

        return groups

    if args.grouping_mode == "independent_tbs":
        # Safe legacy mode:
        # each exported FlexRAN codeblock becomes one independent Aerial TB
        # with input shape [N, 1]. This preserves the exported BG/Zc/K but
        # launches many kernels through the old TB-level API.
        for cb in range(llrs_i8.shape[0]):
            tb_size, code_rate, rate_match_len, rv = grouped_params(meta, 1, args)

            input_llr = None
            if prebuild_inputs:
                input_llr = make_grouped_input_array(
                    llrs_i8[cb : cb + 1],
                    meta,
                    llr_sign=args.llr_sign,
                    llr_scale=args.llr_scale,
                    input_location=input_location,
                    cpu_input_dtype=args.cpu_input_dtype,
                )

            groups.append(
                DecodeGroup(
                    start=cb,
                    end=cb + 1,
                    input_llr=input_llr,
                    tb_size=tb_size,
                    code_rate=code_rate,
                    rate_match_len=rate_match_len,
                    rv=rv,
                )
            )

        return groups

    if args.grouping_mode == "column_grouping":
        # Unsafe legacy experiment:
        # packs C exported codeblocks as columns of one Aerial TB input [N,C]
        # but still uses decoder.decode(...), forcing Aerial to infer TB
        # segmentation from fake tb_size/code_rate. This produced BER≈0.5 in
        # your tests and is kept only for comparison.
        for start in range(0, llrs_i8.shape[0], group_cbs):
            end = min(start + group_cbs, llrs_i8.shape[0])
            c = end - start

            tb_size, code_rate, rate_match_len, rv = grouped_params(meta, c, args)

            input_llr = None
            if prebuild_inputs:
                input_llr = make_grouped_input_array(
                    llrs_i8[start:end],
                    meta,
                    llr_sign=args.llr_sign,
                    llr_scale=args.llr_scale,
                    input_location=input_location,
                    cpu_input_dtype=args.cpu_input_dtype,
                )

            groups.append(
                DecodeGroup(
                    start=start,
                    end=end,
                    input_llr=input_llr,
                    tb_size=tb_size,
                    code_rate=code_rate,
                    rate_match_len=rate_match_len,
                    rv=rv,
                )
            )

        return groups

    raise ValueError(f"Unsupported grouping_mode={args.grouping_mode}")

def get_group_input(
    g: DecodeGroup,
    llrs_i8: np.ndarray,
    meta: Dict[str, Any],
    args: argparse.Namespace,
    input_location: str,
) -> Any:
    if g.input_llr is not None:
        return g.input_llr

    return make_grouped_input_array(
        llrs_i8[g.start : g.end],
        meta,
        llr_sign=args.llr_sign,
        llr_scale=args.llr_scale,
        input_location=input_location,
        cpu_input_dtype=args.cpu_input_dtype,
    )


def call_aerial_decode_legacy(
    decoder: Any,
    input_llrs: List[Any],
    groups: Sequence[DecodeGroup],
    num_iterations: Optional[int],
):
    """Old TB-level Aerial decoder path."""
    if len(input_llrs) != len(groups):
        raise ValueError(f"len(input_llrs)={len(input_llrs)} != len(groups)={len(groups)}")

    kwargs = dict(
        input_llrs=input_llrs,
        tb_sizes=[int(g.tb_size) for g in groups],
        code_rates=[float(g.code_rate) for g in groups],
        redundancy_versions=[int(g.rv) for g in groups],
        rate_match_lengths=[int(g.rate_match_len) for g in groups],
    )

    if num_iterations is not None and num_iterations > 0:
        kwargs["num_iterations"] = int(num_iterations)

    try:
        return decoder.decode(**kwargs)
    except TypeError:
        kwargs.pop("num_iterations", None)
        return decoder.decode(**kwargs)


def _call_explicit_method_candidates(
    method: Any,
    input_llr: Any,
    base_graph: int,
    zc: int,
    n_rows: int,
    num_codeblocks: int,
    num_iterations: Optional[int],
) -> Any:
    """Try common signatures for the Python explicit wrapper method."""
    errors: List[str] = []

    kw = dict(
        input_llrs=[input_llr],
        base_graph=int(base_graph),
        zc=int(zc),
        n_rows=int(n_rows),
        num_codeblocks=int(num_codeblocks),
    )
    if num_iterations is not None and num_iterations > 0:
        kw_with_iter = dict(kw)
        kw_with_iter["num_iterations"] = int(num_iterations)
        try:
            return method(**kw_with_iter)
        except TypeError as exc:
            errors.append(f"keyword+num_iterations: {exc}")

    try:
        return method(**kw)
    except TypeError as exc:
        errors.append(f"keyword: {exc}")

    # Positional fallbacks, useful if the Python binding exposed the C++ method
    # directly rather than with keyword-only Python glue.
    if num_iterations is not None and num_iterations > 0:
        try:
            return method([input_llr], int(base_graph), int(zc), int(n_rows), int(num_codeblocks), int(num_iterations))
        except TypeError as exc:
            errors.append(f"positional+num_iterations: {exc}")

    try:
        return method([input_llr], int(base_graph), int(zc), int(n_rows), int(num_codeblocks))
    except TypeError as exc:
        errors.append(f"positional: {exc}")

    raise TypeError("Could not call explicit decode method. Tried signatures:\n  " + "\n  ".join(errors))


def _call_direct_pycuphy_explicit(
    decoder: Any,
    method_name: str,
    input_llr: Any,
    base_graph: int,
    zc: int,
    n_rows: int,
    num_codeblocks: int,
    num_iterations: Optional[int],
) -> Any:
    """Fallback if only decoder.pycuphy_ldpc_decoder exposes the new method.

    This mirrors Aerial's Python wrapper behavior: CPU NumPy input is copied to
    CuPy/FP16, CuPy input stays on GPU, then CudaArrayHalf is passed to pycuphy.
    """
    low = getattr(decoder, "pycuphy_ldpc_decoder", None)
    if low is None or not hasattr(low, method_name):
        raise AttributeError(f"Neither LdpcDecoder nor pycuphy_ldpc_decoder exposes {method_name}()")

    import cupy as cp  # type: ignore

    # Reuse pycuphy object already imported by aerial.phy5g.ldpc.decoder.py.
    pycuphy = None
    try:
        pycuphy = decoder.__class__.decode.__globals__.get("pycuphy")
    except Exception:
        pycuphy = None
    if pycuphy is None:
        try:
            from aerial.phy5g.ldpc import pycuphy as pycuphy  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import/find pycuphy for direct explicit fallback. "
                "Prefer exposing decode_codeblocks_explicit() on LdpcDecoder."
            ) from exc

    cpu_copy = isinstance(input_llr, np.ndarray)
    if num_iterations is not None and num_iterations > 0 and hasattr(low, "set_num_iterations"):
        low.set_num_iterations(int(num_iterations))
        if hasattr(decoder, "num_iterations"):
            decoder.num_iterations = int(num_iterations)

    stream_obj = getattr(decoder, "_cuda_stream", None)
    raw_stream = getattr(decoder, "cuda_stream", None)

    if stream_obj is not None:
        with stream_obj:
            cp_llr = cp.asarray(input_llr, dtype=cp.float16, order="F")
            wrapped = [pycuphy.CudaArrayHalf(cp_llr)]
    elif raw_stream is not None:
        with cp.cuda.ExternalStream(int(raw_stream)):
            cp_llr = cp.asarray(input_llr, dtype=cp.float16, order="F")
            wrapped = [pycuphy.CudaArrayHalf(cp_llr)]
    else:
        cp_llr = cp.asarray(input_llr, dtype=cp.float16, order="F")
        wrapped = [pycuphy.CudaArrayHalf(cp_llr)]

    method = getattr(low, method_name)
    out = _call_explicit_method_candidates(
        method,
        wrapped[0],
        base_graph,
        zc,
        n_rows,
        num_codeblocks,
        None,  # iteration already set above for low-level object
    )

    if not isinstance(out, (list, tuple)):
        out = [out]

    if stream_obj is not None:
        with stream_obj:
            out = [cp.array(elem) for elem in out]
            if cpu_copy:
                out = [elem.get(order="F") for elem in out]
    elif raw_stream is not None:
        with cp.cuda.ExternalStream(int(raw_stream)):
            out = [cp.array(elem) for elem in out]
            if cpu_copy:
                out = [elem.get(order="F") for elem in out]
    else:
        out = [cp.array(elem) for elem in out]
        if cpu_copy:
            out = [elem.get(order="F") for elem in out]
    return out


def call_aerial_decode_explicit(
    decoder: Any,
    input_llrs: List[Any],
    groups: Sequence[DecodeGroup],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    num_iterations: Optional[int],
):
    """New explicit codeblock-batch path.

    Each DecodeGroup is one [N,C] matrix decoded with explicit BG/Zc/nRows/C.
    This avoids Aerial's TB-size segmentation inference.
    """
    if len(input_llrs) != len(groups):
        raise ValueError(f"len(input_llrs)={len(input_llrs)} != len(groups)={len(groups)}")

    base_graph = int(meta["base_graph"])
    zc = int(meta["zc"])
    n_rows = int(meta["n_rows"])
    method_name = str(args.explicit_method_name)

    outputs: List[Any] = []
    wrapper_method = getattr(decoder, method_name, None)

    for input_llr, g in zip(input_llrs, groups):
        shape = getattr(input_llr, "shape", None)
        if shape is not None:
            expected_n = int(meta["aerial_input_len"])
            if int(shape[0]) != expected_n or int(shape[1]) != g.c:
                raise ValueError(
                    f"explicit input shape {shape} does not match expected "
                    f"({expected_n}, {g.c}) for group {g.start}:{g.end}"
                )

        if wrapper_method is not None:
            decoded = _call_explicit_method_candidates(
                wrapper_method,
                input_llr,
                base_graph,
                zc,
                n_rows,
                g.c,
                num_iterations,
            )
        else:
            decoded = _call_direct_pycuphy_explicit(
                decoder,
                method_name,
                input_llr,
                base_graph,
                zc,
                n_rows,
                g.c,
                num_iterations,
            )

        if isinstance(decoded, (list, tuple)):
            if len(decoded) != 1:
                raise ValueError(
                    f"explicit decode for one group returned {len(decoded)} outputs; expected 1"
                )
            outputs.append(decoded[0])
        else:
            outputs.append(decoded)

    return outputs


def call_aerial_decode(
    decoder: Any,
    input_llrs: List[Any],
    groups: Sequence[DecodeGroup],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    num_iterations: Optional[int],
):
    if args.decoder_api == "legacy":
        return call_aerial_decode_legacy(decoder, input_llrs, groups, num_iterations)

    if args.decoder_api == "explicit":
        return call_aerial_decode_explicit(decoder, input_llrs, groups, meta, args, num_iterations)

    if args.decoder_api == "auto":
        if args.grouping_mode == "explicit_codeblocks":
            return call_aerial_decode_explicit(decoder, input_llrs, groups, meta, args, num_iterations)
        return call_aerial_decode_legacy(decoder, input_llrs, groups, num_iterations)

    raise ValueError(f"Unsupported decoder_api={args.decoder_api}")


# ---------------------------------------------------------------------------
# Decoder output normalization and verification
# ---------------------------------------------------------------------------

def normalize_grouped_decoder_output(decoded_item: Any, c: int, msg_bits: int, decoded_format: str) -> np.ndarray:
    """Normalize decoder output from one grouped synthetic TB to [C, msg_bits]."""
    arr = np.asarray(to_numpy(decoded_item))
    arr = np.squeeze(arr)

    # Common Aerial output for one TB with C CBs: [K, C]
    if arr.ndim == 2:
        if arr.shape[0] >= msg_bits and arr.shape[1] == c:
            return arr[:msg_bits, :c].T.astype(np.uint8, copy=False)
        if arr.shape[0] == c and arr.shape[1] >= msg_bits:
            return arr[:c, :msg_bits].astype(np.uint8, copy=False)
        if arr.shape[1] == c and arr.shape[0] >= msg_bits:
            return arr[:msg_bits, :c].T.astype(np.uint8, copy=False)

    flat = arr.reshape(-1)

    # Fallback for bit-flat output.
    if decoded_format in ("auto", "bits") and flat.size >= c * msg_bits:
        candidate = flat[: c * msg_bits]
        if decoded_format == "auto":
            if candidate.size and (candidate.min() < 0 or candidate.max() > 1):
                candidate = (candidate > 0).astype(np.uint8)
        return candidate.reshape(c, msg_bits).astype(np.uint8, copy=False)

    # Fallback for byte-flat output.
    msg_bytes = (msg_bits + 7) // 8
    if decoded_format in ("auto", "bytes") and flat.size >= c * msg_bytes:
        byte_arr = flat[: c * msg_bytes].astype(np.uint8, copy=False).reshape(c, msg_bytes)
        return bytes_to_bits_lsb(byte_arr, msg_bits)

    raise ValueError(f"Unsupported grouped decoder output shape {arr.shape}; c={c}, msg_bits={msg_bits}")


def run_verify_pass(
    decoder: Any,
    stream: Any,
    cudart: Any,
    groups: Sequence[DecodeGroup],
    input_llrs: List[Any],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    num_iterations: Optional[int],
    msg_bits: int,
    decoded_format: str,
) -> np.ndarray:
    decoded = call_aerial_decode(decoder, input_llrs, groups, meta, args, num_iterations)
    synchronize(cudart, stream)

    if not isinstance(decoded, (list, tuple)):
        decoded = [decoded]

    if len(decoded) != len(groups):
        raise ValueError(
            f"Verification expected {len(groups)} decoded output item(s), got {len(decoded)}"
        )

    decoded_bits_all: List[np.ndarray] = []
    for decoded_item, g in zip(decoded, groups):
        decoded_bits_all.append(
            normalize_grouped_decoder_output(decoded_item, g.c, msg_bits, decoded_format)
        )

    return np.concatenate(decoded_bits_all, axis=0)


# ---------------------------------------------------------------------------
# Timing modes
# ---------------------------------------------------------------------------

@dataclass
class TimingModeResult:
    mode: str
    input_location: str
    prebuild_inputs: bool
    pass_ms_values: np.ndarray
    host_call_ms_values: np.ndarray
    result: Dict[str, Any]


def prepare_input_list(
    groups: Sequence[DecodeGroup],
    llrs_i8: np.ndarray,
    meta: Dict[str, Any],
    args: argparse.Namespace,
    input_location: str,
) -> List[Any]:
    return [
        get_group_input(g, llrs_i8, meta, args, input_location=input_location)
        for g in groups
    ]


def run_one_timing_mode(
    mode: str,
    decoder: Any,
    stream: Any,
    cudart: Any,
    llrs_i8: np.ndarray,
    ref_bits: np.ndarray,
    meta: Dict[str, Any],
    group_cbs: int,
    num_iterations: Optional[int],
    msg_bits: int,
    args: argparse.Namespace,
) -> TimingModeResult:
    if mode == "gpu_decode_only":
        input_location = "gpu"
        prebuild_inputs = True
    elif mode == "gpu_plus_bus":
        input_location = "cpu"
        prebuild_inputs = True
    else:
        raise ValueError(f"Unknown timing mode: {mode}")

    groups = build_decode_groups(
        llrs_i8,
        meta,
        group_cbs,
        args,
        input_location=input_location,
        prebuild_inputs=prebuild_inputs,
    )

    input_llrs = prepare_input_list(groups, llrs_i8, meta, args, input_location=input_location)

    synchronize(cudart, stream)

    first_shape = getattr(input_llrs[0], "shape", None)
    print(f"\nTiming mode         : {mode}")
    print(f"input_location      : {input_location}")
    print(f"prebuild_inputs     : {prebuild_inputs}")
    print(f"first input type    : {type(input_llrs[0]).__name__}")
    print(f"first input shape   : {first_shape}")
    print(f"first tb_size       : {groups[0].tb_size} {'(unused)' if args.grouping_mode == 'explicit_codeblocks' else ''}")
    print(f"first code_rate     : {groups[0].code_rate} {'(unused)' if args.grouping_mode == 'explicit_codeblocks' else ''}")
    print(f"first rm length     : {groups[0].rate_match_len} {'(unused by explicit API)' if args.grouping_mode == 'explicit_codeblocks' else ''}")

    def timed_pass() -> Tuple[float, float]:
        synchronize(cudart, stream)

        pass_t0 = time.perf_counter()
        call_t0 = time.perf_counter()
        _ = call_aerial_decode(decoder, input_llrs, groups, meta, args, num_iterations)
        call_t1 = time.perf_counter()

        synchronize(cudart, stream)
        pass_t1 = time.perf_counter()

        return (pass_t1 - pass_t0) * 1e3, (call_t1 - call_t0) * 1e3

    for _ in range(int(args.warmup_repeats)):
        timed_pass()

    pass_ms_values: List[float] = []
    host_call_ms_values: List[float] = []

    for _ in range(int(args.repeats)):
        pass_ms, host_ms = timed_pass()
        pass_ms_values.append(pass_ms)
        host_call_ms_values.append(host_ms)

    result: Dict[str, Any]
    if args.verify and mode == args.verify_mode:
        decoded_bits = run_verify_pass(
            decoder,
            stream,
            cudart,
            groups,
            input_llrs,
            meta,
            args,
            num_iterations,
            msg_bits,
            args.decoded_format,
        )
        result = compute_ber_bler(decoded_bits, ref_bits)
    else:
        result = {
            "bit_errors": "",
            "total_bits": int(ref_bits.size),
            "ber": "",
            "block_errors": "",
            "total_blocks": int(ref_bits.shape[0]),
            "bler": "",
        }

    return TimingModeResult(
        mode=mode,
        input_location=input_location,
        prebuild_inputs=prebuild_inputs,
        pass_ms_values=np.asarray(pass_ms_values, dtype=np.float64),
        host_call_ms_values=np.asarray(host_call_ms_values, dtype=np.float64),
        result=result,
    )


# ---------------------------------------------------------------------------
# Per-run benchmark and CSV
# ---------------------------------------------------------------------------

def summarize_values(x: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.mean(x)),
        float(np.std(x, ddof=1 if x.size > 1 else 0)),
        float(np.min(x)),
        float(np.max(x)),
    )


def modes_to_run(args: argparse.Namespace) -> List[str]:
    if args.timing_mode == "both":
        return ["gpu_decode_only", "gpu_plus_bus"]
    return [args.timing_mode]



def estimate_bg1_num_cbs_from_tb_size(tb_size: int) -> int:
    """Approximate 38.212 BG1 segmentation count for warning messages only."""
    b_prime = int(tb_size) + 24
    kcb = 8448
    cb_crc = 24
    if b_prime <= kcb:
        return 1
    return int(np.ceil(float(b_prime) / float(kcb - cb_crc)))


def warn_if_unsafe_column_grouping(meta: Dict[str, Any], preview_group: DecodeGroup) -> None:
    if int(meta.get("base_graph", 0)) != 1 or preview_group.c <= 1:
        return

    inferred_c = estimate_bg1_num_cbs_from_tb_size(preview_group.tb_size)
    if inferred_c != preview_group.c:
        print(
            "WARNING: unsafe column_grouping detected. "
            f"You packed C={preview_group.c} columns, but Aerial will likely infer "
            f"about C={inferred_c} codeblocks from tb_size={preview_group.tb_size}. "
            "This can cause BER≈0.5 or CUDA illegal memory access. "
            "Use --grouping-mode independent_tbs."
        )


def benchmark_run_dir(run_dir: Path, decoder: Any, stream: Any, cudart: Any, args: argparse.Namespace, csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    meta, llrs_i8, ref_msg = load_exported_dataset(run_dir, args.limit_codeblocks)

    num_cb = int(meta["num_codeblocks_loaded"])
    llr_len = int(meta["llr_len"])
    msg_bits = int(meta["msg_bits_per_cb"])
    msg_bytes = int(meta["msg_bytes_per_cb"])
    num_iterations = get_num_iterations(meta, args)

    group_cbs = int(args.group_cbs if args.group_cbs is not None else args.batch_size)
    if group_cbs <= 0:
        die("--group-cbs/--batch-size must be > 0")
    if group_cbs > num_cb:
        group_cbs = num_cb

    ref_bits = bytes_to_bits_lsb(ref_msg, msg_bits)

    # Build one preview group on CPU only for printing params.
    preview_group = build_decode_groups(
        llrs_i8[: min(group_cbs, num_cb)],
        meta,
        group_cbs=min(group_cbs, num_cb),
        args=args,
        input_location="cpu",
        prebuild_inputs=False,
    )[0]

    print("\n" + "=" * 80)
    print(f"Run directory       : {run_dir}")
    print(f"mcs_index           : {meta.get('mcs_index', 'N/A')}")
    print(f"mcs_qm              : {meta.get('mcs_qm', 'N/A')}")
    print(f"mcs_R_x1024         : {meta.get('mcs_code_rate_x1024', 'N/A')}")
    print(f"snr_index           : {meta.get('snr_index', 'N/A')}")
    print(f"snr_db              : {meta.get('snr_db', 'N/A')}")
    print(f"export_stage        : {meta.get('export_stage', 'N/A')}")
    print(f"rate_matching       : {meta.get('rate_matching_enabled', 'N/A')}")
    print(f"mother_code_rate    : {meta.get('mother_code_rate', 'N/A')}")
    print(f"num_codeblocks      : {num_cb}")
    print(f"llr_len             : {llr_len}")
    print(f"msg_bits_per_cb     : {msg_bits}")
    print(f"msg_bytes_per_cb    : {msg_bytes}")
    print(f"base_graph          : {meta.get('base_graph', 'N/A')}")
    print(f"zc                  : {meta.get('zc', 'N/A')}")
    print(f"n_rows              : {meta.get('n_rows', 'N/A')}")
    print(f"aerial_input_len    : {meta.get('aerial_input_len', 'N/A')}")
    print(f"aerial_code_rate    : {meta.get('aerial_code_rate', 'N/A')}")
    print(f"rate_match_len/CB   : {meta.get('rate_match_len', 'N/A')}")
    print(f"group_cbs           : {group_cbs}")
    print(f"grouping_mode       : {args.grouping_mode}")
    print(f"decoder_api         : {args.decoder_api}")
    if args.decoder_api in ("explicit", "auto"):
        print(f"explicit_method     : {args.explicit_method_name}")
    print(f"num_aerial_inputs   : {num_cb if args.grouping_mode == 'independent_tbs' else (num_cb + group_cbs - 1) // group_cbs}")
    print(f"first group C       : {preview_group.c}")
    print(f"first tb_size       : {preview_group.tb_size} {'(unused)' if args.grouping_mode == 'explicit_codeblocks' else ''}")
    print(f"first rm length     : {preview_group.rate_match_len} {'(unused by explicit API)' if args.grouping_mode == 'explicit_codeblocks' else ''}")
    print(f"rv                  : {preview_group.rv}")
    print(f"num_iterations      : {num_iterations}")
    print(f"throughput_mode     : {args.throughput_mode}")
    print(f"warmup_repeats      : {args.warmup_repeats}")
    print(f"measured_repeats    : {args.repeats}")
    print("=" * 80)
    if args.grouping_mode == "column_grouping":
        warn_if_unsafe_column_grouping(meta, preview_group)

    rows: List[Dict[str, Any]] = []

    for mode in modes_to_run(args):
        mode_result = run_one_timing_mode(
            mode=mode,
            decoder=decoder,
            stream=stream,
            cudart=cudart,
            llrs_i8=llrs_i8,
            ref_bits=ref_bits,
            meta=meta,
            group_cbs=group_cbs,
            num_iterations=num_iterations,
            msg_bits=msg_bits,
            args=args,
        )

        pass_ms_values = mode_result.pass_ms_values
        host_ms_values = mode_result.host_call_ms_values

        per_cb_us_values = pass_ms_values / float(num_cb) * 1e3
        host_per_cb_us_values = host_ms_values / float(num_cb) * 1e3

        mean_pass_ms, std_pass_ms, min_pass_ms, max_pass_ms = summarize_values(pass_ms_values)
        mean_us, std_us, min_us, max_us = summarize_values(per_cb_us_values)
        mean_host_us, std_host_us, min_host_us, max_host_us = summarize_values(host_per_cb_us_values)

        total_elapsed_ms = float(np.sum(pass_ms_values))
        total_codeblocks_processed = num_cb * int(args.repeats)
        total_bits_processed = num_cb * msg_bits * int(args.repeats)

        row = {
            "run_dir": str(run_dir),
            "timing_mode": mode_result.mode,
            "input_location": mode_result.input_location,
            "prebuild_inputs": mode_result.prebuild_inputs,
            "cpu_input_dtype": args.cpu_input_dtype,
            "mcs_index": meta.get("mcs_index", ""),
            "mcs_qm": meta.get("mcs_qm", ""),
            "mcs_code_rate_x1024": meta.get("mcs_code_rate_x1024", ""),
            "mother_code_rate": meta.get("mother_code_rate", ""),
            "rate_matching_enabled": meta.get("rate_matching_enabled", ""),
            "snr_index": meta.get("snr_index", ""),
            "snr_db": meta.get("snr_db", ""),
            "noise_sigma": meta.get("noise_sigma", ""),
            "num_codeblocks": num_cb,
            "llr_len": llr_len,
            "msg_bits_per_cb": msg_bits,
            "msg_bytes_per_cb": msg_bytes,
            "base_graph": meta.get("base_graph", ""),
            "zc": meta.get("zc", ""),
            "n_rows": meta.get("n_rows", ""),
            "aerial_input_len": meta.get("aerial_input_len", ""),
            "aerial_prepend_punctured_llrs": meta.get("aerial_prepend_punctured_llrs", ""),
            "group_cbs": group_cbs,
            "grouping_mode": args.grouping_mode,
            "decoder_api": args.decoder_api,
            "explicit_method_name": args.explicit_method_name,
            "num_groups": (num_cb if args.grouping_mode == "independent_tbs" else (num_cb + group_cbs - 1) // group_cbs),
            "first_group_tb_size": preview_group.tb_size,
            "first_group_rate_match_len": preview_group.rate_match_len,
            "code_rate": preview_group.code_rate,
            "rv": preview_group.rv,
            "num_iterations": num_iterations if num_iterations is not None else "",
            "llr_sign": args.llr_sign,
            "llr_scale": args.llr_scale,
            "throughput_mode": args.throughput_mode,
            "warmup_repeats": args.warmup_repeats,
            "measured_repeats": args.repeats,
            "total_elapsed_ms": total_elapsed_ms,
            "mean_pass_ms": mean_pass_ms,
            "std_pass_ms": std_pass_ms,
            "min_pass_ms": min_pass_ms,
            "max_pass_ms": max_pass_ms,
            "mean_us_per_codeblock": mean_us,
            "std_us_per_codeblock": std_us,
            "min_us_per_codeblock": min_us,
            "max_us_per_codeblock": max_us,
            "mean_host_call_us_per_codeblock": mean_host_us,
            "std_host_call_us_per_codeblock": std_host_us,
            "min_host_call_us_per_codeblock": min_host_us,
            "max_host_call_us_per_codeblock": max_host_us,
            "codeblocks_per_second": float(total_codeblocks_processed / (total_elapsed_ms / 1e3)),
            "information_mbps": float(total_bits_processed / (total_elapsed_ms / 1e3) / 1e6),
            "aerial_bit_errors": mode_result.result["bit_errors"],
            "aerial_total_bits": mode_result.result["total_bits"],
            "aerial_ber": mode_result.result["ber"],
            "aerial_block_errors": mode_result.result["block_errors"],
            "aerial_total_blocks": mode_result.result["total_blocks"],
            "aerial_bler": mode_result.result["bler"],
        }

        print("\nAerial timing result:")
        print(f"  mode                        : {mode_result.mode}")
        print(f"  mean synchronized pass time : {row['mean_pass_ms']:.3f} ms")
        print(f"  std synchronized pass time  : {row['std_pass_ms']:.3f} ms")
        print(f"  min synchronized pass time  : {row['min_pass_ms']:.3f} ms")
        print(f"  max synchronized pass time  : {row['max_pass_ms']:.3f} ms")
        print(f"  mean latency per CB         : {row['mean_us_per_codeblock']:.3f} us")
        print(f"  std latency per CB          : {row['std_us_per_codeblock']:.3f} us")
        print(f"  host decode-call per CB     : {row['mean_host_call_us_per_codeblock']:.3f} us")
        print(f"  codeblocks/s                : {row['codeblocks_per_second']:.3f}")
        print(f"  information Mbps            : {row['information_mbps']:.3f}")

        if args.verify and mode == args.verify_mode:
            print("Aerial decoder result:")
            print(
                f"  BER={mode_result.result['ber']:.6e}, "
                f"BLER={mode_result.result['bler']:.6e}, "
                f"bit_errors={mode_result.result['bit_errors']}/{mode_result.result['total_bits']}, "
                f"block_errors={mode_result.result['block_errors']}/{mode_result.result['total_blocks']}"
            )
        elif args.verify:
            print(f"Aerial decoder result: verification skipped for mode={mode}; verified mode is {args.verify_mode}")
        else:
            print("Aerial decoder result: verification skipped")

        rows.append(row)

        if csv_path is not None:
            append_csv_row(csv_path, row)
            print(f"  CSV row appended: {csv_path}")

    return rows


CSV_FIELDNAMES = [
    "run_dir",
    "timing_mode",
    "input_location",
    "prebuild_inputs",
    "cpu_input_dtype",
    "mcs_index",
    "mcs_qm",
    "mcs_code_rate_x1024",
    "mother_code_rate",
    "rate_matching_enabled",
    "snr_index",
    "snr_db",
    "noise_sigma",
    "num_codeblocks",
    "llr_len",
    "msg_bits_per_cb",
    "msg_bytes_per_cb",
    "base_graph",
    "zc",
    "n_rows",
    "aerial_input_len",
    "aerial_prepend_punctured_llrs",
    "group_cbs",
    "grouping_mode",
    "decoder_api",
    "explicit_method_name",
    "num_groups",
    "first_group_tb_size",
    "first_group_rate_match_len",
    "code_rate",
    "rv",
    "num_iterations",
    "llr_sign",
    "llr_scale",
    "throughput_mode",
    "warmup_repeats",
    "measured_repeats",
    "total_elapsed_ms",
    "mean_pass_ms",
    "std_pass_ms",
    "min_pass_ms",
    "max_pass_ms",
    "mean_us_per_codeblock",
    "std_us_per_codeblock",
    "min_us_per_codeblock",
    "max_us_per_codeblock",
    "mean_host_call_us_per_codeblock",
    "std_host_call_us_per_codeblock",
    "min_host_call_us_per_codeblock",
    "max_host_call_us_per_codeblock",
    "codeblocks_per_second",
    "information_mbps",
    "aerial_bit_errors",
    "aerial_total_bits",
    "aerial_ber",
    "aerial_block_errors",
    "aerial_total_blocks",
    "aerial_bler",
]


def init_csv(path: Path, append: bool) -> None:
    """Create/truncate CSV at startup unless --append-csv is used."""
    if append and path.exists() and path.stat().st_size > 0:
        print(f"Appending CSV results to existing file: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

    print(f"Writing incremental CSV results to: {path}")


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one completed scenario row and force it to disk."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aerial LDPC benchmark for old and new Agora/FlexRAN datasets"
    )

    p.add_argument("path", type=Path, help="One run directory or recursive export root")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument(
        "--timing-mode",
        choices=["gpu_decode_only", "gpu_plus_bus", "both"],
        default="both",
        help=(
            "gpu_decode_only: prebuilt GPU inputs; "
            "gpu_plus_bus: prebuilt CPU inputs so Aerial copies over PCIe/bus; "
            "both: run both modes"
        ),
    )

    p.add_argument(
        "--cpu-input-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="CPU input dtype for gpu_plus_bus timing. float32 matches Aerial docs; float16 measures smaller bus payload.",
    )

    p.add_argument("--batch-size", type=int, default=128, help="Legacy alias for --group-cbs when --group-cbs is omitted")
    p.add_argument("--group-cbs", type=int, default=None, help="Number of codeblocks/TBs to place in one Aerial decode() call.")
    p.add_argument("--grouping-mode", choices=["explicit_codeblocks", "independent_tbs", "column_grouping"], default="explicit_codeblocks",
                   help="explicit_codeblocks: correct mode for the new explicit API, pack C exported CBs as [N,C] and pass BG/Zc/nRows directly. "
                        "independent_tbs: old safe legacy mode, each exported CB is one Aerial TB [N,1]. "
                        "column_grouping: old unsafe legacy experiment using fake TB sizes.")
    p.add_argument("--decoder-api", choices=["explicit", "legacy", "auto"], default="explicit",
                   help="explicit: use new decode_codeblocks_explicit API; legacy: use old decoder.decode TB-level API; auto: explicit for explicit_codeblocks, legacy otherwise.")
    p.add_argument("--explicit-method-name", default="decode_codeblocks_explicit",
                   help="Name of the new explicit method exposed on LdpcDecoder or pycuphy_ldpc_decoder.")
    p.add_argument("--warmup-repeats", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--limit-codeblocks", type=int, default=None)

    p.add_argument("--llr-sign", type=float, default=1.0, choices=[-1.0, 1.0])
    p.add_argument("--llr-scale", type=float, default=1.0)

    p.add_argument("--tb-size", type=int, default=None, help="Only valid with group size 1")
    p.add_argument("--code-rate", type=float, default=None)
    p.add_argument("--rate-match-len", type=int, default=None, help="Only valid with group size 1")
    p.add_argument("--rv", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=None)

    p.add_argument("--decoded-format", choices=["auto", "bits", "bytes"], default="auto")
    p.add_argument("--csv", type=Path, default=Path("aerial_ldpc_benchmark_modes.csv"))
    p.add_argument("--append-csv", action="store_true",
                   help="Append to an existing CSV instead of overwriting it at startup.")
    p.add_argument("--throughput-mode", action="store_true")

    verify = p.add_mutually_exclusive_group()
    verify.add_argument("--verify", dest="verify", action="store_true", default=True)
    verify.add_argument("--no-verify", dest="verify", action="store_false")

    p.add_argument(
        "--verify-mode",
        choices=["gpu_decode_only", "gpu_plus_bus"],
        default="gpu_decode_only",
        help="Which timing mode to use for BER/BLER verification when --verify is enabled.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.decoder_api == "explicit" and args.grouping_mode == "column_grouping":
        print("WARNING: --decoder-api explicit ignores fake TB sizes; prefer --grouping-mode explicit_codeblocks.", file=sys.stderr)
    if args.decoder_api == "legacy" and args.grouping_mode == "explicit_codeblocks":
        die("--grouping-mode explicit_codeblocks requires --decoder-api explicit or auto")

    run_dirs = find_run_dirs(args.path)
    print(f"Found {len(run_dirs)} run directory/directories:")
    for d in run_dirs:
        print(f"  {d}")

    # Use the explicit CLI num_iterations for constructor if given. Otherwise the
    # per-dataset num_iterations is passed on decode() when the Aerial version supports it.
    decoder, stream, cudart = setup_aerial_decoder(
        gpu=args.gpu,
        throughput_mode=args.throughput_mode,
        initial_iterations=args.num_iterations,
    )

    init_csv(args.csv, append=bool(args.append_csv))

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(benchmark_run_dir(run_dir, decoder, stream, cudart, args, csv_path=args.csv))

    print(f"\nDone. CSV results were written incrementally to: {args.csv}")


if __name__ == "__main__":
    main()
