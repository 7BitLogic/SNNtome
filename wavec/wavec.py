"""
wavec.py — Minimal Pure-FP16 Python Interface for the Kernel 44 Wave Compiler (`libk44wavec.so`)
Supports AMD RDNA4 (`gfx1200` WMMA) and NVIDIA Ada/Blackwell (`sm_89` / `sm_100a` Tensor Cores).
"""

import ctypes
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

ACT_MAP = {
    None: 0, "none": 0, "identity": 0, "linear": 0,
    "relu": 1, "sigmoid": 2, "silu": 3, "swish": 3, "leaky_relu": 4,
}


def _load_lib() -> ctypes.CDLL:
    here = Path(__file__).resolve().parent
    lib_path = here / "libk44wavec.so"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Shared library {lib_path} not found. Run `./wavec/build_wavec.sh` or `make wavec` first."
        )
    lib = ctypes.CDLL(str(lib_path))

    lib.wavec_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.wavec_create.restype = ctypes.c_void_p
    lib.wavec_destroy.argtypes = [ctypes.c_void_p]
    lib.wavec_destroy.restype = None
    lib.wavec_gpu_release.argtypes = [ctypes.c_void_p]
    lib.wavec_gpu_release.restype = ctypes.c_int

    lib.wavec_add_input.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.wavec_add_input.restype = ctypes.c_int

    lib.wavec_add_flatten.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.wavec_add_flatten.restype = ctypes.c_int

    lib.wavec_add_conv2d.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p,
    ]
    lib.wavec_add_conv2d.restype = ctypes.c_int

    lib.wavec_add_linear_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p,
    ]
    lib.wavec_add_linear_f32.restype = ctypes.c_int

    lib.wavec_add_linear_f16.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p,
    ]
    lib.wavec_add_linear_f16.restype = ctypes.c_int

    lib.wavec_add_maxpool.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.wavec_add_maxpool.restype = ctypes.c_int

    lib.wavec_add_upsample2x.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.wavec_add_upsample2x.restype = ctypes.c_int

    lib.wavec_add_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.wavec_add_rmsnorm.restype = ctypes.c_int

    lib.wavec_add_swiglu.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    lib.wavec_add_swiglu.restype = ctypes.c_int

    lib.wavec_add_rope_gqa_attn.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
    ]
    lib.wavec_add_rope_gqa_attn.restype = ctypes.c_int

    lib.wavec_add_split.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.wavec_add_split.restype = ctypes.c_int

    lib.wavec_add_slice.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int),
    ]
    lib.wavec_add_slice.restype = ctypes.c_int

    lib.wavec_add_concat.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p,
    ]
    lib.wavec_add_concat.restype = ctypes.c_int

    lib.wavec_set_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.wavec_set_output.restype = ctypes.c_int

    lib.wavec_set_reuse_canvas.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.wavec_set_reuse_canvas.restype = ctypes.c_int
    lib.wavec_set_batch_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.wavec_set_batch_size.restype = ctypes.c_int
    lib.wavec_set_target_preset.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.wavec_set_target_preset.restype = ctypes.c_int

    lib.wavec_compile.argtypes = [ctypes.c_void_p]
    lib.wavec_compile.restype = ctypes.c_int
    lib.wavec_get_summary_json.argtypes = [ctypes.c_void_p]
    lib.wavec_get_summary_json.restype = ctypes.c_char_p

    lib.wavec_save_tcff.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.wavec_save_tcff.restype = ctypes.c_int
    lib.wavec_load_tcff_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.wavec_load_tcff_weights.restype = ctypes.c_int

    lib.wavec_gpu_init_plan.argtypes = [ctypes.c_void_p]
    lib.wavec_gpu_init_plan.restype = ctypes.c_int
    lib.wavec_gpu_infer_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    lib.wavec_gpu_infer_f32.restype = ctypes.c_int

    return lib


_LIB: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


class WavePlan:
    """Compiled Wave Execution Plan with TCFF v1.0 serialization and GPU execution."""

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL, summary: dict):
        self._handle = handle
        self._lib = lib
        self.summary = summary
        self._gpu_initialized = False

    @property
    def total_gflops(self) -> float:
        return float(self.summary.get("total_gflops", 0.0))

    @property
    def canvas_bytes(self) -> int:
        return int(self.summary.get("canvas_bytes", 0))

    @property
    def num_wave_dispatches(self) -> int:
        return int(self.summary.get("num_waves", self.summary.get("num_wave_dispatches", 0)))

    def save_tcff(self, tcff_path: Union[str, Path], model_prefix: str = "K44Model") -> int:
        n = self._lib.wavec_save_tcff(
            self._handle, str(tcff_path).encode("utf-8"), model_prefix.encode("utf-8")
        )
        if n < 0:
            raise RuntimeError(f"Failed to export TCFF v1.0 file: {tcff_path}")
        return self.num_wave_dispatches

    def load_tcff_weights(self, tcff_path: Union[str, Path]) -> None:
        rc = self._lib.wavec_load_tcff_weights(self._handle, str(tcff_path).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to load TCFF weights from: {tcff_path}")
        self._gpu_initialized = False

    def init_gpu(self) -> None:
        if not self._gpu_initialized:
            rc = self._lib.wavec_gpu_init_plan(self._handle)
            if rc != 0:
                raise RuntimeError("Failed to initialize GPU execution state for WavePlan.")
            self._gpu_initialized = True

    def infer_f32(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        self.init_gpu()
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        out_shape = tuple(self.summary["output_shape"])
        batch = int(self.summary.get("batch_size", 1))
        c, h, w = out_shape
        out_elems = batch * c * h * w if batch > 1 else c * h * w
        out_arr = np.empty((batch, c, h, w) if batch > 1 else (c, h, w), dtype=np.float32)
        gpu_ms = ctypes.c_float(0.0)
        rc = self._lib.wavec_gpu_infer_f32(
            self._handle,
            x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(x_f32.size),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(out_elems),
            ctypes.byref(gpu_ms),
        )
        if rc != 0:
            raise RuntimeError("GPU inference failed in wavec_gpu_infer_f32.")
        return out_arr, float(gpu_ms.value)


class WaveGraph:
    """5-Pass Wave Graph Builder for Vision (2D Spatial) and LLM (1D Token) Connectomes."""

    def __init__(
        self,
        max_B_per_wave: int = 288,
        default_S: int = 0,
        kv_cache_max_seq: int = 2048,
        fold_rmsnorm_gamma: bool = True,
        reuse_canvas: bool = False,
        batch_size: int = 1,
        target: str = "auto",
    ):
        self._lib = get_lib()
        self._handle = self._lib.wavec_create(
            int(max_B_per_wave), int(default_S), int(kv_cache_max_seq), 1 if fold_rmsnorm_gamma else 0
        )
        self._lib.wavec_set_reuse_canvas(self._handle, 1 if reuse_canvas else 0)
        self._lib.wavec_set_batch_size(self._handle, int(batch_size))
        self._lib.wavec_set_target_preset(self._handle, target.encode("utf-8"))

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.wavec_destroy(self._handle)
            self._handle = None

    def add_input(self, name: str, c: int, h: int = 1, w: int = 1, is_1d: bool = False) -> str:
        self._lib.wavec_add_input(self._handle, name.encode(), int(c), int(h), int(w), 1 if is_1d else 0)
        return name

    def add_conv2d(
        self, name: str, in_blob: str, out_blob: str,
        cin: int, cout: int, kh: int, kw: int,
        weights: np.ndarray, bias: Optional[np.ndarray] = None,
        stride: int = 1, pad: int = 0, act: Optional[str] = "silu",
        res_blob: Optional[str] = None,
    ) -> str:
        w_f32 = np.ascontiguousarray(weights, dtype=np.float32)
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_f32 is not None else None
        res_c = res_blob.encode() if res_blob else None
        self._lib.wavec_add_conv2d(
            self._handle, name.encode(), in_blob.encode(), out_blob.encode(),
            int(cin), int(cout), int(kh), int(kw), int(stride), int(pad), ACT_MAP.get(act, 0),
            w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), b_ptr, res_c,
        )
        return out_blob

    def add_linear(
        self, name: str, in_blob: str, out_blob: str,
        cin: int, cout: int, weights: np.ndarray,
        bias: Optional[np.ndarray] = None, act: Optional[str] = None,
        res_blob: Optional[str] = None,
    ) -> str:
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_f32 is not None else None
        res_c = res_blob.encode() if res_blob else None
        if weights.dtype == np.float16 or weights.dtype == np.uint16:
            w_u16 = np.ascontiguousarray(weights.view(np.uint16), dtype=np.uint16)
            self._lib.wavec_add_linear_f16(
                self._handle, name.encode(), in_blob.encode(), out_blob.encode(),
                int(cin), int(cout), ACT_MAP.get(act, 0),
                w_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), b_ptr, res_c,
            )
        else:
            w_f32 = np.ascontiguousarray(weights, dtype=np.float32)
            self._lib.wavec_add_linear_f32(
                self._handle, name.encode(), in_blob.encode(), out_blob.encode(),
                int(cin), int(cout), ACT_MAP.get(act, 0),
                w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), b_ptr, res_c,
            )
        return out_blob

    def add_rmsnorm(self, name: str, in_blob: str, out_blob: str, dim: int, gamma: np.ndarray, eps: float = 1e-5, pre_folded: bool = False) -> str:
        g_f32 = np.ascontiguousarray(gamma, dtype=np.float32)
        self._lib.wavec_add_rmsnorm(
            self._handle, name.encode(), in_blob.encode(), out_blob.encode(),
            int(dim), ctypes.c_float(eps), g_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1 if pre_folded else 0,
        )
        return out_blob

    def add_swiglu(self, name: str, gate_blob: str, up_blob: str, out_blob: str, dim: int = 0) -> str:
        self._lib.wavec_add_swiglu(
            self._handle, name.encode(), gate_blob.encode(), up_blob.encode(), out_blob.encode()
        )
        return out_blob

    def add_rope_gqa_attn(
        self, name: str, q_blob: str, k_blob: str, v_blob: str, out_blob: str,
        h_q: int, h_kv: int, head_dim: int, rope_theta: float = 10000.0, llama3_scaling: bool = False,
    ) -> str:
        self._lib.wavec_add_rope_gqa_attn(
            self._handle, name.encode(), q_blob.encode(), k_blob.encode(), v_blob.encode(), out_blob.encode(),
            int(h_q), int(h_kv), int(head_dim), ctypes.c_float(rope_theta), 1 if llama3_scaling else 0,
        )
        return out_blob

    def add_maxpool(self, name: str, in_blob: str, out_blob: str, kernel_size: int = 5, stride: int = 1, pad: int = 2) -> str:
        self._lib.wavec_add_maxpool(
            self._handle, name.encode(), in_blob.encode(), out_blob.encode(),
            int(kernel_size), int(stride), int(pad),
        )
        return out_blob

    def add_upsample2x(self, name: str, in_blob: str, out_blob: str) -> str:
        self._lib.wavec_add_upsample2x(self._handle, name.encode(), in_blob.encode(), out_blob.encode())
        return out_blob

    def add_split(self, name: str, in_blob: str, out_blobs: Sequence[str]) -> List[str]:
        arr = (ctypes.c_char_p * len(out_blobs))(*[b.encode() for b in out_blobs])
        self._lib.wavec_add_split(self._handle, name.encode(), in_blob.encode(), len(out_blobs), arr)
        return list(out_blobs)

    def add_slice(self, name: str, in_blob: str, out_blobs: Sequence[str], slice_sizes: Sequence[int]) -> List[str]:
        arr = (ctypes.c_char_p * len(out_blobs))(*[b.encode() for b in out_blobs])
        sz = (ctypes.c_int * len(slice_sizes))(*[int(s) for s in slice_sizes])
        self._lib.wavec_add_slice(self._handle, name.encode(), in_blob.encode(), len(out_blobs), arr, sz)
        return list(out_blobs)

    def add_concat(self, name: str, in_blobs: Sequence[str], out_blob: str) -> str:
        arr = (ctypes.c_char_p * len(in_blobs))(*[b.encode() for b in in_blobs])
        self._lib.wavec_add_concat(self._handle, name.encode(), len(in_blobs), arr, out_blob.encode())
        return out_blob

    def set_output(self, out_blob: str) -> None:
        self._lib.wavec_set_output(self._handle, out_blob.encode())

    def compile(self) -> WavePlan:
        rc = self._lib.wavec_compile(self._handle)
        if rc != 0:
            raise RuntimeError("WaveGraph 5-pass compilation failed.")
        raw_json = self._lib.wavec_get_summary_json(self._handle)
        summary = json.loads(raw_json.decode("utf-8")) if raw_json else {}
        return WavePlan(self._handle, self._lib, summary)

