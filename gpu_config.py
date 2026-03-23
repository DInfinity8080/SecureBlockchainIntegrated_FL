"""
GPU / Accelerator configuration for Federated Learning.

Detects available hardware and configures TensorFlow accordingly:
  - Apple Silicon (M1/M2/M3) via Metal (requires compatible tensorflow-metal)
  - NVIDIA GPUs via CUDA
  - AMD GPUs via ROCm (Linux only)
  - Falls back to CPU gracefully

Control via environment variable:
  FL_DEVICE=gpu    Force GPU usage
  FL_DEVICE=cpu    Force CPU usage
  FL_DEVICE=auto   (default) Use GPU only if a large model benefits from it

Import this module BEFORE importing tensorflow in any script.
"""

import os
import platform


def configure_gpu():
    """Detect and configure the best available compute device.

    Returns a string describing the device that will be used.
    """
    system = platform.system()
    machine = platform.machine()

    # ── Check for user override ──────────────────────────────────
    user_choice = os.environ.get("FL_DEVICE", "auto").lower()

    if user_choice == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        return "CPU (forced via FL_DEVICE=cpu)"

    # ── Apple Silicon (Metal) ────────────────────────────────────
    if system == "Darwin" and machine == "arm64":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        try:
            import tensorflow as tf
            metal_devices = tf.config.list_physical_devices("GPU")
            if metal_devices and user_choice == "gpu":
                for device in metal_devices:
                    try:
                        tf.config.experimental.set_memory_growth(device, True)
                    except RuntimeError:
                        pass
                return f"Apple Metal ({len(metal_devices)} GPU)"
            elif metal_devices and user_choice == "auto":
                # Auto mode: for small FL models, CPU is faster due to
                # Metal init overhead (~5-10s per process).  GPU only
                # helps with larger models (>1M parameters).
                # Default to CPU for this FL workload.
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                return "CPU (Apple Silicon — Metal available, use FL_DEVICE=gpu to enable)"
            else:
                return "CPU (Apple Silicon — tensorflow-metal not available)"
        except Exception as e:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                import tensorflow as tf
            except Exception:
                pass
            return f"CPU (Metal plugin error: {e})"

    # ── NVIDIA (CUDA) / AMD (ROCm) on Linux or Windows ───────────
    if system in ("Linux", "Windows"):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        is_rocm = os.path.exists("/opt/rocm") or "ROCM_HOME" in os.environ

        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            gpu_names = []
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_names.append(details.get("device_name", "GPU"))
                except Exception:
                    gpu_names.append("GPU")
            backend = "ROCm" if is_rocm else "CUDA"
            return f"{backend} ({', '.join(gpu_names)})"
        else:
            if system == "Windows":
                return "CPU (no CUDA GPUs detected — AMD GPUs need ROCm on Linux)"
            else:
                return "CPU (no GPUs detected — install CUDA or ROCm drivers)"

    # ── Fallback ─────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    return "CPU"


# Auto-configure on import
DEVICE_NAME = configure_gpu()
