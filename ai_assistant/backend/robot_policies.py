import sys
from pathlib import Path
import importlib.util

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def _import_inference_function(script_path: str, function_name: str):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("_temp_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

_pick_and_place_fn = None
_use_slicer_fn = None
_transfer_slices_fn = None

def _get_pick_and_place_fn():
    global _pick_and_place_fn
    if _pick_and_place_fn is None:
        script_path = project_root / "scripts" / "run_inference_pick_and_place.py"
        _pick_and_place_fn = _import_inference_function(str(script_path), "run_pick_and_place")
    return _pick_and_place_fn

def _get_use_slicer_fn():
    global _use_slicer_fn
    if _use_slicer_fn is None:
        script_path = project_root / "scripts" / "run_inference_use_slicer.py"
        _use_slicer_fn = _import_inference_function(str(script_path), "run_use_slicer")
    return _use_slicer_fn

def _get_transfer_slices_fn():
    global _transfer_slices_fn
    if _transfer_slices_fn is None:
        script_path = project_root / "scripts" / "run_inference_transfer_slices.py"
        _transfer_slices_fn = _import_inference_function(str(script_path), "run_transfer_slices")
    return _transfer_slices_fn

def policy_pick_and_place(model_id=None, num_episodes=None, episode_time_s=None, task_description=None) -> str:
    try:
        run_fn = _get_pick_and_place_fn()
        kwargs = {}
        if model_id: kwargs["model_id"] = model_id
        if num_episodes is not None: kwargs["num_episodes"] = num_episodes
        if episode_time_s is not None: kwargs["episode_time_s"] = episode_time_s
        if task_description: kwargs["task_description"] = task_description
        run_fn(**kwargs)
        return "✓ COMPLETED: Pick and place finished."
    except Exception as e:
        return f"ERROR: {e}"

def policy_use_slicer(model_id=None, num_episodes=None, episode_time_s=None, task_description=None) -> str:
    try:
        run_fn = _get_use_slicer_fn()
        kwargs = {}
        if model_id: kwargs["model_id"] = model_id
        if num_episodes is not None: kwargs["num_episodes"] = num_episodes
        if episode_time_s is not None: kwargs["episode_time_s"] = episode_time_s
        if task_description: kwargs["task_description"] = task_description
        run_fn(**kwargs)
        return "✓ COMPLETED: Slicing finished."
    except Exception as e:
        return f"ERROR: {e}"

def policy_transfer_slices(model_id=None, num_episodes=None, episode_time_s=None, task_description=None) -> str:
    try:
        run_fn = _get_transfer_slices_fn()
        kwargs = {}
        if model_id: kwargs["model_id"] = model_id
        if num_episodes is not None: kwargs["num_episodes"] = num_episodes
        if episode_time_s is not None: kwargs["episode_time_s"] = episode_time_s
        if task_description: kwargs["task_description"] = task_description
        run_fn(**kwargs)
        return "✓ COMPLETED: Transfer finished."
    except Exception as e:
        return f"ERROR: {e}"

POLICY_FUNCTIONS = {
    "run_pick_and_place": policy_pick_and_place,
    "run_use_slicer": policy_use_slicer,
    "run_transfer_slices": policy_transfer_slices,
}
