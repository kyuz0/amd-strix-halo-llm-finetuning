import gc, torch
from peft import PeftModel

def full_cleanup(model=None, trainer=None):
    """Aggressive memory release after training (ROCm-safe)."""
    try:
        acc = getattr(trainer, "accelerator", None)
        if acc:
            # Drop optimizer and model refs held by accelerate
            try:
                acc.state = None
                acc.optimizer = None
            except Exception:
                pass
            if hasattr(acc, "free_memory"):
                acc.free_memory()
    except Exception:
        pass

    try:
        if trainer:
            trainer.model = None
            trainer.optimizer = None
    except Exception:
        pass

    try:
        if isinstance(model, PeftModel):
            model.base_model = None
    except Exception:
        pass

    del model, trainer
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
