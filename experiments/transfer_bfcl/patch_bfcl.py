"""Register Qwen3.5-0.8B (base, g23v3, g23v4) x (stock, campaign-prompt) in a BFCL checkout."""
import pathlib, re, shutil, sys
root = pathlib.Path(sys.argv[1]) / "bfcl_eval"
shutil.copy(pathlib.Path(__file__).with_name("qwen35_fc.py"), root / "model_handler/local_inference/qwen35_fc.py")

cfg = root / "constants/model_config.py"
s = cfg.read_text()
imp = "from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler\n"
assert imp in s
if "qwen35_fc" not in s:
    s = s.replace(imp, imp + "from bfcl_eval.model_handler.local_inference.qwen35_fc import Qwen35FCHandler\n")
VARIANTS = [
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "https://huggingface.co/Qwen/Qwen3.5-0.8B", "Qwen"),
    ("Qwen/Qwen3.5-0.8B-toolpatch-g23v3", "Qwen3.5-0.8B-toolpatch-g23v3", "https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v3", "scasella91"),
    ("Qwen/Qwen3.5-0.8B-toolpatch-g23v4", "Qwen3.5-0.8B-toolpatch-g23v4", "https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v4", "scasella91"),
]
entries, names = "", []
for name, disp, url, org in VARIANTS:
    for suffix, dsuffix in (("-FC", " (FC)"), ("-campaign-prompt-FC", " (FC, campaign system prompt)")):
        key = name + suffix
        names.append(key)
        entries += f'''    "{key}": ModelConfig(
        model_name="{name}",
        display_name="{disp}{dsuffix}",
        url="{url}",
        org="{org}",
        license="apache-2.0",
        model_handler=Qwen35FCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
'''
anchor = '    "Qwen/Qwen3-0.6B-FC": ModelConfig('
assert anchor in s
if '"Qwen/Qwen3.5-0.8B-FC"' not in s:
    s = s.replace(anchor, entries + anchor)
cfg.write_text(s)

sup = root / "constants/supported_models.py"
t = sup.read_text()
if '"Qwen/Qwen3.5-0.8B-FC"' not in t:
    t = t.replace('    "Qwen/Qwen3-0.6B-FC",', "".join(f'    "{n}",\n' for n in names) + '    "Qwen/Qwen3-0.6B-FC",')
    assert '"Qwen/Qwen3.5-0.8B-FC"' in t
sup.write_text(t)
print("patched", len(names), "entries")
