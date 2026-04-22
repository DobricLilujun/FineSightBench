import sys
import json
from pathlib import Path
from finesightbench.evaluation.framework import evaluate_model_on_val_data

model_name = sys.argv[1]
local_files_only = len(sys.argv) > 2 and sys.argv[2] == '--local'

r = evaluate_model_on_val_data(
    model_name=model_name,
    val_root=Path('data/val_data'),
    output_dir=Path('outputs/vlm_eval'),
    max_samples_per_split=60,
    local_files_only=local_files_only,
    run_attention_test=True,
)
print(json.dumps({
    'model_name': r.get('model_name'),
    'status': r.get('status'),
    'accuracy': r.get('accuracy'),
    'num_evaluated': r.get('num_evaluated'),
    'num_correct': r.get('num_correct'),
    'num_errors': r.get('num_errors'),
    'attention_test': r.get('attention_test', {}).get('status'),
    'elapsed_sec': round(r.get('elapsed_sec', 0), 1),
}, indent=2))
print('accuracy_by_split:', json.dumps(r.get('accuracy_by_split', {}), indent=2))
print('accuracy_by_task:', json.dumps(r.get('accuracy_by_task', {}), indent=2))
