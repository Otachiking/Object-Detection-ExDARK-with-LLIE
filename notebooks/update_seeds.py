import json
import glob
import os

notebooks = glob.glob('C:/CODE/KULIAH/TA/TA-IQBAL-ObjectDetectionExDARKwithLLIE/notebooks/*.ipynb')

for nb_path in notebooks:
    if 'RESULT' in nb_path:
        continue
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # 1) Inject SEED = 42
            for i, line in enumerate(source):
                if 'QUICK_TEST  = False' in line and not any('SEED        = 42' in l for l in source):
                    source.insert(i, 'SEED        = 42     # @param {type:"integer"}\n')
                    modified = True
                    break
                    
            # 2) Override cfg['seed']
            for i, line in enumerate(source):
                if 'set_global_seed(cfg["seed"])' in line and not any('cfg["seed"] = SEED' in l for l in source):
                    source.insert(i, "if 'SEED' in locals() or 'SEED' in globals():\n")
                    source.insert(i+1, '    cfg["seed"] = SEED\n')
                    modified = True
                    break
                    
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f'Updated {os.path.basename(nb_path)}')

print('Done!')
