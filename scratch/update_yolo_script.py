import json
import glob

notebooks = glob.glob('c:/CODE/KULIAH/TA/TA-IQBAL-ObjectDetectionExDARKwithLLIE/notebooks/scenario_*.ipynb')

for nb in notebooks:
    with open(nb, 'r', encoding='utf-8') as f:
        data = json.load(f)
    changed = False
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            # Cell 0.1
            if '0.1 · Environment Setup' in source and 'YOLO_MODEL' not in source:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if 'EPOCHS' in line and '@param' in line:
                        new_source.append('YOLO_MODEL  = "yolo11n.pt" # @param {type:"string"}\n')
                cell['source'] = new_source
                changed = True
            
            # Cell 0.3
            if '0.3 · Load Configuration & Define Paths' in source and 'YOLO_MODEL' not in source:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if 'set_global_seed(cfg["seed"])' in line:
                        new_source.append('\n# Override YOLO model dari variable notebook\n')
                        new_source.append('cfg["yolo"]["model"] = YOLO_MODEL\n')
                cell['source'] = new_source
                changed = True
                
    if changed:
        with open(nb, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Updated {nb}")
