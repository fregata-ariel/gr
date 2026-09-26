import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_balanced_mask_n24_s0', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '0', '--sample-seed', '1000', '--ref-legal-mask'])
