import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_depth1_base_n24_s1', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '1', '--sample-seed', '1001'])
