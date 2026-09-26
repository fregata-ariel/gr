import sys
sys.path.insert(0, "/content")
for _m in ("train_ar", "grammar_mask"):
    sys.modules.pop(_m, None)   # colab exec reuses one kernel: drop cached modules
import train_ar
train_ar.main(["--out", "/content/ptr4_n24", "--epochs", "300",
               "--patience", "20", "--num-samples", "400",
               "--pointer", "--pointer-legal", "--pointer-dist-bias",
               "--pointer-dist-bias-mode", "context",
               "--struct-pos", "--struct-pos-mode", "sinusoidal"])
