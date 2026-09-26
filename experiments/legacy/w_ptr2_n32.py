import sys
sys.path.insert(0, "/content")
import train_ar
train_ar.main(["--out", "/content/ptr2_n32", "--epochs", "300",
               "--patience", "20", "--num-samples", "400",
               "--pointer", "--pointer-legal", "--pointer-dist-bias"])
