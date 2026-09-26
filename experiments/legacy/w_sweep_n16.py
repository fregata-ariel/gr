import sys
sys.path.insert(0, "/content")
import train_ar
train_ar.main(["--out", "/content/sweep_n16", "--epochs", "300",
               "--patience", "20", "--num-samples", "400"])
