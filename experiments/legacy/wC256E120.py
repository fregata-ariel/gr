import sys
sys.path.insert(0, "/content")
import train_ar
train_ar.main([
    "--out", "/content/n24_c256e120", "--epochs", "120",
    "--d-model", "256", "--nhead", "8",
    "--num-layers", "6", "--dim-feedforward", "1024",
])
