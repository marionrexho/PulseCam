import importlib.util, sys, os
import numpy as np, torch, torch.nn.functional as F
import onnxruntime as ort

TB = sys.argv[1]            # toolbox dir
CKPT = sys.argv[2]          # checkpoint name inside final_model_release
OUT = sys.argv[3]           # output .onnx path

spec = importlib.util.spec_from_file_location("ep", os.path.join(TB, "neural_methods", "model", "EfficientPhys.py"))
ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)

sd = torch.load(os.path.join(TB, "final_model_release", CKPT), map_location="cpu")
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
print("dense shape:", tuple(sd["final_dense_1.weight"].shape))

model = ep.EfficientPhys(frame_depth=10, img_size=72)
print(model.load_state_dict(sd))
model.eval()

x = torch.randn(11, 3, 72, 72) * 1.0
with torch.no_grad():
    ref = model(x).numpy()

# Export-friendly TSM (pad + concat instead of in-place slice assignment); mathematically identical
def tsm_forward(self, x):
    nt, c, h, w = x.size()
    n_batch = nt // self.n_segment
    x = x.view(n_batch, self.n_segment, c, h, w)
    fold = c // self.fold_div
    left = F.pad(x[:, 1:, :fold], (0, 0, 0, 0, 0, 0, 0, 1))
    right = F.pad(x[:, :-1, fold:2 * fold], (0, 0, 0, 0, 0, 0, 1, 0))
    out = torch.cat([left, right, x[:, :, 2 * fold:]], dim=2)
    return out.view(nt, c, h, w)
ep.TSM.forward = tsm_forward
# torch.diff is not exportable; frame[i+1] - frame[i] is the same thing
torch.diff = lambda t, dim=0: t.narrow(dim, 1, t.size(dim) - 1) - t.narrow(dim, 0, t.size(dim) - 1)

with torch.no_grad():
    patched = model(x).numpy()
print("patched vs original max diff:", float(np.abs(patched - ref).max()))

torch.onnx.export(model, x, OUT, input_names=["frames"], output_names=["pulse"], opset_version=17, dynamo=False)

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
o = sess.run(None, {"frames": x.numpy()})[0]
print("onnx vs original max diff:", float(np.abs(o - ref).max()), "| output shape:", o.shape, "| size MB:", round(os.path.getsize(OUT) / 1e6, 2))
print("ops:", sorted({n.op_type for n in __import__("onnx").load(OUT).graph.node}))
