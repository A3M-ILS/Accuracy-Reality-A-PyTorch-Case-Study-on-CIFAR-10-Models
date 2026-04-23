import torch, time

x = torch.randn(6000, 6000, device="cuda")
y = torch.randn(6000, 6000, device="cuda")

t0 = time.time()
z = x @ y
torch.cuda.synchronize()
print("Time:", time.time() - t0)
