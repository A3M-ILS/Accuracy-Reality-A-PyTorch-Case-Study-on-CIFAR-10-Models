import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

def main():
    assert torch.cuda.is_available(), "CUDA not available."
    device = torch.device("cuda")

    # Perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # CIFAR-10 -> upscale to 224x224 to make it heavy
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = "./data"
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Start larger; adjust if OOM
    batch_size = 64  # try 512 if it fits; if OOM, try 192/128
    num_workers = 2
    persistent_workers = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )

    # Bigger model for stress
    model = torchvision.models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    accum_steps = 4

    # New AMP API (no warning)
    scaler = torch.amp.GradScaler("cuda")

    def evaluate():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return 100.0 * correct / total

    epochs = 10
    print(f"GPU: {torch.cuda.get_device_name(0)} | batch_size={batch_size} | workers={num_workers}")
    print("Starting BIG stress training (224x224)...")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        optimizer.zero_grad(set_to_none=True)  # ✅ only once per epoch (or after each step update)

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ✅ forward + loss scaled for accumulation
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y) / accum_steps

            # ✅ backward
            scaler.scale(loss).backward()

            # ✅ only update weights every accum_steps mini-batches
            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * accum_steps  # (undo the /accum_steps for readable loss)

            if step % 50 == 0:
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() / (1024**3)
                reserv = torch.cuda.memory_reserved() / (1024**3)
                print(f"Epoch {epoch:02d} Step {step:04d} | loss={running/50:.4f} | alloc={alloc:.2f}GB | reserved={reserv:.2f}GB")
                running = 0.0

        # ✅ handle leftover gradients if number of steps isn't divisible by accum_steps
        if len(train_loader) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.time() - t0
        acc = evaluate()
        print(f"Epoch {epoch:02d} done in {dt:.1f}s | test_acc={acc:.2f}%")


    print("Done.")

if __name__ == "__main__":
    main()
