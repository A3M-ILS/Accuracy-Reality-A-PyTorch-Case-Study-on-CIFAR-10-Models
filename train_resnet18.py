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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = "./data"
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    batch_size = 128
    num_workers = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=False)

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

    epochs = 50
    print(f"GPU: {torch.cuda.get_device_name(0)} | batch_size={batch_size} | workers={num_workers}")
    print("Starting CIFAR-10 training (32x32)...")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        torch.cuda.synchronize()
        acc = evaluate()
        print(f"Epoch {epoch:02d}/{epochs} | loss={running_loss/len(train_loader):.4f} | test_acc={acc:.2f}% | lr={scheduler.get_last_lr()[0]:.4f} | time={time.time()-t0:.1f}s")

    save_path = "resnet18_cifar10_32.pth"
    torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, save_path)
    print(f"Model saved successfully: {save_path}")
    print("Done.")

if __name__ == "__main__":
    main()