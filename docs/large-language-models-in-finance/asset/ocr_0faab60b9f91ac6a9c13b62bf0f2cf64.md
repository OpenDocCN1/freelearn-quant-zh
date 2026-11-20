```
import torch
from torch.amp import autocast, GradScaler

class FinancialMixedPrecisionTrainer:
    def __init__(self, model, optimizer, device_type = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.device_type = device_type
        self.scaler = GradScaler(device_type, enabled = (device_type == "cuda"))
        self.fp32_modules = ["financial_calculations", "numerical_ops"]

    def training_step(self, batch):
        self.optimizer.zero_grad(set_to_none = True)
        amp_dtype = torch.float16 if self.device_type == "cuda" else torch.bfloat16
        with autocast(device_type = self.device_type, dtype = amp_dtype):
            outputs = self.model(**batch)
            language_loss = outputs.loss

        # Keep numerically delicate financial losses in FP32.
        financial_loss = self.compute_financial_loss_fp32(outputs)
        loss = language_loss + financial_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach()
```

Gradient clipping provides an additional safeguard by preventing gradient explosion during training, particularly important when processing numerical financial data that may contain extreme values:

\(\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \le \theta \\ \theta \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \theta \end{cases} \quad (2.35)\)

where \(\theta\) represents the clipping threshold and \(\|\mathbf{g}\|\) is the gradient norm.

Lastly, adaptive optimization algorithms such as AdamW are commonly used for Transformer training because they combine adaptive updates with decoupled weight decay. Their suitability should still be established through controlled experiments on the target task [Kingma and Ba, 2014]: