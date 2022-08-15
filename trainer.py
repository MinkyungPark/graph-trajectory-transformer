import math
import torch
from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from utils import Timer, to

class Trainer:
    def __init__(self, **kwargs):
        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

        for k,v in kwargs.items():
            setattr(self, k, v)

    def get_optimizer(self, model):
        if self.optimizer is None:
            self.optimizer = model.configure_optimizers(self)
        return self.optimizer

    def model_parr(self, model):
        if self.device != 'cpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).cuda()
        else:
            self.device = 'cpu'
            self.model = model

    def train(self, model, dataset, n_epochs=1, log_freq=100):
        self.model_parr(model)
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = self.get_optimizer(raw_model)

        model.train(True)
        vocab_size = dataset.vocab_size

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

        for _ in range(n_epochs):
            losses = []
            timer = Timer()
            for it, batch in enumerate(loader):
                batch = to(batch, self.device)
                with torch.set_grad_enabled(True):
                    logits, loss = model(*batch)
                    loss = loss.mean()
                    losses.append(loss.item())

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if self.lr_decay:
                    y = batch[-2]
                    self.n_tokens += (y != vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < self.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, self.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.learning_rate

                if it % log_freq == 0:
                    print(
                        f'[ training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')

            self.n_epochs += 1

