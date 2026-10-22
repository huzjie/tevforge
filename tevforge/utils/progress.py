# -*- coding: utf-8 -*-
"""A tiny progress tracker (no external tqdm dependency)."""


class Progress:
    def __init__(self, total, label="", every=10):
        self.total = max(1, total)
        self.done = 0
        self.label = label
        self.every = every

    def update(self, n=1):
        self.done += n
        if self.label and (self.done % self.every == 0 or self.done >= self.total):
            print("[%s] %d/%d" % (self.label, self.done, self.total), flush=True)

    def ratio(self):
        return self.done / self.total
