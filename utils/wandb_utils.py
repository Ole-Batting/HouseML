import cv2
import numpy as np
import pytorch_lightning as pl
import torch as T
import torchvision
import wandb


class WandbImageCallback(pl.Callback):
    def __init__(self, val_samples, max_samples=12, depth=True, debug=False):
        super().__init__()
        self.val_x, self.val_brick = val_samples
        self.val_x = self.val_x[:max_samples]
        self.val_bgr = (self.val_x[:,:3].permute(0,2,3,1).numpy() * 255).astype(np.uint8)
        self.val_brick = (self.val_brick[:max_samples].numpy() * 255).astype(np.uint8)
        self.debug = debug
        self.max_samples = max_samples
        self.linewidth = 1
        
        image_size = self.val_bgr.shape[1:3]
            
        self.canvas = np.zeros((self.max_samples, *image_size, 3), dtype=np.uint8)

        for i in range(max_samples):
            canvas = np.ascontiguousarray(self.val_bgr[i], dtype=np.uint8)
            thresh = (self.val_brick[i] >= 128).astype(np.uint8)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), self.linewidth)
            self.canvas[i] = canvas
    
    def on_validation_end(self, trainer, pl_module):
        val_x = self.val_x.to(device=pl_module.device)
        canvas = self.canvas.copy()
    
        out = pl_module(val_x).cpu()

        canvas2 = np.zeros_like(self.val_bgr)

        for i in range(self.max_samples):
            seg = (out[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
            blr = cv2.GaussianBlur(seg, (3, 3), 0)
            _, thr = cv2.threshold(blr, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            canvas_copy = np.ascontiguousarray(canvas[i])
            cv2.drawContours(canvas_copy, contours, -1, (255, 0, 0), self.linewidth)
            canvas2[i] = canvas_copy
        
        canvas2 = T.Tensor(canvas2).permute(0,3,1,2)[:,[2,1,0],:,:].to(device=pl_module.device)
        
        grid = torchvision.utils.make_grid(canvas2, 4)
    
        cap = "Input with: GT (green), output-Otsu (blue)"
        trainer.logger.experiment.log({
            "Test_examples": wandb.Image(grid, caption=cap),
            "global_step": trainer.global_step
        })