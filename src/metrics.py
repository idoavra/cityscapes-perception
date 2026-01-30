import torch

class StreamSegMetrics:
    """
    Fast confusion matrix based metrics. 
    Maintains a running tally of TP, FP, FN for the whole epoch.
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = None

    def update(self, label_trues, label_preds):
        # Flatten and filter out the ignore_index (255)
        mask = (label_trues >= 0) & (label_trues < self.n_classes)
        label_trues = label_trues[mask].to(torch.int64)
        label_preds = label_preds[mask].to(torch.int64)

        # Calculate confusion matrix using bincount
        # (trues * n + preds) creates a unique index for every cell in the matrix
        indices = self.n_classes * label_trues + label_preds
        hist = torch.bincount(indices, minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        
        if self.confusion_matrix is None:
            self.confusion_matrix = hist
        else:
            self.confusion_matrix += hist

    def get_results(self):
        """Returns mIoU and per-class IoU"""
        hist = self.confusion_matrix
        tp = torch.diag(hist)
        fp = hist.sum(dim=0) - tp
        fn = hist.sum(dim=1) - tp
        
        iu = tp / (tp + fp + fn + 1e-7)
        miou = torch.nanmean(iu)
        
        return {
            "Overall mIoU": miou.item(),
            "Class IoU": iu.tolist() # Convert tensor to list for the printer
        }

    def reset(self):
        self.confusion_matrix = None