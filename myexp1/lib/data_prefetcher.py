import torch

class DataPrefetcher(object):  # 在训练过程中减少CPU和GPU之间的数据传输等待时间，从而提高训练效率。
    def __init__(self, loader, cfg):
        self.cfg = cfg
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            if self.cfg.mode == 'train':
                self.next_input, self.next_target, self.next_syn_image, self.next_syn_mask, self.next_region, _, _ = next(self.loader)
            else:
                self.next_input, self.next_target, _, _ = next(self.loader)
            # self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            self.next_input = self.next_input.float()  # if need
            self.next_target = self.next_target.float()  # if need

            if self.cfg.mode == 'train':
                self.next_syn_image = self.next_syn_image.cuda(non_blocking=True)
                self.next_syn_mask = self.next_syn_mask.cuda(non_blocking=True)
                self.next_region = self.next_region.cuda(non_blocking=True)

                self.next_syn_image = self.next_syn_image.float() #if need
                self.next_syn_mask = self.next_syn_mask.float() #if need
                self.next_region = self.next_region.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if self.cfg.mode == 'train':
            syn_image = self.next_syn_image
            syn_mask = self.next_syn_mask
            region = self.next_region

            self.preload()
            return input, target, syn_image, syn_mask, region

        else:
            self.preload()
            return input, target
