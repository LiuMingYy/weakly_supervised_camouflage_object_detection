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
                self.next_input, self.next_input_insert, self.next_target, _, _, _, self.next_insert_target, self.next_target1 = next(self.loader)
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
                self.next_input_insert = self.next_input_insert.cuda(non_blocking=True)
                self.next_insert_target = self.next_insert_target.cuda(non_blocking=True)
                self.next_target1 = self.next_target1.cuda(non_blocking=True)

                self.next_input_insert = self.next_input_insert.float()
                self.next_insert_target = self.next_insert_target.float()
                self.next_target1 = self.next_target1.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target

        if self.cfg.mode == 'train':
            input_insert = self.next_input_insert
            insert_target = self.next_insert_target
            target1 = self.next_target1
            self.preload()
            return input, input_insert, target, insert_target, target1

        else:
            return input, target

        # else:
        #     self.preload()
        #     return input, target
