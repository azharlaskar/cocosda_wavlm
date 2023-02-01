import itertools
from collections import defaultdict

import torch
from torch.utils.data.sampler import BatchSampler
import numpy as np

NUMPY_RANDOM = np.random
def safe_random_choice(input_data, size,p=None):
    replace = len(input_data) < size
    return NUMPY_RANDOM.choice(input_data, size=size, replace=replace)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 
class HierarchicalSampler(BatchSampler):
    def __init__(
        self,
        labels,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
        max_seg_per_spk=500,
    ):
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        self.max_seg_per_spk=max_seg_per_spk
        self.batch_size = batch_size
        self.batches_per_super_tuple = batches_per_super_tuple
        self.samples_per_class = samples_per_class
        self.super_classes_per_batch = super_classes_per_batch

        # checks
        assert (
            self.batch_size % super_classes_per_batch == 0
        ), f"batch_size should be a multiple of {super_classes_per_batch}"
        self.sub_batch_len = self.batch_size // super_classes_per_batch

        if self.samples_per_class != "all":
            assert self.samples_per_class > 0
            assert (
                self.sub_batch_len % self.samples_per_class == 0
            ), "batch_size not a multiple of samples_per_class"

        all_super_labels = set(labels[:, outer_label])
        self.super_image_lists = {slb: defaultdict(list) for slb in all_super_labels}
        
        self.all_label = {slb: 0 for slb in set(labels[:, inner_label])}
        self.set_label=set(labels[:, inner_label])
        # print(self.set_label)
        for idx, instance in enumerate(labels):
            slb, lb = instance[outer_label], instance[inner_label]
            self.super_image_lists[slb][lb].append(idx)

        self.super_pairs = list(
            itertools.combinations(all_super_labels, super_classes_per_batch)
        )
        self.reshuffle()
    def __iter__(
        self,
    ):
        print("Sampler reshuffle")
        self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(
        self,
    ):
        return len(self.batches)

    def reshuffle(self):
        batches = []
        self.all_label = {}
        for combinations in self.super_pairs:

            for b in range(self.batches_per_super_tuple):

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    NUMPY_RANDOM.shuffle(all_classes)
                    for cl in all_classes:
                        # print(f"Class {cl} {self.all_label[cl]}",)
                        if cl not in self.all_label:
                            self.all_label[cl] = 0
                        if self.all_label[cl] >= self.max_seg_per_spk:
                            # print(f"full {cl}")
                            continue
                        if len(sub_batch) >= self.sub_batch_len:
                            break
                        instances = self.super_image_lists[slb][cl]
                        samples_per_class = (
                            self.samples_per_class
                            if self.samples_per_class != "all"
                            else len(instances)
                        )
                        if len(sub_batch) + samples_per_class > self.sub_batch_len:
                            continue
                        
                        sub_batch.extend(
                            safe_random_choice(instances, size=samples_per_class)
                        )
                        
                        self.all_label[cl] += samples_per_class
                    batch.extend(sub_batch)
                    # print(len(sub_batch))
                if len(batch) >1:
                  batches.append(batch)
        NUMPY_RANDOM.shuffle(batches)
        self.batches = batches
        m = min([len(i) for i in batches])
        print(f"Done sampler min len batch = {m}")