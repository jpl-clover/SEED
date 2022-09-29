import torch
import torch.nn as nn

from .models.utils import (
    get_model_from_name,
    swav_resnet_proj_hidden_dim_map,
    simclr_resnet_models,
)
from .utils import concat_all_gather


class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """

    def __init__(self, args):
        """
        Arguments:
            args: Namespace object

        `args` needs to have the following attributes:
            - queue (int): queue size (default 65536; needs to be divisble by batch size)
            - temp (float): temperature for student encoder
            - distill_t (float): distillation temperature for teacher encoder
            - dim (int): feature dimension (default: 128)
            - distributed (bool): whether we're running in distributed mode
            - student_arch (str): student encoder architecture
            - student_weights (str): path to student encoder weights (can be True/False as well, if the arch is in torch_hub_models.json)
            - student_mlp (bool): whether to add an MLP to the student encoder
            - teacher_arch (str): teacher encoder architecture
            - teacher_weights (str): path to teacher encoder weights (can be True/False as well, if the arch is in torch_hub_models.json)
            - teacher_ssl (str): SSL method used to pretrain the teacher. Can be "simclr", "moco" or "swav". Defaults to simclr
        """
        super(SEED, self).__init__()

        self.K = args.queue
        self.student_t = args.temp
        self.distill_t = args.distill_t
        self.dim = args.dim
        self.dist = args.distributed

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        self.student = get_model_from_name(
            args.student_arch,
            args.dim,
            pretrained=args.student_weights,
            freeze_enc=False,
            init_fc=True,
        )
        self.teacher = get_model_from_name(
            args.teacher_arch,
            args.dim,
            pretrained=args.teacher_weights,
            freeze_enc=True,
            init_fc=True,
        )
        if args.student_mlp:
            if "fc" in self.student._modules and isinstance(self.student.fc, nn.Linear):
                dim_mlp = self.student.fc.weight.shape[1]
                self.student.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.fc
                )
            # handle efficientnet and mobilenet from torch hub, without fc
            elif "classifier" in self.student._modules and isinstance(
                self.student.classifier, nn.Sequential
            ):
                dim_mlp = self.student.classifier[1].weight.shape[1]
                self.student.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.classifier
                )
            else:
                raise NotImplementedError(
                    f"Don't know how to add 1-layer projection head to student model {args.student_arch} yet."
                )

            if args.teacher_ssl == "moco":  # create a projection head for MoCo
                dim_mlp = self.teacher.fc.weight.shape[1]
                self.teacher.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher.fc)

        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, image):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """

        # compute query features
        s_emb = self.student(image)  # NxC
        s_emb = nn.functional.normalize(s_emb, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            t_emb = self.teacher(image)  # keys: NxC
            t_emb = nn.functional.normalize(t_emb, dim=1)

        # cross-Entropy Loss
        logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])
        logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])

        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

        logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
        logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

        # compute soft labels
        logit_stu /= self.student_t
        logit_tea = nn.functional.softmax(logit_tea / self.distill_t, dim=1)

        # de-queue and en-queue
        self._dequeue_and_enqueue(t_emb, concat=self.dist)

        return logit_stu, logit_tea

