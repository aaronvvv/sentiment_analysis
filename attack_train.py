# FGM
import paddle
import gc
class FGM:
    def __init__(self, model, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}
        self.name = "fgm"

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                self.backup[name] = param.clone()
                norm = paddle.norm(param.grad)
                if norm and not paddle.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.stop_gradient =True
                    param.add_(r_at)
                    param.stop_gradient = False
                    #param[:] = param.add(r_at)
                    #param.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if not para.stop_gradient and emb_name in name:
                assert name in self.backup
                para[:] = self.backup[name]#有问题
        del self.backup
        self.backup = {}
        gc.collect()


# PGD
class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
        self.name = "pgd"

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = paddle.norm(param.grad)
                if norm != 0 and not paddle.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        del self.backup
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if paddle.norm(r) > self.eps:
            r = self.eps * r / paddle.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and param.grad is not None:
                param.grad = self.grad_backup[name]

cross_loss = paddle.nn.loss.CrossEntropyLoss()

def attack(fun,model,batch_data,use_n_gpus=True):
    input_ids, token_type_ids, labels = batch_data
    if fun.name == "fgm":
        fun.attack()

        logits = model(input_ids, token_type_ids)
        loss_adv = cross_loss(logits, labels)

        if use_n_gpus:
            loss_adv = loss_adv.mean()

        loss_adv.backward()

        fun.restore()
    elif fun.name == "pgd":
        fun.backup_grad()
        pgd_k = 3
        for _t in range(pgd_k):
            fun.attack(is_first_attack=(_t == 0))

            if _t != pgd_k - 1:
                model.zero_grad()
            else:
                fun.restore_grad()
 
            logits = model(input_ids, token_type_ids)
            loss_adv = cross_loss(logits, labels)

            if use_n_gpus:
                loss_adv = loss_adv.mean()

            loss_adv.backward()

        fun.restore()