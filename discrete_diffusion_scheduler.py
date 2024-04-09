import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch.cuda.amp import autocast

eps = 1e-8


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a[t].unsqueeze(1)
    while len(out.shape)<len(x_shape):
        out = out.unsqueeze(-1)
    return out

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1=0.9999, att_T=0.000001, ctt_1=0.000001, ctt_T=0.9999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct

    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N

    at = torch.tensor(at.astype('float64'))
    bt = torch.tensor(bt.astype('float64'))
    ct = torch.tensor(ct.astype('float64'))
    att = torch.tensor(att.astype('float64'))
    btt = torch.tensor(btt.astype('float64'))
    ctt = torch.tensor(ctt.astype('float64'))
    return at, bt, ct, att, btt, ctt

# def alpha_schedule2(time_step, N=100, att_1=0.9999, att_T=0.000001, ctt_1=0.000001, ctt_T=0.9999):
#     att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
#     att = np.concatenate(([1], att))
#     at = att[1:] / att[:-1]
#
#     bt = (np.exp(-np.arange(1,len(at)+1)))*(1-at)/N
#
#
#     ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
#     ctt = np.concatenate(([0], ctt))
#     one_minus_ctt = 1 - ctt
#     one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
#     ct = 1 - one_minus_ct
#
#     bt = (1 - at - ct) / N
#     att = np.concatenate((att[1:], [1]))
#     ctt = np.concatenate((ctt[1:], [0]))
#     btt = (1 - att - ctt) / N
#
#     at = torch.tensor(at.astype('float64'))
#     bt = torch.tensor(bt.astype('float64'))
#     ct = torch.tensor(ct.astype('float64'))
#     att = torch.tensor(att.astype('float64'))
#     btt = torch.tensor(btt.astype('float64'))
#     ctt = torch.tensor(ctt.astype('float64'))
#     return at, bt, ct, att, btt, ctt


class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            mask_id, a_classes, c_classes,
            e_classes,model,
            diffusion_step=100,
            alpha_init_type='alpha1',
            auxiliary_loss_weight=1e-4,
            adaptive_auxiliary_loss=True,
            batch_size=256,
            max_length=10
    ):
        super().__init__()

        self.mask_id = mask_id
        self.a_classes = a_classes
        self.c_classes = c_classes
        self.e_classes = e_classes
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length

        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = [1,1]

        if alpha_init_type == "alpha1":
            at1, bt1, ct1, att1, btt1, ctt1 = alpha_schedule(self.num_timesteps, N=self.a_classes - 1)
            at2, bt2, ct2, att2, btt2, ctt2 = alpha_schedule(self.num_timesteps, N=self.c_classes - 1)
            at3, bt3, ct3, att3, btt3, ctt3 = alpha_schedule(self.num_timesteps, N=self.e_classes - 1)
        else:
            print("alpha_init_type is Wrong !! ")

        log_at1 = torch.log(at1)
        log_bt1 = torch.log(bt1)
        log_ct1 = torch.log(ct1)
        log_at2 = torch.log(at2)
        log_bt2 = torch.log(bt2)
        log_ct2 = torch.log(ct2)
        log_at3 = torch.log(at3)
        log_bt3 = torch.log(bt3)
        log_ct3 = torch.log(ct3)

        log_cumprod_at1 = torch.log(att1)
        log_cumprod_bt1 = torch.log(btt1)
        log_cumprod_ct1 = torch.log(ctt1)

        log_cumprod_at2 = torch.log(att2)
        log_cumprod_bt2 = torch.log(btt2)
        log_cumprod_ct2 = torch.log(ctt2)

        log_cumprod_at3 = torch.log(att3)
        log_cumprod_bt3 = torch.log(btt3)
        log_cumprod_ct3 = torch.log(ctt3)

        log_1_min_ct1 = log_1_min_a(log_ct1)
        log_1_min_cumprod_ct1 = log_1_min_a(log_cumprod_ct1)

        log_1_min_ct2 = log_1_min_a(log_ct2)
        log_1_min_cumprod_ct2 = log_1_min_a(log_cumprod_ct2)

        log_1_min_ct3 = log_1_min_a(log_ct3)
        log_1_min_cumprod_ct3 = log_1_min_a(log_cumprod_ct3)

        # assert log_add_exp(log_ct1, log_1_min_ct1).abs().sum().item() < 1.e-5
        # assert log_add_exp(log_cumprod_ct1, log_1_min_cumprod_ct1).abs().sum().item() < 1.e-5
        #
        # assert log_add_exp(log_ct2, log_1_min_ct2).abs().sum().item() < 1.e-5
        # assert log_add_exp(log_cumprod_ct2, log_1_min_cumprod_ct2).abs().sum().item() < 1.e-5
        #
        # assert log_add_exp(log_ct3, log_1_min_ct3).abs().sum().item() < 1.e-5
        # assert log_add_exp(log_cumprod_ct3, log_1_min_cumprod_ct3).abs().sum().item() < 1.e-5

        self.register_buffer('log_at1', log_at1.float())
        self.register_buffer('log_bt1', log_bt1.float())
        self.register_buffer('log_ct1', log_ct1.float())
        self.register_buffer('log_cumprod_at1', log_cumprod_at1.float())
        self.register_buffer('log_cumprod_bt1', log_cumprod_bt1.float())
        self.register_buffer('log_cumprod_ct1', log_cumprod_ct1.float())
        self.register_buffer('log_1_min_ct1', log_1_min_ct1.float())
        self.register_buffer('log_1_min_cumprod_ct1', log_1_min_cumprod_ct1.float())

        self.register_buffer('log_at2', log_at2.float())
        self.register_buffer('log_bt2', log_bt2.float())
        self.register_buffer('log_ct2', log_ct2.float())
        self.register_buffer('log_cumprod_at2', log_cumprod_at2.float())
        self.register_buffer('log_cumprod_bt2', log_cumprod_bt2.float())
        self.register_buffer('log_cumprod_ct2', log_cumprod_ct2.float())
        self.register_buffer('log_1_min_ct2', log_1_min_ct2.float())
        self.register_buffer('log_1_min_cumprod_ct2', log_1_min_cumprod_ct2.float())

        self.register_buffer('log_at3', log_at3.float())
        self.register_buffer('log_bt3', log_bt3.float())
        self.register_buffer('log_ct3', log_ct3.float())
        self.register_buffer('log_cumprod_at3', log_cumprod_at3.float())
        self.register_buffer('log_cumprod_bt3', log_cumprod_bt3.float())
        self.register_buffer('log_cumprod_ct3', log_cumprod_ct3.float())
        self.register_buffer('log_1_min_ct3', log_1_min_ct3.float())
        self.register_buffer('log_1_min_cumprod_ct3', log_1_min_cumprod_ct3.float())


    def multinomial_kl(self, log_prob1, log_prob2):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep1(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at1, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt1, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct1, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct1, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :1, :] + log_1_min_ct, log_ct),
                log_add_exp(log_x_t[:, 1:, :] + log_at, log_bt)
            ],
            dim=1
        )

        return log_probs

    def q_pred_one_timestep2(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at2, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt2, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct2, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct2, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :1, :] + log_1_min_ct, log_ct),
                log_add_exp(log_x_t[:, 1:, :] + log_at, log_bt)
            ],
            dim=1
        )

        return log_probs

    def q_pred_one_timestep3(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at3, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt3, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct3, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct3, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :1, :] + log_1_min_ct, log_ct),
                log_add_exp(log_x_t[:, 1:, :] + log_at, log_bt)
            ],
            dim=1
        )

        return log_probs

    def q_pred1(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at1, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt1, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct1, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct1, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :1, :] + log_1_min_cumprod_ct, log_cumprod_ct),
                log_add_exp(log_x_start[:, 1:, :] + log_cumprod_at, log_cumprod_bt)
            ],
            dim=1
        )
        return log_probs

    def q_pred2(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at2, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt2, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct2, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct2, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :1, :] + log_1_min_cumprod_ct, log_cumprod_ct),
                log_add_exp(log_x_start[:, 1:, :] + log_cumprod_at, log_cumprod_bt)
            ],
            dim=1
        )

        return log_probs

    def q_pred3(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at3, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt3, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct3, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct3, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :1, :] + log_1_min_cumprod_ct, log_cumprod_ct),
                log_add_exp(log_x_start[:, 1:, :] + log_cumprod_at, log_cumprod_bt)
            ],
            dim=1
        )

        return log_probs

    def predict_start(self, log_A_t, log_C_t,log_E_t, t):  # p(x0|xt)
        A_t = log_onehot_to_index(log_A_t)
        C_t = log_onehot_to_index(log_C_t)
        E_t = log_onehot_to_index(log_E_t)

        A_out,C_out,E_out,_ = self.model(A_t, C_t, E_t, t.unsqueeze(-1).float()/self.num_timesteps)

        # A_out, C_out, E_out = F.softmax(A_out,dim=1) ,\
        #                         F.softmax(C_out,dim=1) ,\
        #                         F.softmax(E_out ,dim=1)

        A_out, C_out, E_out = F.softmax(A_out,dim=1)  * (1+2e-4)-1e-4,\
                                F.softmax(C_out,dim=1) * (1+2e-4)-1e-4,\
                                F.softmax(E_out ,dim=1) * (1+2e-4)-1e-4

        A_out, C_out, E_out = A_out + (A_out>1)*(1-A_out.detach()) + (A_out<1e-40)*(1e-40-A_out.detach()), \
                                C_out + (C_out > 1) * (1 - C_out.detach()) + (C_out < 1e-40) * (1e-40 - C_out.detach()), \
                                    E_out + (E_out > 1) * (1 - E_out.detach()) + (E_out < 1e-40) * (1e-40 - E_out.detach()),

        A_out = A_out / A_out.sum(1,keepdim=True).detach()
        C_out = C_out / C_out.sum(1, keepdim=True).detach()
        E_out = E_out / E_out.sum(1, keepdim=True).detach()

        A_out, C_out, E_out = torch.clip(A_out,1e-40,1),\
                                torch.clip(C_out,1e-40,1),\
                                    torch.clip(E_out, 1e-40,1)

        A_log_pred = torch.log(A_out.double()).float()
        C_log_pred = torch.log(C_out.double()).float()
        E_log_pred = torch.log(E_out.double()).float()

        batch_size = log_A_t.size()[0]

        A_zero_vector = torch.zeros((batch_size, 1, log_A_t.shape[-1]),device=log_A_t.device).type_as(log_A_t) - 70
        A_log_pred = torch.cat((A_zero_vector, A_log_pred), dim=1)
        A_log_pred = torch.clamp(A_log_pred, -70, 0)

        C_zero_vector = torch.zeros((batch_size, 1, log_C_t.shape[-1]),device=log_C_t.device).type_as(log_C_t) - 70
        C_log_pred = torch.cat((C_zero_vector, C_log_pred), dim=1)
        C_log_pred = torch.clamp(C_log_pred, -70, 0)

        E_zero_vector = torch.zeros((batch_size, 1, log_E_t.shape[-1],log_E_t.shape[-1]),device=log_E_t.device).type_as(log_E_t) - 70
        E_log_pred = torch.cat((E_zero_vector, E_log_pred), dim=1)
        E_log_pred = torch.clamp(E_log_pred, -70, 0)
        return A_log_pred,C_log_pred,E_log_pred

    def q_posterior1(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0
        assert t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_id).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_t.shape[-1])

        log_qt = self.q_pred1(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_zero_vector,log_qt[:,1:,:]), dim=1)
        log_qt = log_qt[:, 1:, :]
        log_cumprod_ct = extract(self.log_cumprod_ct1, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.a_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((log_one_vector,ct_cumprod_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep1(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_zero_vector,log_qt_one_timestep[:, 1:, :]), dim=1)
        log_ct = extract(self.log_ct1, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.a_classes - 1, -1)
        ct_vector = torch.cat((log_one_vector,ct_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, 1:, :] - log_qt
        q = torch.cat((log_zero_vector,q), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred1(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_posterior2(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_id).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_t.shape[-1])

        log_qt = self.q_pred2(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,1:,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, 1:, :]
        log_cumprod_ct = extract(self.log_cumprod_ct1, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.c_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep2(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_zero_vector,log_qt_one_timestep[:, 1:, :]), dim=1)
        log_ct = extract(self.log_ct2, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.c_classes - 1, -1)
        ct_vector = torch.cat((log_one_vector, ct_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, 1:, :] - log_qt
        q = torch.cat((log_zero_vector, q), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred2(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_posterior3(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps

        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_id).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_t.shape[-1], log_x_t.shape[-1])

        log_qt = self.q_pred3(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,1:,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, 1:, :]
        log_cumprod_ct = extract(self.log_cumprod_ct3, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.e_classes - 1, -1,-1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep3(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_zero_vector,log_qt_one_timestep[:, 1:, :]), dim=1)
        log_ct = extract(self.log_ct3, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.e_classes - 1, -1,-1)
        ct_vector = torch.cat((log_one_vector, ct_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, 1:, :] - log_qt
        q = torch.cat((log_zero_vector,q), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred3(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_A,log_C,log_E,t):  # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))

        log_A_recon,log_C_recon,log_E_recon = self.predict_start(log_A,log_C,log_E,t)
        log_A_pred = self.q_posterior1(
                log_x_start=log_A_recon, log_x_t=log_A, t=t)
        log_C_pred = self.q_posterior2(
                log_x_start=log_C_recon, log_x_t=log_C, t=t)
        log_E_pred = self.q_posterior3(
                log_x_start=log_E_recon, log_x_t=log_E, t=t)
        return log_A_pred,log_C_pred,log_E_pred

    @torch.no_grad()
    def p_sample(self, log_A,log_C,log_E, t):  # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob_A,model_log_prob_C,model_log_prob_E = self.p_pred(log_A,log_C,log_E,t)
        A_out = self.log_sample_categorical1(model_log_prob_A,t)
        C_out = self.log_sample_categorical2(model_log_prob_C,t)
        E_out = self.log_sample_categorical3(model_log_prob_E,t)
        return A_out,C_out,E_out

    def log_sample_categorical1(self, logits,t):  # use gumbel to sample onehot vector from log probability
        # uniform = torch.rand_like(logits)
        # gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
        # # sample = (logits/torch.clamp(t/self.num_timesteps,1e-10,1)+gumbel_noise).argmax(dim=1)

        # sample = (logits + gumbel_noise).argmax(dim=1)
        probs = torch.softmax(logits,dim=1)
        if not self.model.training:
            probs = probs * (probs > 1e-5).float()
        # sample = torch.distributions.Categorical(probs=probs.transpose(-1, -2)).sample()

        probs_2d = probs.transpose(-1,-2).reshape(-1, self.a_classes)
        samples_2d = torch.multinomial(probs_2d, 1, True).unsqueeze(0)
        sample = samples_2d.reshape(probs.shape[0],probs.shape[-1])

        log_sample = index_to_log_onehot(sample, self.a_classes)

        return log_sample

    def log_sample_categorical2(self, logits,t):  # use gumbel to sample onehot vector from log probability
        # uniform = torch.rand_like(logits)
        # gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
        # # sample = (logits/torch.clamp(t/self.num_timesteps,1e-10,1)+gumbel_noise).argmax(dim=1)
        # sample = (logits + gumbel_noise).argmax(dim=1)
        # log_sample = index_to_log_onehot(sample, self.c_classes)

        probs = torch.softmax(logits,dim=1)
        if not self.model.training:
            probs = probs * (probs > 1e-5).float()
        # sample = torch.distributions.Categorical(probs=probs.transpose(-1, -2)).sample()\
        probs_2d = probs.transpose(-1,-2).reshape(-1, self.c_classes)
        samples_2d = torch.multinomial(probs_2d, 1, True).unsqueeze(0)
        sample = samples_2d.reshape(probs.shape[0],probs.shape[-1])

        log_sample = index_to_log_onehot(sample, self.c_classes)
        return log_sample

    def log_sample_categorical3(self, logits,t):  # use gumbel to sample onehot vector from log probability
        # uniform = torch.rand_like(logits)
        # gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        # # sample = (logits/torch.clamp(t/self.num_timesteps,1e-10,1) + gumbel_noise).argmax(dim=1)
        # sample = (logits+ gumbel_noise).argmax(dim=1)

        probs = torch.softmax(logits,dim=1)
        if not self.model.training:
            probs = probs * (probs > 1e-5).float()
        # sample = torch.distributions.Categorical(probs=probs.permute(0,2,3,1)).sample()
        probs_2d = probs.permute(0,2,3,1).reshape(-1, self.e_classes)
        samples_2d = torch.multinomial(probs_2d, 1, True).unsqueeze(0)
        sample = samples_2d.reshape(probs.shape[0],probs.shape[-1],probs.shape[-1])

        sample = sample.float()

        upper_triangular_mask = torch.zeros([1,sample.shape[-1], sample.shape[-1]], device=logits.device, dtype=torch.float)
        indices = torch.tril_indices(row=sample.shape[-1], col=sample.shape[-1], offset=-1)
        upper_triangular_mask[:, indices[0], indices[1]] = 1
        sample = sample * upper_triangular_mask
        sample = sample + sample.transpose(-1, -2)
        sample = sample.long()

        log_sample = index_to_log_onehot(sample, self.e_classes)
        return log_sample
    def q_sample1(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred1(log_x_start, t)
        log_sample = self.log_sample_categorical1(log_EV_qxt_x0,t)
        return log_sample
    def q_sample2(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred2(log_x_start, t)
        log_sample = self.log_sample_categorical2(log_EV_qxt_x0,t)
        return log_sample

    def q_sample3(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred3(log_x_start, t)
        log_sample = self.log_sample_categorical3(log_EV_qxt_x0,t)
        return log_sample

    def sample_time(self, b, device):
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def _add_noise1(self,x,t,num_classes):
        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)
        log_xt = self.q_sample1(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        return log_x_start,log_xt,xt

    def _add_noise2(self,x,t,num_classes):
        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)
        log_xt = self.q_sample2(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        return log_x_start,log_xt,xt

    def _add_noise3(self,x,t,num_classes):
        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)
        log_xt = self.q_sample3(log_x_start=log_x_start, t=t)

        upper_triangular_mask = torch.zeros([1, 1,x.shape[1], x.shape[2]], device=x.device, dtype=torch.float)
        indices = torch.tril_indices(row=x.size(1), col=x.size(2), offset=0)
        upper_triangular_mask[:, :,indices[0], indices[1]] = 1
        log_xt = log_xt * upper_triangular_mask
        log_xt = log_xt + log_xt.transpose(-1,-2)

        xt = log_onehot_to_index(log_xt)
        return log_x_start,log_xt,xt

    def _train_loss_single1(self,x,xt,log_xt,log_x0_recon,t,pt,num_classes):
        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)

        log_model_prob = self.q_posterior1(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)  # go through q(xt_1|xt,x0)

        log_true_prob = self.q_posterior1(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.mask_id).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        # kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        # decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float().unsqueeze(-1)
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        loss1 = kl_loss / pt.unsqueeze(-1)
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:, 1:, :], log_x0_recon[:, 1:, :])
            kl_aux = kl_aux * mask_weight
            # kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t.unsqueeze(-1) / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / (pt.unsqueeze(-1))
            vb_loss += loss2
            vb_loss = sum_except_batch(vb_loss)
        return log_model_prob, vb_loss

    def _train_loss_single2(self,x,xt,log_xt,log_x0_recon,t,pt,num_classes):
        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)

        log_model_prob = self.q_posterior2(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)  # go through q(xt_1|xt,x0)

        log_true_prob = self.q_posterior2(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.mask_id).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        # kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        # decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float().unsqueeze(-1)
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        loss1 = kl_loss / pt.unsqueeze(-1)
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:, 1:, :], log_x0_recon[:, 1:, :])
            kl_aux = kl_aux * mask_weight
            # kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t.unsqueeze(-1) / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / (pt.unsqueeze(-1))
            vb_loss += loss2
            vb_loss = sum_except_batch(vb_loss)
        return log_model_prob, vb_loss

    def _train_loss_single3(self,x,xt,log_xt,log_x0_recon,t,pt,num_classes):

        x_start = x
        log_x_start = index_to_log_onehot(x_start, num_classes)

        log_model_prob = self.q_posterior3(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)  # go through q(xt_1|xt,x0)

        log_true_prob = self.q_posterior3(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.mask_id).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]

        upper_triangular_mask = torch.zeros([1, x.shape[-1], x.shape[-1]], device=x.device, dtype=torch.float)
        indices = torch.tril_indices(row=x.shape[-1], col=x.shape[-1], offset=-1)
        upper_triangular_mask[:, indices[0], indices[1]] = 1

        kl = kl * mask_weight * upper_triangular_mask
        # kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob) * upper_triangular_mask
        # decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float().unsqueeze(-1).unsqueeze(-1)
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        loss1 = kl_loss / pt.unsqueeze(-1).unsqueeze(-1)
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:, 1:, :], log_x0_recon[:, 1:, :])
            kl_aux = kl_aux * mask_weight *  upper_triangular_mask
            # kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t.unsqueeze(-1).unsqueeze(-1) / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / (pt.unsqueeze(-1).unsqueeze(-1))
            vb_loss += loss2
            vb_loss = sum_except_batch(vb_loss)
        return log_model_prob, vb_loss

    def _train_loss(self, A,C,E):  # get the KL loss
        b, device = A.size(0), A.device
        assert self.loss_type == 'vb_stochastic'
        t, pt = self.sample_time(b, device)

        log_A_start,log_At,At = self._add_noise1(A,t,self.a_classes)
        log_C_start, log_Ct, Ct = self._add_noise2(C ,t, self.c_classes)
        log_E_start, log_Et, Et = self._add_noise3(E, t, self.e_classes)

        log_A0_recon,log_C0_recon,log_E0_recon = self.predict_start(log_At,log_Ct,log_Et,t=t)  # P_theta(x0|xt)
        A_log_model_prob, A_vb_loss = self._train_loss_single1(A,At,log_At,log_A0_recon,t,pt,self.a_classes)
        C_log_model_prob, C_vb_loss = self._train_loss_single2(C,Ct,log_Ct,log_C0_recon,  t, pt,self.c_classes)
        E_log_model_prob, E_vb_loss = self._train_loss_single3(E, Et,log_Et,log_E0_recon, t, pt,self.e_classes)
        vb_loss = A_vb_loss + C_vb_loss + E_vb_loss
        return A_log_model_prob,C_log_model_prob, E_log_model_prob,vb_loss

    @property
    def device(self):
        return self.model.to_logits[-1].weight.device

    def forward(self,A,C,E):
        A_log_model_prob,C_log_model_prob, E_log_model_prob, loss = self._train_loss(A,C,E)
        loss = loss.sum() / (A.size()[0] * A.size()[1])
        return A_log_model_prob,C_log_model_prob, E_log_model_prob,loss

    def sample(self,sample_num=128):
        device = self.log_at1.device
        batch_size=256
        sample_num_list = [batch_size] * (sample_num//batch_size) + ([] if sample_num%batch_size==0 else [sample_num%batch_size])

        molecule_list = []
        for num in sample_num_list:
            A = torch.zeros(size=(num,self.a_classes,self.max_length),device=device)
            A[:,self.mask_id,:] = 1
            C = torch.zeros(size=(num, self.c_classes, self.max_length), device=device)
            C[:, self.mask_id,:] = 1
            E = torch.zeros(size=(num,self.e_classes, self.max_length,self.max_length), device=device)
            E[:, self.mask_id,:,:] = 1

            log_A = torch.log(A + 1e-30)
            log_C = torch.log(C + 1e-30)
            log_E = torch.log(E + 1e-30)

            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1):
                    t = torch.full((log_A.shape[0],), diffusion_index, device=device, dtype=torch.long)
                    log_A,log_C,log_E = self.p_sample(log_A,log_C,log_E,t)  # log_z is log_onehot

            content_token = [log_onehot_to_index(log_A).cpu(),log_onehot_to_index(log_C).cpu(),log_onehot_to_index(log_E).cpu()]
            molecule_list.append(content_token)
        return molecule_list