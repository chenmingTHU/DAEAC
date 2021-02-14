import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

'''
Evaluation metrics:

Acc = (TP + TN) / (TP + TN + FP + FN)
Se = TP / (TP + FN)
Sp = TN / (TN + FP)
Pp = TP / (TP + FP)

'''


class Eval(object):

    def __init__(self, num_class=4):

        self.num_class = num_class

    def _metrics(self, predictions, labels):

        '''
        :param predictions: A numpy array
        :param labels: A numpy array
        :return: Evaluation results in a dictionary format
        '''

        preds_n = np.where(predictions == 0, 1, 0)
        labels_n = np.where(labels == 0, 1, 0)

        preds_v = np.where(predictions == 1, 1, 0)
        labels_v = np.where(labels == 1, 1, 0)

        preds_s = np.where(predictions == 2, 1, 0)
        labels_s = np.where(labels == 2, 1, 0)

        preds_f = np.where(predictions == 3, 1, 0)
        labels_f = np.where(labels == 3, 1, 0)

        '''VEB'''
        fv = np.sum(np.where((preds_f + labels_v) == 2, 1, 0))

        s_v = preds_v + labels_v
        tn_v = np.sum(np.where(s_v == 0, 1, 0))
        tp_v = np.sum(np.where(s_v == 2, 1, 0))
        m_v = preds_v - labels_v
        fn_v = np.sum(np.where(m_v == 1, 1, 0))
        fp_v = np.sum(np.where(m_v == -1, 1, 0)) - fv

        Se_v = tp_v / (tp_v + fn_v)
        Pp_v = tp_v / (tp_v + fp_v)
        FPR_v = fp_v / (tn_v + fp_v)
        Acc_v = (tp_v + tn_v) / (tp_v + tn_v + fp_v + fn_v)

        '''SVEB'''

        s_s = preds_s + labels_s
        tn_s = np.sum(np.where(s_s == 0, 1, 0))
        tp_s = np.sum(np.where(s_s == 2, 1, 0))
        m_s = preds_s - labels_s
        fn_s = np.sum(np.where(m_s == 1, 1, 0))
        fp_s = np.sum(np.where(m_s == -1, 1, 0))

        Se_s = tp_s / (tp_s + fn_s)
        Pp_s = tp_s / (tp_s + fp_s)
        FPR_s = fp_s / (tn_s + fp_s)
        Acc_s = (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s)

        '''Normal'''

        s_n = preds_n + labels_n
        s_f = preds_f + labels_f

        tn_n = np.sum(np.where(s_n == 2, 1, 0))
        tp_f = np.sum(np.where(s_f == 2, 1, 0))

        Sp = tn_n / np.sum(preds_n)

        Se_f = tp_f / np.sum(preds_f)

        Acc = (tn_n + tp_s + tp_v + tp_f) / len(labels)

        eval_results = {
            'VEB': {'Se': Se_v,
                    'Pp': Pp_v,
                    'FPR': FPR_v,
                    'Acc': Acc_v},

            'SVEB': {'Se': Se_s,
                     'Pp': Pp_s,
                     'FPR': FPR_s,
                     'Acc': Acc_s},

            'F': {'Se': Se_f},
            'Sp': Sp,
            'Acc': Acc
        }

        return eval_results

    def _confusion_matrix(self, y_pred, y_label):

        return confusion_matrix(y_true=y_label, y_pred=y_pred)

    def _f1_score(self, y_pred, y_true):

        return f1_score(y_true=y_true, y_pred=y_pred, average=None)

    def _get_recall(self, y_pred, y_label):
        recalls = recall_score(y_pred=y_pred, y_true=y_label, average=None)
        return recalls

    def _get_precisions(self, y_pred, y_label):
        precisions = precision_score(y_pred=y_pred, y_true=y_label, average=None)
        return precisions

    def _sklean_metrics(self, y_pred, y_label):

        precisions = precision_score(y_pred=y_pred, y_true=y_label, average=None)
        recalls = recall_score(y_pred=y_pred, y_true=y_label, average=None)

        Pp = {'N': precisions[0], 'V': precisions[1], 'S': precisions[2], 'F': precisions[3]}
        Se = {'N': recalls[0], 'V': recalls[1], 'S': recalls[2], 'F': recalls[3]}

        return Pp, Se


class CorrelationAnalysis(object):

    def __init__(self, w):

        self.w = w
        self.omega = self._calculate_omega()
        self.po = self._calculate_correlation_matrix()

    def _calculate_omega(self):

        if not type(self.w) is np.ndarray:
            print('The weight must be numpy array')
            raise ValueError

        d = self.w.shape[0]

        omega = d * np.linalg.inv(np.matmul(self.w.transpose(), self.w))

        return omega

    def _calculate_correlation_matrix(self):

        po = np.zeros(shape=self.omega.shape)

        for i in range(self.omega.shape[0]):
            for j in range(self.omega.shape[1]):

                po[i][j] = -1.0 * self.omega[i][j] / (np.sqrt(self.omega[i][i] * self.omega[j][j]))

        return po


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Transpose') != -1:
#         m.weight.data.normal_(0.0, 0.01)
#         m.bias.data.normal_(0.0, 0.01)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.01)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.01)
#         m.bias.data.normal_(0.0, 0.01)

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
#         nn.init.kaiming_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.01)
#         nn.init.zeros_(m.bias)
#     elif classname.find('Linear') != -1:
#         nn.init.xavier_normal_(m.weight)
#         nn.init.zeros_(m.bias)

class EMA(object):
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def init_weights(m):

    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):

    nn.init.constant_(module.weight, val)

    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


# def obtain_pseudo_labels(net, loaded_models, t_batch, thrs, use_thr=True):
#
#     # Choosing Top-K reliable samples; K = 20
#     K = 20
#
#     net.load_state_dict(loaded_models)
#     net.cuda()
#     net.eval()
#
#     _, preds = net(t_batch)
#     preds = F.log_softmax(preds, dim=1).exp()
#     preds = preds.detach().cpu().numpy()
#     confidences = np.max(preds, axis=1)
#     pesudo_labels = np.argmax(preds, axis=1)
#     net.cpu()
#
#     if use_thr:
#         _legal_indices = []
#         for l in range(4):
#             indices_l = np.argwhere(pesudo_labels == l)
#             if len(indices_l) > 0:
#                 indices_l = np.squeeze(indices_l, axis=1)
#                 confidences_l = confidences[indices_l]
#                 legal_indices_l = np.argwhere(confidences_l >= thrs[l])
#                 if len(legal_indices_l) > 0:
#                     legal_indices_l = np.squeeze(legal_indices_l, axis=1)
#                     _legal_indices.append(indices_l[legal_indices_l])
#         _legal_indices = np.concatenate(_legal_indices)
#         pesudo_labels = pesudo_labels[_legal_indices]
#     else:
#         _legal_indices = np.arange(len(pesudo_labels))
#
#     return pesudo_labels, _legal_indices

# def obtain_pseudo_labels(net, loaded_models, t_batch, thrs, use_thr=True):

#     predicts = []
#     confidences = []
#     for idx in range(len(loaded_models)):
#         net.load_state_dict(loaded_models[idx])
#         net.cuda()
#         net.eval()

#         _, preds = net(t_batch)
#         preds = F.log_softmax(preds, dim=1).exp()
#         preds = preds.detach().cpu().numpy()
#         confidences.append(np.expand_dims(preds, axis=2))
#         preds = np.expand_dims(np.argmax(preds, axis=1), axis=1)
#         predicts.append(preds)

#         net.cpu()

#     predicts = np.concatenate(predicts, axis=1)
#     confidences = np.concatenate(confidences, axis=2)
#     confidences = np.max(np.mean(confidences, axis=2), axis=1)
#     pesudo_labels = []

#     for i in range(predicts.shape[0]):
#         predict_line = predicts[i]
#         pesudo_labels.append(np.argmax(np.bincount(predict_line)))

#     pesudo_labels = np.array(pesudo_labels)

#     if use_thr:
#         _legal_indices = []
#         for l in range(4):
#             indices_l = np.argwhere(pesudo_labels == l)
#             if len(indices_l) > 0:
#                 indices_l = np.squeeze(indices_l, axis=1)
#                 confidences_l = confidences[indices_l]
#                 legal_indices_l = np.argwhere(confidences_l >= thrs[l])
#                 if len(legal_indices_l) > 0:
#                     legal_indices_l = np.squeeze(legal_indices_l, axis=1)
#                     _legal_indices.append(indices_l[legal_indices_l])
#         _legal_indices = np.concatenate(_legal_indices)
#         pesudo_labels = pesudo_labels[_legal_indices]
#     else:
#         _legal_indices = np.arange(len(pesudo_labels))

#     return pesudo_labels, _legal_indices


def obtain_pseudo_labels(net, loaded_models, t_batch, thrs, use_thr=True):

    num_classes = 4

    predicts, confidences = [], []
    for idx in range(len(loaded_models)):
        net.load_state_dict(loaded_models[idx])
        net.cuda()
        net.eval()

        _, preds = net(t_batch)
        preds = F.log_softmax(preds, dim=1).exp().detach()
        confs = preds.unsqueeze(dim=2)
        preds = torch.argmax(preds, dim=1, keepdim=True)

        confidences.append(confs)
        predicts.append(preds)
    
    confidences = torch.cat(confidences, dim=2)
    predicts = torch.cat(predicts, dim=1)
    confidences, _ = torch.max(torch.mean(confidences, dim=2), dim=1)

    predicts_cate = []
    for i in range(num_classes):
        mask_i = (predicts == i).int()
        counter_i = torch.sum(mask_i, dim=1, keepdim=True)
        predicts_cate.append(counter_i)
    predicts_cate = torch.cat(predicts_cate, dim=1)
    pseudo_labels = torch.argmax(predicts_cate, dim=1)

    if use_thr:
        _legal_indices = []
        for l in range(num_classes):
            indices_l = (pseudo_labels == l)
            num_indices_l = torch.sum(indices_l).item()
            if num_indices_l > 0:
                _legal_indices_l = torch.nonzero(indices_l & (confidences >= thrs[l])).squeeze(dim=1)
                _legal_indices.append(_legal_indices_l)
        _legal_indices = torch.cat(_legal_indices, dim=0)
        pseudo_labels = pseudo_labels[_legal_indices]
    else:
        _legal_indices = torch.arange(pseudo_labels.size(0))
    
    return pseudo_labels, _legal_indices

