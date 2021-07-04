import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])  # 把x先拉成一行，然后把所有的x摞起来，变成n行


class Architect(object):

    def __init__(self, model, model2, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.model2 = model2
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)  # 用来更新α的optimizer

        """
      我们更新梯度就是theta = theta + v + weight_decay * theta
        1.theta就是我们要更新的参数
        2.weight_decay*theta为正则化项用来防止过拟合
        3.v的值我们分带momentum和不带momentum：
          普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
          带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
      """
    # 【完全复制外面的Network更新w的过程】，对应公式6第一项的w − ξ*dwLtrain(w, α)
    # 不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新


def _compute_unrolled_model(self, input, target, eta, optim_1):
    loss = self.model._loss(input, target)  # Ltrain
    theta = _concat(self.model.parameters()).data  # 把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
    try:
        moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
            self.network_momentum)  # momentum*v,用的就是Network进行w更新的momentum
    except:
        moment = torch.zeros_like(theta)  # 不加momentum
    dtheta = _concat(torch.autograd.grad(loss,
                                         self.model.parameters())).data + self.network_weight_decay * theta  # 前面的是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项
    # 对参数进行更新，等价于optimizer.step()
    unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))  # w − ξ*dwLtrain(w, α)

    return unrolled_model


def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()  # 清除上一步的残余更新参数值
    if unrolled:  # 用论文的提出的方法
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:  # 不用论文提出的bilevel optimization，只是简单的对α求导
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()  # 应用梯度：根据反向传播得到的梯度进行参数的更新， 这些parameters的梯度是由loss.backward()得到的，optimizer存了这些parameters的指针
    # 因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数


def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()  # 反向传播，计算梯度


def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # 计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    # w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta,
                                                  network_optimizer)  # unrolled_model里的w已经是做了一次更新后的w，也就是得到了w'
    # Lval
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)  # 对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新

    unrolled_loss.backward()
    # dαLval(w',α)
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # 对alpha求梯度
    # dw'Lval(w',α)
    vector = [v.grad.data for v in unrolled_model.parameters()]  # unrolled_model.parameters()得到w‘
    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta, ig.data)
    # 对α进行更新
    for v, g in zip(self.model.arch_parameters(), dalpha):
        if v.grad is None:
            v.grad = Variable(g.data)
        else:
            v.grad.data.copy_(g.data)


# 对应optimizer.step()，对新建的模型的参数进行更新
def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()  # Returns a dictionary containing a whole state of the module.

    params, offset = {}, 0
    for k, v in self.model.named_parameters():  # k是参数的名字，v是参数
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset + v_length].view(v.size())  # 将参数k的值更新为theta对应的值
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)  # 模型中的参数已经更新为做一次反向传播后的值
    model_new.load_state_dict(model_dict)  # 恢复模型中的参数，也就是我新建的mode_new中的参数为model_dict
    return model_new.cuda()


# 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
def _hessian_vector_product(self, vector, input, target, r=1e-2):  # vector就是dw'Lval(w',α)
    R = r / _concat(vector).norm()  # epsilon

    # dαLtrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)  # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(2 * R,
                    v)  # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)  # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


# 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
def _hessian_vector_product(self, vector, input, target, r=1e-2):  # vector就是dw'Lval(w',α)
    R = r / _concat(vector).norm()  # epsilon

    # dαLtrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)  # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target, reduction='none')

    logits = unrolled_model_w1(input_valid)
    u = -criterion(logits, target_valid)

    z = [[1 if target_i == target_j else 0 for target_j in target_valid] for target_i in target_train]
    z = torch.FloatTensor(z).cuda()

    x = [[self.v(input_j) * self.v(input_i) for input_j in input_valid] for input_i in input_train]
    x = torch.FloatTensor(x).cuda()
    x = F.softmax(x, dim=1)  # compute softmax along with the rows

    a = F.sigmoid(torch.dot(x * z * u, self.r))

    grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), torch.ones_like(loss))


    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(2 * R,
                    v)  # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target, reduction='none')
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)  # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]