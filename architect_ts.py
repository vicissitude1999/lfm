import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model1, model2, v, r, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model1 = model1
    self.model2 = model2
    self.v = v
    self.r = r
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  # compute w1
  def _compute_unrolled_model(self, input_train, target_train, eta, optim_1):
    loss = self.model1._loss(input_train, target_train)
    theta = _concat(self.model1.parameters()).data
    try:
      moment = _concat(optim_1.state[v]['momentum_buffer']
                       for v in self.model1.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(
        loss, self.model1.parameters())).data + self.network_weight_decay * theta
    unrolled_model_w1 = self._construct_model_from_theta(
        theta.sub(eta, moment + dtheta))

    return unrolled_model_w1

  # valid data are used to compute weights
  def _compute_unrolled_model_w2(self, input_train, target_train, input_valid, target_valid,
                                 eta2, optim_2, unrolled_model_w1):
    # compute weights
    u = -unrolled_model_w1._loss(input_valid, target_valid, reduction='none')

    z = [[1 if target_i == target_j else 0 for target_j in target_valid] for target_i in target_train]
    z = torch.FloatTensor(z).cuda()

    x = [[self.v(input_j)*self.v(input_i) for input_j in input_valid] for input_i in input_train]
    x = torch.FloatTensor(x).cuda()
    x = F.softmax(x, dim=1) # compute softmax along with the rows

    a = F.sigmoid(torch.dot(x * z * u, self.r))

    loss = self.model2._loss(input_train, target_train, reduction='none')
    weighted_loss = torch.dot(a, loss)

    # unroll the weights
    theta = _concat(self.model2.parameters()).data
    try:
      moment = _concat(optim_2.state[p]['momentum_buffer']
                       for p in self.model2.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(
      weighted_loss, self.model2.parameters())).data + self.network_weight_decay * theta
    unrolled_model_w2 = self._construct_model_from_theta(theta.sub(eta2, moment + dtheta))

    return unrolled_model_w2
    

  def step(self,
           input_train, target_train, input_valid, target_valid,
           eta, eta2, optim_1, optim_2,
           unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
          input_train, target_train, input_valid, target_valid, eta, eta2, optim_1, optim_2)
    else:
        pass
    self.optimizer.step()


  def _backward_step_unrolled(self,
                              input_train,
                              target_train,
                              input_valid,
                              target_valid,
                              eta, eta2,
                              optim_1, optim_2):
    unrolled_model_w1 = self._compute_unrolled_model(
        input_train, target_train, eta, optim_1)
    unrolled_model_w2 = self._compute_unrolled_model_w2(
      input_train, target_train, input_valid, target_valid, eta2, optim_2, unrolled_model_w1
    )
    unrolled_loss = unrolled_model_w2._loss(input_valid, target_valid)
    unrolled_loss.backward()

    dalpha = [v.grad for v in unrolled_model_w2.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model_w2.parameters()]
    implicit_grads = self._hessian_vector_product(vector, unrolled_model_w1,
                                                  input_train, target_train, input_valid, target_valid)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)


  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()


  def _hessian_vector_product(self, vector, unrolled_model_w1,
                              input_train, target_train, input_valid, target_valid,
                              r=1e-2):
    # epsilon
    R = r / _concat(vector).norm()

    # compute weights
    u = unrolled_model_w1._loss(input_valid, target_valid, reduction='none')

    z = [[1 if target_i == target_j else 0 for target_j in target_valid] for target_i in target_train]
    z = torch.FloatTensor(z).cuda()

    x = [[self.v(input_j) * self.v(input_i) for input_j in input_valid] for input_i in input_train]
    x = torch.FloatTensor(x).cuda()
    x = F.softmax(x, dim=1)  # compute softmax along with the rows

    a = F.sigmoid(torch.dot(x * z * u, self.r))

    # dα weighted Ltrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)
    loss = self.model2._loss(input_train, target_train, reduction='none')
    weighted_loss = torch.dot(a, loss)
    grads_p = torch.autograd.grad(weighted_loss, self.model.arch_parameters())

    # dα weighted Ltrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(2 * R, v)
    loss = self.model2._loss(input_train, target_train, reduction='none')
    weighted_loss = torch.dot(a, loss)
    grads_n = torch.autograd.grad(weighted_loss, self.model.arch_parameters())

    # change w- back to w
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]