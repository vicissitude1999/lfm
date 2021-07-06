import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import utils


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model1, model2, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model1 = model1
    self.model2 = model2
    self.optimizer = torch.optim.Adam(self.model2.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self,
           input_train, target_train,
           input_valid, target_valid,
           input_mixed, target_a, target_b, lam,
           eta, eta2, optim_1, optim_2,
           unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
          input_train, target_train,
          input_valid, target_valid,
          input_mixed, target_a, target_b, lam,
          eta, eta2, optim_1, optim_2)
    else:
        pass
    self.optimizer.step()


  def _backward_step_unrolled(self,
                              input_train, target_train,
                              input_valid, target_valid,
                              input_mixed, target_a, target_b, lam,
                              eta, eta2, optim_1, optim_2):
    unrolled_model_w1 = self._compute_unrolled_model(
        input_train, target_train, eta, optim_1)
    unrolled_model_w2, weights = self._compute_unrolled_model_w2(
      input_train,
      input_mixed, target_a, target_b, lam,
      eta2, optim_2, unrolled_model_w1
    )
    unrolled_loss = unrolled_model_w2._loss(input_valid, target_valid)
    unrolled_loss.backward()

    dalpha = [v.grad for v in unrolled_model_w2.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model_w2.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, weights)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(ig.data, alpha=eta)

    for v, g in zip(self.model2.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)


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
    unrolled_model_w1 = self._construct_model_from_theta(self.model1, theta.sub(moment + dtheta, alpha=eta))

    return unrolled_model_w1


  # valid data are used to compute weights
  def _compute_unrolled_model_w2(self, input_train, # target_train same as target_a
                                 input_mixed, target_a, target_b, lam,
                                 eta2, optim_2, unrolled_model_w1):
    loss_w1_unrolled = unrolled_model_w1._loss(input_train, target_a, reduction='none')
    loss_mixed = self.model2._loss(input_mixed, target_b, reduction='none')
    weights = utils.compute_weighted_loss(loss_w1_unrolled, target_a, target_b, lam)
    weighted_loss_mixed = torch.dot(weights, loss_mixed)

    # unroll the weights
    theta = _concat(self.model2.parameters()).data
    try:
      moment = _concat(optim_2.state[p]['momentum_buffer']
                       for p in self.model2.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(
      weighted_loss_mixed, self.model2.parameters())).data + self.network_weight_decay * theta
    unrolled_model_w2 = self._construct_model_from_theta(self.model2, theta.sub(moment + dtheta, alpha=eta2))

    return unrolled_model_w2, weights


  def _hessian_vector_product(self, vector, input_train, target_train, w, r=1e-2):
    # epsilon
    R = r / _concat(vector).norm()

    # dα weighted Ltrain(w+,α)
    for p, v in zip(self.model2.parameters(), vector):
        p.data.add_(v, alpha=R)
    loss = self.model2._loss(input_train, target_train, reduction='none')
    weighted_loss = torch.dot(w, loss)
    grads_p = torch.autograd.grad(weighted_loss, self.model2.arch_parameters())

    # dα weighted Ltrain(w-,α)
    for p, v in zip(self.model2.parameters(), vector):
        p.data.sub_(v, alpha=2*R)
    loss = self.model2._loss(input_train, target_train, reduction='none')
    weighted_loss = torch.dot(w, loss)
    grads_n = torch.autograd.grad(weighted_loss, self.model2.arch_parameters())

    # change w- back to w
    for p, v in zip(self.model2.parameters(), vector):
        p.data.add_(v, alpha=R)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


  def _construct_model_from_theta(self, model, theta):
    model_new = model.new()
    model_dict = model.state_dict()

    params, offset = {}, 0
    for k, v in model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()