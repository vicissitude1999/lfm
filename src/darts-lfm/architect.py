import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, model_rw, model_beta, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        
        self.model = model
        self.model_rw = model_rw
        self.model_beta = model_beta
        
        self.optimizer_a = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.optimizer_beta = torch.optim.Adam(self.model_beta.parameters(),
                                                lr=args.learning_rate_beta, betas=(0.5, 0.999))

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _compute_reweighted_unrolled_model(self, input, target, weights, eta, network_optimizer):
        logits = self.model_rw(input)
        loss = F.cross_entropy(logits, target, reduction='none')
        loss = torch.dot(loss, weights)
        theta = _concat(self.model_rw.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model_rw.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.model_rw.parameters())).data + self.network_weight_decay * theta
        unrolled_model_rw = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))
        return unrolled_model_rw

    def step(self, input_train, target_train, input_valid, target_valid,
             eta, eta_rw, network_optimizer, network_optimizer_rw, unrolled):
        self.optimizer_a.zero_grad()
        self.optimizer_beta.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid,
                                         eta, eta_rw, network_optimizer, network_optimizer_rw)
        else:
            logits = self.model(input_valid)
            logits_rw = self.model_rw(input_valid)
            output = self.model_beta(logits, logits_rw)
            valid_loss = F.cross_entropy(output, target_valid)
            valid_loss.backward()
        self.optimizer_a.step()
        self.optimizer_beta.step()

    # the following functions are used for when unrolled = True
    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid,
                                eta, eta_rw, network_optimizer, network_optimizer_rw):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)

        with torch.no_grad():
            logits = unrolled_model(input_train)
            weights = F.cross_entropy(logits, target_train, reduction='none')
            weights = weights / weights.sum()

        reweighted_unrolled_model = self._compute_reweighted_unrolled_model(
            input_train, target_train, weights, eta_rw, network_optimizer_rw)

        # Validation stage loss function
        logits = unrolled_model(input_valid)
        logits_rw = reweighted_unrolled_model(input_valid)
        output = self.model_beta(logits, logits_rw)
        valid_loss = F.cross_entropy(output, target_valid)
        valid_loss.backward()

        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        dalpha_rw = [
            v.grad for v in reweighted_unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        vector_rw = [
            v.grad.data for v in reweighted_unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train)
        implicit_grads_rw = self._hessian_vector_product_rw(
            vector_rw, input_train, target_train, weights)

        for g, g_rw in zip(dalpha, dalpha_rw):
            g.data.add_(g_rw)

        for g, ig, ig_rw in zip(dalpha, implicit_grads, implicit_grads_rw):
            g.data.sub_(eta, ig.data)
            g.data.sub_(eta_rw, ig_rw.data)

        for v, v_rw, g in zip(self.model.arch_parameters(), self.model_rw.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
            if v_rw.grad is None:
                v_rw.grad = Variable(g.data)
            else:
                v_rw.grad.data.copy_(g.data)

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

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_rw(self, vector, input, target, weights, r=1e-2):
        R = r / _concat(vector).norm()
        # print(R)
        for p, v in zip(self.model_rw.parameters(), vector):
            p.data.add_(R, v)
        logits = self.model_rw(input)
        loss = F.cross_entropy(logits, target, reduction='none')
        loss = torch.dot(loss, weights)
        grads_p = torch.autograd.grad(
            loss, self.model_rw.arch_parameters())

        for p, v in zip(self.model_rw.parameters(), vector):
            p.data.sub_(2 * R, v)
        logits = self.model_rw(input)
        loss = F.cross_entropy(logits, target, reduction='none')
        loss = torch.dot(loss, weights)
        grads_n = torch.autograd.grad(
            loss, self.model_rw.arch_parameters())

        for p, v in zip(self.model_rw.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
