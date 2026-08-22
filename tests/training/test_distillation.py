"""Tests for Knowledge Distillation loss (RIL TASK-221 / DEC-055)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.training.distillation import DistillationLoss


def test_kl_vanishes_when_logits_match():
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 10)
    labels = torch.randint(0, 10, (4, 7))
    loss = DistillationLoss(temperature=3.0, alpha=0.5)(logits, labels, teacher_logits=logits)
    # student == teacher -> KL == 0; loss is just alpha*CE (still positive).
    ce = nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, 10), labels.reshape(-1))
    assert abs(loss.item() - 0.5 * ce.item()) < 1e-5


def test_alpha_ce_only_endpoint():
    torch.manual_seed(1)
    s = torch.randn(3, 5, 8)
    t = torch.randn(3, 5, 8)
    labels = torch.randint(0, 8, (3, 5))
    loss = DistillationLoss(temperature=2.0, alpha=1.0)(s, labels, teacher_logits=t)
    ce = nn.CrossEntropyLoss(ignore_index=-100)(s.reshape(-1, 8), labels.reshape(-1))
    assert abs(loss.item() - ce.item()) < 1e-6


def test_alpha_kd_only_endpoint_and_manual_kl():
    torch.manual_seed(2)
    temp, alpha = 3.0, 0.0
    s = torch.randn(2, 6, 5)
    t = torch.randn(2, 6, 5)
    labels = torch.randint(0, 5, (2, 6))
    loss = DistillationLoss(temperature=temp, alpha=alpha)(s, labels, teacher_logits=t)
    s_logp = torch.log_softmax(s.reshape(-1, 5) / temp, dim=-1)
    t_p = torch.softmax(t.reshape(-1, 5) / temp, dim=-1)
    manual = functional.kl_div(s_logp, t_p, reduction="batchmean") * temp**2
    assert abs(loss.item() - manual.item()) < 1e-5


def test_gradient_flows_only_to_student():
    torch.manual_seed(3)
    student = nn.Linear(6, 5)
    teacher = nn.Linear(6, 5)
    x = torch.randn(4, 6)
    labels = torch.randint(0, 5, (4,))
    lossfn = DistillationLoss(temperature=2.0, alpha=0.5, teacher=teacher)
    loss = lossfn(student(x), labels, inputs=x)
    loss.backward()
    assert all(p.grad is not None for p in student.parameters()), "student must get gradients"
    assert all(p.grad is None for p in teacher.parameters()), "teacher must NOT get gradients"


def test_ignore_index_masks_ce_and_kl():
    torch.manual_seed(4)
    s = torch.randn(2, 6, 4)
    t = torch.randn(2, 6, 4)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    # All-ignored labels -> CE == 0 and KL == 0 (rank-ignored positions).
    loss = DistillationLoss(temperature=2.0, alpha=0.5)(s, labels, teacher_logits=t)
    assert loss.item() == 0.0


def test_invalid_args_raise():
    for kw in ({"temperature": 0.0}, {"temperature": -1.0}, {"alpha": 1.5}, {"alpha": -0.1}):
        try:
            DistillationLoss(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"DistillationLoss({kw}) should have raised")


def test_teacher_module_path_matches_explicit_logits():
    torch.manual_seed(5)
    teacher = nn.Linear(6, 5)
    student = nn.Linear(6, 5)
    x = torch.randn(3, 6)
    labels = torch.randint(0, 5, (3,))
    loss_module = DistillationLoss(temperature=2.0, alpha=0.5, teacher=teacher)
    via_inputs = loss_module(student(x), labels, inputs=x)
    with torch.no_grad():
        tl = teacher(x)
    via_explicit = loss_module(student(x), labels, teacher_logits=tl)
    assert abs(via_inputs.item() - via_explicit.item()) < 1e-6
