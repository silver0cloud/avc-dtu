import torch

from avavs.models.baselines import ImageBindProjection, NaiveLateFusionMLP
from avavs.models.fusion_mlp import FusionMLP, regression_loss
from avavs.models.student_teacher import StudentMLP, TeacherProbe, distillation_loss


def test_fusion_mlp_output_shape_and_norm():
    model = FusionMLP(audio_dim=128, latent_dim=512)
    audio = torch.randn(4, 128)
    out = model(audio)
    assert out.shape == (4, 512)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)


def test_regression_loss_is_finite_and_zero_for_identical_vectors():
    pred = torch.randn(4, 512)
    pred = pred / pred.norm(dim=-1, keepdim=True)
    loss = regression_loss(pred, pred)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_naive_late_fusion_shape():
    model = NaiveLateFusionMLP(audio_dim=128, visual_dim=512, latent_dim=512)
    out = model(torch.randn(2, 128), torch.randn(2, 512))
    assert out.shape == (2, 512)


def test_imagebind_projection_shape():
    model = ImageBindProjection(audio_dim=128, visual_dim=512, latent_dim=512)
    out = model(torch.randn(3, 128), torch.randn(3, 512))
    assert out.shape == (3, 512)


def test_student_teacher_shapes_and_distillation_loss():
    teacher = TeacherProbe(visual_dim=512, num_classes=10)
    student = StudentMLP(audio_dim=128, visual_dim=512, latent_dim=512, num_classes=10)

    audio, visual = torch.randn(5, 128), torch.randn(5, 512)
    student_emb, student_logits = student(audio, visual)
    teacher_logits = teacher(visual)

    assert student_emb.shape == (5, 512)
    assert student_logits.shape == (5, 10)
    assert teacher_logits.shape == (5, 10)

    loss = distillation_loss(student_emb, student_logits, student_emb.detach(), teacher_logits.detach())
    assert torch.isfinite(loss)
