from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pytest
import torch

from m2_conduccion.alert_policy import choose_threshold, threshold_table
from m2_conduccion.data.crop_audit import crop, cross_partition_pairs
from m2_conduccion.data.duplicates_check import fingerprints, perceptual_pairs
from m2_conduccion.eda import audit
from m2_conduccion.predict import classify
from m2_conduccion.temporal import TemporalAlert
from m2_conduccion.train import build

GROUPING = {"sin_alerta": ["segura", "giro"], "con_alerta": ["escribir", "hablar", "otros"]}


def test_classify_operational_checkpoint(tmp_path):
    params = {"width": 4, "depth": 2, "dropout": 0.0}
    model = build("scratch", params)
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([3.0, 0.0, 2.9, 2.9, 0.0]))
    checkpoint = tmp_path / "model.pt"
    torch.save({"family": "scratch", "parameters": params, "state_dict": model.state_dict(),
                "threshold": 0.5, "temperature": 1.0, "class_grouping": GROUPING,
                "image_size": 32}, checkpoint)
    image = BytesIO()
    Image.new("RGB", (50, 40), "white").save(image, format="PNG")
    image.seek(0)
    result = classify(image, checkpoint=checkpoint)
    assert result["clase"] == "segura"
    assert sum(result["probabilidades"].values()) == pytest.approx(1)
    assert result["alerta"]
    assert result["gradcam"].shape == (32, 32)
    assert np.isfinite(result["gradcam"]).all()
    assert not classify(image, 1, checkpoint=checkpoint)["alerta"]


def test_perceptual_audit_keeps_mirrors_together():
    rng = np.random.default_rng(22)
    pictures = [Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)) for _ in range(25)]
    pictures[1] = ImageOps.mirror(pictures[0])
    hashes = [fingerprints(image) for image in pictures]
    assert perceptual_pairs(hashes, 0)[(0, 1)] == 0
    frame = pd.DataFrame({"label": np.repeat(np.arange(5), 5), "sha256": list(range(25))})
    summary, _ = audit(frame, hashes, {"hash_distance": 0, "split": [0.6, 0.2, 0.2], "seed": 2})
    assert summary["cross_split_matches"] == 0
    assert frame.loc[0, "split"] == frame.loc[1, "split"]


def test_alert_table_counts_both_error_types():
    cfg = {"class_grouping": GROUPING, "default_threshold": 0.5, "false_alarm_cost": 1, "miss_cost": 5}
    logits = np.array([[4, 0, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 4, 0, 0]])
    labels = np.array([2, 1, 2])
    table = threshold_table(logits, labels, 1.0, cfg)
    at_half = table[table.threshold == 0.5].iloc[0]
    assert (at_half.fp, at_half.fn) == (1, 1)
    assert at_half.fp_per_1000 == pytest.approx(1000 / 3)
    assert table.loc[table.threshold == choose_threshold(table), "fn"].iloc[0] == 0


def test_corner_crop_is_found_across_partitions():
    original = Image.fromarray(np.random.default_rng(7).integers(0, 256, (64, 64), dtype=np.uint8))
    derivative = crop(original, 0.80, (0, 1))
    rows = cross_partition_pairs([original, derivative], np.array(["train", "test"]), 7, 0.975)
    assert len(rows) == 1 and rows[0][:2] == (0, 1)
    assert cross_partition_pairs([original, derivative], np.array(["train", "train"]), 7, 0.975) == []


def test_temporal_alert_requires_persistence_and_respects_cooldown():
    policy = TemporalAlert(0.4)
    first = [policy.update(i / 5, risk) for i, risk in enumerate([0.5, 0.2, 0.5, 0.2, 0.5])]
    assert first == [False, False, False, False, True]
    assert not any(policy.update(i / 5, 0.9) for i in range(5, 10))
    assert not policy.update(21, 0.9, moving=False)
    assert not any(policy.update(22 + i / 5, 0.9) for i in range(4))
    assert policy.update(23, 0.9)


def test_resnet_frozen_prefix_preserves_logits():
    net = build("transfer", {"dropout": 0.4, "unfreeze": ["layer4"]}).eval()
    data = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(net(data), net.forward_from_prefix(net.through_layer3(data)))
