"""The H2b / S6 configs compose and declare exactly their variable."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs" / "model")


def _compose(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=name)


def test_joint_arm_declares_only_the_joint_loss():
    base = _compose("lotsa_mini_v3_head8_zeroshot")
    joint = _compose("lotsa_mini_v3_head8_joint_zeroshot")
    assert joint.model.name == "timejepa_lotsa_mini_v3_head8_joint_zs"
    assert joint.training.loss.lambda_joint > 0 and joint.training.loss.joint_target == "frozen"
    assert joint.training.loss.joint_sigreg is True and joint.training.loss.get("sigreg")
    assert float(joint.training.loss.get("lambda_anchor", 0.0)) == 0.0
    assert list(joint.training.loss.get("critic_steps") or []) == []
    assert joint.model.decoder.quantile_hidden_dim == base.model.decoder.quantile_hidden_dim
    assert joint.data.data_dir == base.data.data_dir


def test_critic_arm_inherits_joint_and_declares_the_loop():
    critic = _compose("lotsa_mini_v3_head8_critic_zeroshot")
    L = critic.training.loss
    assert critic.model.name == "timejepa_lotsa_mini_v3_head8_critic_zs"
    assert L.lambda_joint > 0
    assert list(L.critic_steps) == [0, 1, 2, 3, 4]
    assert L.critic_route == "A" and L.critic_target == "center"
    assert 0 < L.critic_batch_fraction <= 1 and L.critic_alpha > 0


@pytest.mark.parametrize("name,expected", [
    ("lotsa_mini_v3_head8_joint_eval", "timejepa_lotsa_mini_v3_head8_joint_zs"),
    ("lotsa_mini_v3_head8_critic_eval", "timejepa_lotsa_mini_v3_head8_critic_zs"),
])
def test_eval_twins_share_the_namespace(name, expected):
    c = _compose(name)
    assert c.model.name == expected and c.model.decoder.quantile_hidden_dim == 1536
