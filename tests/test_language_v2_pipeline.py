import json
import pytest
from nmind.language.data_sources import DataSourceRegistry, LicenseDecision
from nmind.language.corpus_pipeline import build_records, clean_text, chinese_ratio, import_approved_file
from nmind.language.world import LanguageWorld
from nmind.language.curriculum import starter_tasks, run_oracle_task
from nmind.language.learning_v2 import TrainingPermissionError, require_training_authorization

def test_unapproved_source_is_rejected(tmp_path):
    reg = DataSourceRegistry(tmp_path / "sources.json")
    reg.register(LicenseDecision("x", "r1", "cmn", "https://x", review_status="REVIEW_REQUIRED"))
    with pytest.raises(PermissionError): reg.require_approved("x")

def test_approved_local_import_and_manifest(tmp_path):
    reg = DataSourceRegistry(tmp_path / "sources.json")
    reg.register(LicenseDecision("x", "r1", "cmn", "local", permitted_use="research", review_status="APPROVED"))
    p = tmp_path / "a.txt"; p.write_text("中文短句。\n\n中文短句。\n\nhello", encoding="utf-8")
    records = import_approved_file(reg, "x", p)
    assert len(records) == 2 and all(r.split == "train" for r in records)

def test_cleaning_redacts_sensitive_material():
    text, hits = clean_text("邮箱 a@b.com\napi_key=secret123\n电话 138 0013 8000")
    assert hits == {"email": 1, "phone": 1, "secret": 1} and "secret123" not in text and chinese_ratio(text) > 0

def test_duplicate_and_limit():
    records = build_records("x", ["甲", "甲", "乙"], token_limit=3)
    assert len(records) == 1

def test_world_oracle_is_independent_and_updates():
    world = LanguageWorld(); assert world.oracle("苹果在哪里？")["answer"] == "桌子"
    result = world.act("move", "苹果", "盒子")
    assert result["ok"] and world.oracle("苹果在哪里？")["answer"] == "盒子"

def test_invalid_action_fails_closed():
    assert LanguageWorld().act("run_command", "苹果")["ok"] is False

def test_curriculum_oracle_tasks():
    results = [run_oracle_task(t) for t in starter_tasks()]
    assert all(r["verified"] for r in results)

def test_training_gate_requires_explicit_positive_budget():
    with pytest.raises(TrainingPermissionError): require_training_authorization(authorized=False, max_steps=0)
    with pytest.raises(TrainingPermissionError): require_training_authorization(authorized=True, max_steps=0)
