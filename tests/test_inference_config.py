"""
Tests for the inference configuration system.

Validates YAML config loading, InferenceConfig building,
and backward compatibility (config=None doesn't break existing behavior).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.test_models import (
    load_model_profiles,
    load_prompt_templates,
    build_inference_config,
    InferenceConfig,
    InferenceResult,
    MODEL_NAME_MAP,
)


class TestModelProfilesLoading:
    """Tests for config/model_profiles.yaml loading."""

    def test_load_model_profiles_returns_dict(self):
        profiles = load_model_profiles()
        assert isinstance(profiles, dict)

    def test_all_expected_models_present(self):
        profiles = load_model_profiles()
        expected = {"medgemma_27b", "medgemma_4b", "hulumed", "gemma3"}
        assert expected == set(profiles.keys())

    def test_each_model_has_required_fields(self):
        profiles = load_model_profiles()
        for name, cfg in profiles.items():
            assert "model_id" in cfg, f"{name} missing model_id"
            assert "backend" in cfg, f"{name} missing backend"
            assert "image_max_size" in cfg, f"{name} missing image_max_size"
            assert "profiles" in cfg, f"{name} missing profiles"

    def test_each_model_has_default_profile(self):
        profiles = load_model_profiles()
        for name, cfg in profiles.items():
            assert "default" in cfg["profiles"], f"{name} missing default profile"

    def test_registry_name_matches_map(self):
        profiles = load_model_profiles()
        for name, cfg in profiles.items():
            expected_registry = MODEL_NAME_MAP.get(name)
            assert cfg.get("registry_name") == expected_registry, \
                f"{name}: registry_name {cfg.get('registry_name')} != expected {expected_registry}"

    def test_backend_values_valid(self):
        profiles = load_model_profiles()
        valid_backends = {"transformers", "ollama"}
        for name, cfg in profiles.items():
            assert cfg["backend"] in valid_backends, \
                f"{name} has invalid backend: {cfg['backend']}"

    def test_transformers_models_have_generate_params(self):
        profiles = load_model_profiles()
        for name, cfg in profiles.items():
            if cfg["backend"] == "transformers":
                for prof_name, prof in cfg["profiles"].items():
                    assert "generate_params" in prof, \
                        f"{name}/{prof_name} missing generate_params"

    def test_ollama_models_have_backend_options(self):
        profiles = load_model_profiles()
        for name, cfg in profiles.items():
            if cfg["backend"] == "ollama":
                for prof_name, prof in cfg["profiles"].items():
                    assert "backend_options" in prof, \
                        f"{name}/{prof_name} missing backend_options"

    def test_hulumed_must_have_do_sample_true(self):
        """Hulu-Med MUST use do_sample=True to avoid premature EOS."""
        profiles = load_model_profiles()
        for prof_name, prof in profiles["hulumed"]["profiles"].items():
            gen = prof.get("generate_params", {})
            assert gen.get("do_sample") is True, \
                f"hulumed/{prof_name}: do_sample must be True (greedy causes EOS)"

    def test_image_max_size_values(self):
        profiles = load_model_profiles()
        assert profiles["hulumed"]["image_max_size"] == 512
        assert profiles["medgemma_27b"]["image_max_size"] == 896
        assert profiles["medgemma_4b"]["image_max_size"] == 896
        assert profiles["gemma3"]["image_max_size"] == 896


class TestPromptTemplatesLoading:
    """Tests for config/prompts.yaml loading."""

    def test_load_prompt_templates_returns_dict(self):
        prompts = load_prompt_templates()
        assert isinstance(prompts, dict)

    def test_clinician_v1_exists(self):
        prompts = load_prompt_templates()
        assert "clinician_v1" in prompts

    def test_each_template_has_system_and_suffix(self):
        prompts = load_prompt_templates()
        for name, cfg in prompts.items():
            assert "system" in cfg, f"{name} missing system prompt"
            assert "suffix" in cfg, f"{name} missing suffix"

    def test_clinician_v1_contains_clinician_framing(self):
        """System prompt must frame as clinician to avoid MedGemma refusal."""
        prompts = load_prompt_templates()
        system = prompts["clinician_v1"]["system"]
        assert "clinician" in system.lower() or "doctor" in system.lower()

    def test_hulumed_thinking_has_step_by_step_suffix(self):
        prompts = load_prompt_templates()
        if "hulumed_thinking" in prompts:
            suffix = prompts["hulumed_thinking"]["suffix"]
            assert "step by step" in suffix.lower()


class TestBuildInferenceConfig:
    """Tests for build_inference_config() merge logic."""

    def test_build_default_config(self):
        config = build_inference_config("medgemma_27b")
        assert isinstance(config, InferenceConfig)
        assert config.model_name == "medgemma_27b"
        assert config.profile_name == "default"
        assert config.prompt_template == "clinician_v1"
        assert config.backend == "transformers"

    def test_build_with_specific_profile(self):
        config = build_inference_config("medgemma_27b", profile_name="creative")
        assert config.profile_name == "creative"
        assert config.generate_params.get("do_sample") is True
        assert config.generate_params.get("temperature") == 0.7

    def test_build_with_specific_prompt(self):
        config = build_inference_config("medgemma_27b", prompt_template="structured_output")
        assert config.prompt_template == "structured_output"
        assert "structured format" in config.user_suffix.lower() or "format" in config.system_prompt.lower()

    def test_build_hulumed_default_has_decode_params(self):
        config = build_inference_config("hulumed")
        assert "use_think" in config.decode_params
        assert config.decode_params["use_think"] is False

    def test_build_hulumed_thinking_profile(self):
        config = build_inference_config("hulumed", profile_name="thinking")
        assert config.decode_params.get("use_think") is True
        assert config.generate_params.get("max_new_tokens") == 4096

    def test_build_gemma3_has_backend_options(self):
        config = build_inference_config("gemma3")
        assert config.backend == "ollama"
        assert config.registry_name is None
        assert isinstance(config.backend_options, dict)

    def test_build_gemma3_creative_has_options(self):
        config = build_inference_config("gemma3", profile_name="creative")
        assert config.backend_options.get("temperature") == 0.8

    def test_registry_name_set_correctly(self):
        assert build_inference_config("medgemma_27b").registry_name == "medgemma"
        assert build_inference_config("medgemma_4b").registry_name == "mg4b"
        assert build_inference_config("hulumed").registry_name == "hulumed"
        assert build_inference_config("gemma3").registry_name is None

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_inference_config("nonexistent_model")

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            build_inference_config("medgemma_27b", profile_name="nonexistent")

    def test_unknown_prompt_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt template"):
            build_inference_config("medgemma_27b", prompt_template="nonexistent")

    def test_system_prompt_not_empty(self):
        config = build_inference_config("medgemma_27b")
        assert len(config.system_prompt.strip()) > 0

    def test_image_max_size_propagated(self):
        config = build_inference_config("hulumed")
        assert config.image_max_size == 512


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_create_result(self):
        result = InferenceResult(
            model_name="medgemma_27b",
            profile_name="default",
            prompt_template="clinician_v1",
            output="Test output",
            inference_time_sec=1.5,
        )
        assert result.model_name == "medgemma_27b"
        assert result.error is None

    def test_result_with_error(self):
        result = InferenceResult(
            model_name="hulumed",
            profile_name="default",
            prompt_template="clinician_v1",
            output="Error: OOM",
            inference_time_sec=0.1,
            error="CUDA OOM",
        )
        assert result.error == "CUDA OOM"


class TestBackwardCompatibility:
    """Verify that config=None doesn't break existing analyze_with_* signatures."""

    def test_analyze_with_gemma3_accepts_no_config(self):
        """Verify the function signature accepts config=None (default)."""
        import inspect
        from src.engine.test_models import analyze_with_gemma3
        sig = inspect.signature(analyze_with_gemma3)
        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_analyze_with_medgemma_accepts_no_config(self):
        import inspect
        from src.engine.test_models import analyze_with_medgemma
        sig = inspect.signature(analyze_with_medgemma)
        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_analyze_with_hulumed_accepts_no_config(self):
        import inspect
        from src.engine.test_models import analyze_with_hulumed
        sig = inspect.signature(analyze_with_hulumed)
        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_analyze_with_medgemma_4b_accepts_no_config(self):
        import inspect
        from src.engine.test_models import analyze_with_medgemma_4b
        sig = inspect.signature(analyze_with_medgemma_4b)
        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_run_inference_requires_processor_model_for_transformers(self):
        """run_inference should raise ValueError if transformers backend without processor/model."""
        from src.engine.test_models import run_inference
        config = build_inference_config("medgemma_27b")
        with pytest.raises(ValueError, match="requires processor and model"):
            run_inference("dummy.jpg", "test", config)
