"""
Environment-aware settings for the ED Triage backend.

Uses pydantic-settings BaseSettings so values can be overridden with env vars
or a .env file at the project root.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    sagemaker_endpoint_name: str = "edtriage-live"
    aws_region: str = "us-east-1"
    aws_profile: str = ""
    use_mock: bool = False
    default_model: str = "arch4"

    model_config = {
        "env_prefix": "TRIAGE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
