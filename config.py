from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model_main: str = Field(default="gpt-4.1-mini", alias="MODEL_MAIN")
    model_eval: str = Field(default="gpt-4.1-nano", alias="MODEL_EVAL")
    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")
    eval_goal: str = Field(
        default="Provide a concise, source-grounded recommendation for insurance operations.",
        alias="EVAL_GOAL",
    )
    eval_output_mode: Literal["executive", "analyst"] = Field(
        default="analyst",
        alias="EVAL_OUTPUT_MODE",
    )


settings = Settings()
