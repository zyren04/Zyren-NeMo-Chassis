"""
Discriminated Union Input Models

Provides type-safe polymorphic inputs using Pydantic's Discriminator pattern.
Enables multiple input types (e.g., GitHub repo, local path, HTTP URL) in a
type-safe way instead of Optional fields with runtime checks.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag


class TypedBaseModel(BaseModel):
    """
    Base model with built-in discriminator support.

    Subclasses automatically get a 'type' field based on their class name.
    The discriminator method extracts the type from the data dict.
    """

    @classmethod
    def static_type(cls) -> str:
        """Get the static type identifier for this model."""
        return cls.__name__.lower().replace("input", "").replace("config", "")

    @classmethod
    def discriminator(cls, v: dict[str, Any]) -> str:
        """Extract the type discriminator from a data dict."""
        result = v.get("type", cls.static_type())
        return str(result) if result is not None else cls.static_type()


# Example: Repository source inputs
class GitHubRepoInput(TypedBaseModel):
    """Input for cloning from a GitHub repository."""

    type: Literal["github"] = "github"
    url: str = Field(..., description="GitHub repository URL (https://github.com/owner/repo)")
    branch: str | None = Field(
        default=None, description="Branch to clone (default: default branch)"
    )
    token: str | None = Field(default=None, description="GitHub token for private repos")


class GitLabRepoInput(TypedBaseModel):
    """Input for cloning from a GitLab repository."""

    type: Literal["gitlab"] = "gitlab"
    url: str = Field(..., description="GitLab repository URL")
    branch: str | None = Field(default=None, description="Branch to clone")
    token: str | None = Field(default=None, description="GitLab token for private repos")


class LocalPathInput(TypedBaseModel):
    """Input for using a local filesystem path."""

    type: Literal["local"] = "local"
    path: str = Field(..., description="Absolute or relative path to local directory")


class HttpUrlInput(TypedBaseModel):
    """Input for downloading from an HTTP/HTTPS URL."""

    type: Literal["http"] = "http"
    url: str = Field(..., description="HTTP/HTTPS URL to download")
    headers: dict[str, str] = Field(default_factory=dict, description="Optional HTTP headers")


# Discriminated union for repository sources
RepoSourceInput = Annotated[
    Annotated[GitHubRepoInput, Tag("github")]
    | Annotated[GitLabRepoInput, Tag("gitlab")]
    | Annotated[LocalPathInput, Tag("local")]
    | Annotated[HttpUrlInput, Tag("http")],
    Discriminator(TypedBaseModel.discriminator),
]


# Example: LLM provider inputs
class NIMModelInput(TypedBaseModel):
    """Input for NVIDIA NIM model."""

    type: Literal["nim"] = "nim"
    model: str = Field(..., description="NIM model identifier (e.g., nvidia/nemotron-3-ultra)")
    api_key: str | None = Field(default=None, description="NVIDIA API key")
    base_url: str | None = Field(default=None, description="Custom NIM endpoint URL")
    max_rpm: int = Field(default=40, description="Max requests per minute")
    rate_limit_mode: str = Field(
        default="token_bucket", description="Rate limit mode: token_bucket or strict"
    )


class OpenAIModelInput(TypedBaseModel):
    """Input for OpenAI-compatible model."""

    type: Literal["openai"] = "openai"
    model: str = Field(..., description="Model name (e.g., gpt-4)")
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str | None = Field(default=None, description="Custom OpenAI endpoint URL")


class LocalModelInput(TypedBaseModel):
    """Input for local model (e.g., Ollama, llama.cpp)."""

    type: Literal["local"] = "local"
    model: str = Field(..., description="Model identifier")
    base_url: str = Field(default="http://localhost:11434", description="Local model server URL")


# Discriminated union for LLM provider inputs
LLMProviderInput = Annotated[
    Annotated[NIMModelInput, Tag("nim")]
    | Annotated[OpenAIModelInput, Tag("openai")]
    | Annotated[LocalModelInput, Tag("local")],
    Discriminator(TypedBaseModel.discriminator),
]


# Example: Output format inputs
class JsonOutputInput(TypedBaseModel):
    """Input for JSON output format."""

    type: Literal["json"] = "json"
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for validation"
    )


class MarkdownOutputInput(TypedBaseModel):
    """Input for Markdown output format."""

    type: Literal["markdown"] = "markdown"
    template: str | None = Field(default=None, description="Optional Markdown template")


class TextOutputInput(TypedBaseModel):
    """Input for plain text output format."""

    type: Literal["text"] = "text"
    prefix: str = Field(default="", description="Prefix for output")
    suffix: str = Field(default="", description="Suffix for output")


# Discriminated union for output formats
OutputFormatInput = Annotated[
    Annotated[JsonOutputInput, Tag("json")]
    | Annotated[MarkdownOutputInput, Tag("markdown")]
    | Annotated[TextOutputInput, Tag("text")],
    Discriminator(TypedBaseModel.discriminator),
]


__all__ = [
    "TypedBaseModel",
    "GitHubRepoInput",
    "GitLabRepoInput",
    "LocalPathInput",
    "HttpUrlInput",
    "RepoSourceInput",
    "NIMModelInput",
    "OpenAIModelInput",
    "LocalModelInput",
    "LLMProviderInput",
    "JsonOutputInput",
    "MarkdownOutputInput",
    "TextOutputInput",
    "OutputFormatInput",
]
