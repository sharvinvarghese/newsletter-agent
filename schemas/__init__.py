"""Pydantic schemas used across the agent.

Everything the LLM returns flows through one of these models and everything the
application stores in the LangGraph state is one of these models - never a bare
dictionary.
"""

from schemas.articles import (
    ArticleEvaluation,
    ArticleEvaluationBatch,
    ArticleEvaluationItem,
    ArticleSummary,
    ArticleSummaryBatch,
    DuplicateRecord,
    EvaluatedArticle,
    NewsArticle,
    ResearchOutcome,
    ResearchStats,
)
from schemas.base import (
    AgentMode,
    AgentModel,
    CleanStr,
    HttpUrlStr,
    IsoDateTime,
    UnitScore,
)
from schemas.critique import CritiqueResult, HumanApproval
from schemas.goal import AgentPlan, GoalSpec
from schemas.newsletter import Newsletter, NewsletterSection
from schemas.result import AgentEvent, AgentResult

__all__ = [
    "AgentEvent",
    # base
    "AgentMode",
    "AgentModel",
    "AgentPlan",
    # results
    "AgentResult",
    "ArticleEvaluation",
    "ArticleEvaluationBatch",
    "ArticleEvaluationItem",
    "ArticleSummary",
    "ArticleSummaryBatch",
    "CleanStr",
    # critique & approval
    "CritiqueResult",
    "DeduplicationResult",
    "DuplicateRecord",
    "EvaluatedArticle",
    # goal & plan
    "GoalSpec",
    "HttpUrlStr",
    "HumanApproval",
    "IsoDateTime",
    # articles
    "NewsArticle",
    "Newsletter",
    # newsletter
    "NewsletterSection",
    "ResearchOutcome",
    "ResearchStats",
    "UnitScore",
]