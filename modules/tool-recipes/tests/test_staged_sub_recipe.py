"""Tests for staged sub-recipe support.

Tests cover:
- ApprovalGatePausedError.resume_session_id field for tracking child sessions
"""

import pytest
from amplifier_module_tool_recipes.executor import ApprovalGatePausedError


# =============================================================================
# ApprovalGatePausedError resume_session_id Tests
# =============================================================================


class TestApprovalGatePausedErrorResumeSessionId:
    """Tests for the resume_session_id field on ApprovalGatePausedError."""

    def test_resume_session_id_defaults_to_none(self):
        """resume_session_id should default to None when not provided."""
        error = ApprovalGatePausedError(
            session_id="x",
            stage_name="y",
            approval_prompt="z",
        )
        assert error.resume_session_id is None

    def test_resume_session_id_preserved_when_set(self):
        """resume_session_id should preserve the value when explicitly set."""
        error = ApprovalGatePausedError(
            session_id="parent-session",
            stage_name="planning",
            approval_prompt="Approve?",
            resume_session_id="child-session",
        )
        assert error.resume_session_id == "child-session"

    def test_existing_fields_unchanged_when_resume_session_id_provided(self):
        """Existing fields remain unchanged when resume_session_id is provided."""
        error = ApprovalGatePausedError(
            session_id="parent-123",
            stage_name="execution",
            approval_prompt="Do you approve the execution?",
            resume_session_id="child-456",
        )
        assert error.session_id == "parent-123"
        assert error.stage_name == "execution"
        assert error.approval_prompt == "Do you approve the execution?"
        assert error.resume_session_id == "child-456"
