"""Tests for staged sub-recipe support.

Tests cover:
- ApprovalGatePausedError.resume_session_id field for tracking child sessions
- _execute_recipe_step child session management (save/resume/cleanup)
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from amplifier_module_tool_recipes.executor import ApprovalGatePausedError
from amplifier_module_tool_recipes.executor import RecipeExecutor
from amplifier_module_tool_recipes.executor import RecursionState
from amplifier_module_tool_recipes.models import Step


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


# =============================================================================
# _execute_recipe_step Child Session Management Tests
# =============================================================================


def _make_executor():
    """Create a RecipeExecutor with minimal mocks."""
    coordinator = MagicMock()
    coordinator.session = MagicMock()
    coordinator.config = {"agents": {}}
    coordinator.hooks = None
    coordinator.get_capability.return_value = AsyncMock()

    session_manager = MagicMock()
    session_manager.create_session.return_value = "test-session-id"
    session_manager.is_cancellation_requested.return_value = False
    session_manager.is_immediate_cancellation.return_value = False

    return RecipeExecutor(coordinator, session_manager)


def _make_sub_recipe_file(tmp_path: Path) -> Path:
    """Create a minimal sub-recipe YAML file."""
    content = """\
name: sub-recipe
description: A sub-recipe for testing
version: "1.0.0"

steps: []
"""
    sub_recipe_path = tmp_path / "sub.yaml"
    sub_recipe_path.write_text(content)
    return sub_recipe_path


class TestExecuteRecipeStepChildSession:
    """Tests for child session management in _execute_recipe_step."""

    @pytest.mark.asyncio
    async def test_resume_saved_child_session(self):
        """Saved child session ID is passed to execute_recipe and cleaned up on success."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sub_recipe_path = _make_sub_recipe_file(tmp_path)

            executor = _make_executor()

            # Patch execute_recipe to return a successful result
            executor.execute_recipe = AsyncMock(return_value={"result": "done"})

            step = Step(
                id="call-sub",
                type="recipe",
                recipe=str(sub_recipe_path),
            )
            context = {"_child_session_call-sub": "saved-child-session-id"}
            recursion_state = RecursionState()

            await executor._execute_recipe_step(
                step=step,
                context=context,
                project_path=tmp_path,
                recursion_state=recursion_state,
                parent_recipe_path=None,
            )

            # The saved session ID should have been passed as session_id
            call_kwargs = executor.execute_recipe.call_args[1]
            assert call_kwargs["session_id"] == "saved-child-session-id"

            # The child session key should be cleaned up after success
            assert "_child_session_call-sub" not in context

    @pytest.mark.asyncio
    async def test_saves_child_session_id_on_approval_pause(self):
        """Child session ID is saved in context when ApprovalGatePausedError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sub_recipe_path = _make_sub_recipe_file(tmp_path)

            executor = _make_executor()

            # Patch execute_recipe to raise ApprovalGatePausedError
            child_error = ApprovalGatePausedError(
                session_id="child-paused-session",
                stage_name="review",
                approval_prompt="Approve?",
            )
            executor.execute_recipe = AsyncMock(side_effect=child_error)

            step = Step(
                id="call-sub",
                type="recipe",
                recipe=str(sub_recipe_path),
            )
            context: dict = {}
            recursion_state = RecursionState()

            with pytest.raises(ApprovalGatePausedError):
                await executor._execute_recipe_step(
                    step=step,
                    context=context,
                    project_path=tmp_path,
                    recursion_state=recursion_state,
                    parent_recipe_path=None,
                )

            # The child's session ID should be saved in context
            assert context.get("_child_session_call-sub") == "child-paused-session"
