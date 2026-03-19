"""Tests for recipe event delivery fixes.

Verifies two bugs are fixed:
1. observability.events registration in mount() — so hooks-logging discovers recipe events
2. _show_progress uses await hooks.emit() instead of fire-and-forget asyncio.create_task
"""

import inspect
import re
from unittest.mock import MagicMock

import pytest

from amplifier_module_tool_recipes.executor import RecipeExecutor


EXPECTED_RECIPE_EVENTS = [
    "recipe:start",
    "recipe:step",
    "recipe:complete",
    "recipe:approval",
    "recipe:loop_iteration",
    "recipe:loop_complete",
]


class TestMountRegistersObservabilityEvents:
    """Verify mount() contributes recipe events to observability.events."""

    @pytest.mark.asyncio
    async def test_mount_contributes_observability_events(self):
        """mount() must contribute all recipe lifecycle events via coordinator.contribute()
        so that hook modules discover them via collect_contributions(), regardless of
        module mount order."""
        from amplifier_module_tool_recipes import mount

        coordinator = MagicMock()
        coordinator.mount_points = {"tools": {}}

        await mount(coordinator, config=None)

        # Verify contribute was called with observability.events and all recipe events
        coordinator.contribute.assert_called_once_with(
            "observability.events", EXPECTED_RECIPE_EVENTS
        )

    @pytest.mark.asyncio
    async def test_mount_does_not_use_register_capability_for_events(self):
        """mount() must not use register_capability for observability.events.

        register_capability is order-dependent — if the hook module mounts before
        the recipes module, it reads an empty list and never discovers recipe events.
        coordinator.contribute() is the correct multi-provider pattern.
        """
        from amplifier_module_tool_recipes import mount

        coordinator = MagicMock()
        coordinator.mount_points = {"tools": {}}

        await mount(coordinator, config=None)

        # register_capability must NOT be called for observability.events
        for call in coordinator.register_capability.call_args_list:
            args = call[0]
            assert args[0] != "observability.events", (
                "mount() must use coordinator.contribute() not register_capability() "
                "for observability.events"
            )


class TestShowProgressIsAsync:
    """Verify _show_progress is an async def (not sync def)."""

    def test_show_progress_is_async(self):
        """_show_progress must be async def so events are awaited, not fire-and-forget."""
        assert inspect.iscoroutinefunction(RecipeExecutor._show_progress), (
            "_show_progress must be async def, not sync def"
        )


class TestNoFireAndForgetEmit:
    """Verify no asyncio.create_task(hooks.emit(...)) pattern in executor source."""

    def test_no_fire_and_forget_emit_in_executor(self):
        """Source code must not contain create_task(hooks.emit pattern.

        The fire-and-forget pattern asyncio.create_task(hooks.emit(...)) causes
        events to be lost because they are scheduled but never awaited.
        The fix is to use 'await hooks.emit(...)' directly.
        """
        source_file = inspect.getfile(RecipeExecutor)
        with open(source_file) as f:
            source = f.read()

        # Check for the fire-and-forget anti-pattern
        pattern = r"create_task\s*\(\s*hooks\.emit"
        matches = re.findall(pattern, source)
        assert len(matches) == 0, (
            f"Found {len(matches)} fire-and-forget emit pattern(s) "
            f"(asyncio.create_task(hooks.emit(...))). "
            f"Use 'await hooks.emit(...)' instead."
        )
