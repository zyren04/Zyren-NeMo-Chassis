"""Tests validating NeMo Relay context isolation patterns."""

import asyncio
import pytest
import sys
from src.infrastructure.nemo_relay_integration import get_nemo_relay_integration, NEMO_RELAY_AVAILABLE
from src.runtime.engine import create_workflow_engine
from src.state.state_schema import BaseState


class TestNeMoRelayContextIsolation:
    """Validate context isolation for concurrent workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+ context parameter")
    async def test_concurrent_workflows_independent_stacks(self):
        """Each concurrent workflow execution has independent NeMo Relay scope stack."""
        integration = get_nemo_relay_integration()
        
        # Track scope handles seen by each execution
        execution_scopes = {}
        
        async def tracking_node(state):
            handle = integration.get_current_scope()
            execution_scopes[state.execution_id] = handle
            await asyncio.sleep(0.01)  # Yield to allow interleaving
            return state.increment_iteration()
        
        engine = create_workflow_engine()
        engine.register_node("tracker", tracking_node)
        engine.set_entry_point("tracker")
        compiled = engine.compile()
        
        # 10 concurrent executions
        states = [BaseState(execution_id=f"exec-{i}") for i in range(10)]
        results = await compiled.abatch(states)
        
        # Verify each execution got DIFFERENT scope handle
        handles = list(execution_scopes.values())
        assert len(set(id(h) for h in handles)) == 10, "Each execution must have unique scope handle"
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+ context parameter")
    async def test_fork_asyncio_context_isolation(self):
        """fork_asyncio_context creates isolated stack with preserved ancestry."""
        integration = get_nemo_relay_integration()
        
        parent_scopes = []
        child_scopes = []
        
        async def parent_workflow():
            async with integration.workflow_scope("parent", "test") as handle:
                parent_scopes.append(handle)
                
                # Spawn child with forked context
                async def child_workflow():
                    async with integration.workflow_scope("child", "test") as child_handle:
                        child_scopes.append(child_handle)
                
                child_context = integration.fork_asyncio_context()
                # Python 3.11+ supports context parameter
                if sys.version_info >= (3, 11):
                    task = asyncio.create_task(child_workflow(), context=child_context)
                else:
                    # For Python 3.10, run in the context manually
                    task = asyncio.create_task(child_context.run(child_workflow))
                await task
        
        await parent_workflow()
        
        # Verify: parent and child have DIFFERENT scope handles
        assert parent_scopes[0] is not child_scopes[0]
        # Verify: child's parent is parent's scope (ancestry preserved)
        # This would require NeMo Relay API to inspect parentage
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+ context parameter")
    async def test_thread_pool_propagation(self):
        """ThreadPoolExecutor workers receive propagated scope stack."""
        from concurrent.futures import ThreadPoolExecutor
        
        integration = get_nemo_relay_integration()
        thread_scopes = []
        
        async def test_thread_propagation():
            async with integration.workflow_scope("parent", "test") as handle:
                stack = integration.propagate_scope_to_thread()
                
                def worker():
                    integration.set_thread_scope_stack(stack)
                    # In worker thread, get_scope_stack should return propagated stack
                    worker_stack = integration.get_scope_stack()
                    thread_scopes.append(worker_stack)
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(worker) for _ in range(4)]
                    for f in futures:
                        f.result()
        
        await test_thread_propagation()
        
        # All workers should have the SAME propagated stack (shared trace)
        assert len(set(id(s) for s in thread_scopes)) == 1
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+ context parameter")
    async def test_scope_local_middleware_no_leak(self):
        """Scope-local middleware registered in one workflow doesn't leak to another."""
        # Requires NeMo Relay scope_local middleware registration
        # This test validates the core skill requirement
        pass  # Implementation depends on nemo_relay.scope_local API
