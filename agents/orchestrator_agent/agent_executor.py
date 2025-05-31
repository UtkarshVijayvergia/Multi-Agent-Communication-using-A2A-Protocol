import logging
from collections.abc import AsyncGenerator, AsyncIterable
from google.adk.events import Event
from google.adk.agents import RunConfig
from google.genai import types
from pydantic import ConfigDict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    Part,
    TextPart,
    TaskState,
    TaskStatus,
    UnsupportedOperationError
)
from a2a.utils.errors import ServerError

from .agent import OrchestratorAgent
from ..utils.converters import convert_a2a_parts_to_genai, convert_genai_parts_to_a2a
from ..utils.communication_tracker import tracker  # ← ADD THIS IMPORT

logger = logging.getLogger(__name__)


# Custom RunConfig to pass TaskUpdater through ADK
class A2ARunConfig(RunConfig):
    """Custom override of ADK RunConfig to pass TaskUpdater through the event loop."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    current_task_updater: TaskUpdater


class OrchestratorAgentExecutor(AgentExecutor):
    """Orchestrator Agent Executor using ADK framework."""
    
    def __init__(self, research_agent_endpoint: str):
        self.agent = OrchestratorAgent(research_agent_endpoint)

    def _run_agent(self, session_id: str, new_message: types.Content, task_updater: TaskUpdater) -> AsyncGenerator[Event]:
        """Run the ADK agent with custom run config."""
        return self.agent.runner.run_async(
            session_id=session_id,
            user_id='self',
            new_message=new_message,
            run_config=A2ARunConfig(current_task_updater=task_updater),
        )

    async def _process_request(self, new_message: types.Content, session_id: str, task_updater: TaskUpdater) -> AsyncIterable[TaskStatus | Artifact]:
        session = await self._upsert_session(session_id)
        session_id = session.id
        
        # Track request processing start
        tracker.log_event(
            event_type="request_processing_start",
            source_agent="orchestrator_agent",
            context_id=session_id,
            content="Starting request processing",
            metadata={"session_id": session_id}
        )
        
        async for event in self._run_agent(session_id, new_message, task_updater):
            logger.debug('Received ADK event: %s', event)
            
            # Track ADK events
            tracker.log_event(
                event_type="adk_event",
                source_agent="orchestrator_agent",
                context_id=session_id,
                content=f"ADK Event: {type(event).__name__}",
                metadata={"event_type": type(event).__name__, "has_content": bool(event.content)}
            )
            
            if event.is_final_response():
                response = convert_genai_parts_to_a2a(event.content.parts)
                logger.debug('Yielding final response: %s', response)
                task_updater.add_artifact(response)
                task_updater.complete()
                
                # Track final response
                tracker.log_event(
                    event_type="final_response",
                    source_agent="orchestrator_agent",
                    context_id=session_id,
                    content="Generated final response",
                    metadata={"response_parts": len(response)}
                )
                break
                
            if calls := event.get_function_calls():
                for call in calls:
                    # Track tool calls
                    tracker.log_event(
                        event_type="tool_call_detected",
                        source_agent="orchestrator_agent",
                        context_id=session_id,
                        content=f"Tool call: {call.name}",
                        metadata={"tool_name": call.name, "args": str(call.args)}
                    )
                    
                    if call.name == 'message_research_agent':
                        task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message([
                                Part(root=TextPart(text='🔄 Coordinating with research agent'))
                            ]),
                        )
                        
                        # Track coordination status
                        tracker.log_event(
                            event_type="coordination_status",
                            source_agent="orchestrator_agent",
                            context_id=session_id,
                            content="Coordinating with research agent",
                            metadata={"target_agent": "research_agent"}
                        )
            elif not event.get_function_calls():
                logger.debug('Yielding update response')
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(event.content.parts)
                    ),
                )
                
                # Track status updates
                if event.content and event.content.parts:
                    content_text = ' '.join([part.text for part in event.content.parts if hasattr(part, 'text') and part.text is not None])
                    tracker.log_event(
                        event_type="status_update",
                        source_agent="orchestrator_agent",
                        context_id=session_id,
                        content=f"Status update: {content_text[:100]}..." if content_text else "Status update: (no text content)",
                        metadata={"state": "working"}
                    )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Track execution start
        tracker.log_event(
            event_type="agent_execution_start",
            source_agent="orchestrator_agent",
            context_id=context.context_id,
            task_id=context.task_id,
            content="Starting orchestrator agent execution",
            metadata={"task_id": context.task_id, "context_id": context.context_id}
        )
        
        try:
            if not context.current_task:
                updater.submit()
                # Track task submission
                tracker.log_event(
                    event_type="task_submit",
                    source_agent="orchestrator_agent",
                    context_id=context.context_id,
                    task_id=context.task_id,
                    content="Task submitted to queue"
                )
            
            updater.start_work()
            # Track work start
            tracker.log_event(
                event_type="work_start",
                source_agent="orchestrator_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content="Agent started working on task"
            )
            
            # Track message conversion
            message_text = ' '.join([part.text for part in context.message.parts if hasattr(part, 'text')])
            tracker.log_event(
                event_type="message_conversion",
                source_agent="orchestrator_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content=f"Processing user message: {message_text[:100]}...",
                metadata={"original_parts": len(context.message.parts)}
            )
            
            await self._process_request(
                types.UserContent(
                    parts=convert_a2a_parts_to_genai(context.message.parts),
                ),
                context.context_id,
                updater,
            )
            
            # Track successful completion
            tracker.log_event(
                event_type="task_complete",
                source_agent="orchestrator_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content="Orchestration completed successfully"
            )
                    
        except Exception as e:
            # Use proper method name - it should be 'fail_task' not 'fail'
            updater.fail_task(
                message=updater.new_agent_message([
                    Part(TextPart(text=f"Orchestration failed: {str(e)}"))
                ])
            )
            
            # Track failure
            tracker.log_event(
                event_type="task_failed",
                source_agent="orchestrator_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content=f"Orchestration failed: {str(e)}",
                metadata={"error": str(e)}
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Track cancellation
        tracker.log_event(
            event_type="task_cancelled",
            source_agent="orchestrator_agent",
            context_id=context.context_id,
            task_id=context.task_id,
            content="Task was cancelled"
        )
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        return await self.agent.runner.session_service.get_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        ) or await self.agent.runner.session_service.create_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        )