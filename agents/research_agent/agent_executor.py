import logging
from collections.abc import AsyncGenerator
from google.adk.events import Event
from google.adk.agents import RunConfig
from google.genai import types
from pydantic import ConfigDict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, TaskState, UnsupportedOperationError
from a2a.utils.errors import ServerError

from .agent import ResearchAgent
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


class ResearchAgentExecutor(AgentExecutor):
    """Research Agent Executor using ADK framework."""
    
    def __init__(self):
        self.agent = ResearchAgent()

    def _run_agent(self, session_id: str, new_message: types.Content, task_updater: TaskUpdater) -> AsyncGenerator[Event]:
        """Run the ADK agent with custom run config."""
        return self.agent.runner.run_async(
            session_id=session_id,
            user_id='self',
            new_message=new_message,
            run_config=A2ARunConfig(current_task_updater=task_updater),
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Track execution start
        tracker.log_event(
            event_type="agent_execution_start",
            source_agent="research_agent",
            context_id=context.context_id,
            task_id=context.task_id,
            content=f"Starting research agent execution",
            metadata={"task_id": context.task_id, "context_id": context.context_id}
        )
        
        try:
            # Submit and start work
            if not context.current_task:
                updater.submit()
                # Track task submission
                tracker.log_event(
                    event_type="task_submit",
                    source_agent="research_agent",
                    context_id=context.context_id,
                    task_id=context.task_id,
                    content="Task submitted to queue"
                )
            
            updater.start_work()
            # Track work start
            tracker.log_event(
                event_type="work_start",
                source_agent="research_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content="Agent started working on task"
            )
            
            # Convert A2A message to GenAI format
            new_message = types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts)
            )
            
            # Track message conversion
            message_text = ' '.join([part.text for part in context.message.parts if hasattr(part, 'text')])
            tracker.log_event(
                event_type="message_conversion",
                source_agent="research_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content=f"Converted A2A message to GenAI format: {message_text[:100]}...",
                metadata={"original_parts": len(context.message.parts)}
            )
            
            # Get or create session
            session = await self._upsert_session(context.context_id)
            
            # Track session info
            tracker.log_event(
                event_type="session_ready",
                source_agent="research_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content=f"Session ready: {session.id}",
                metadata={"session_id": session.id}
            )
            
            # Run ADK agent
            async for event in self._run_agent(session.id, new_message, updater):
                logger.debug('Received ADK event: %s', event)
                
                # Track ADK events
                tracker.log_event(
                    event_type="adk_event",
                    source_agent="research_agent",
                    context_id=context.context_id,
                    task_id=context.task_id,
                    content=f"ADK Event: {type(event).__name__}",
                    metadata={"event_type": type(event).__name__, "has_content": bool(event.content)}
                )
                
                if event.is_final_response():
                    # Convert response and complete task
                    response = convert_genai_parts_to_a2a(event.content.parts)
                    updater.add_artifact(response)
                    updater.complete()
                    
                    # Track completion
                    tracker.log_event(
                        event_type="task_complete",
                        source_agent="research_agent",
                        context_id=context.context_id,
                        task_id=context.task_id,
                        content="Research task completed successfully",
                        metadata={"response_parts": len(response)}
                    )
                    break
                    
                elif event.get_function_calls():
                    # Track tool calls
                    calls = event.get_function_calls()
                    for call in calls:
                        tracker.log_event(
                            event_type="tool_call_detected",
                            source_agent="research_agent",
                            context_id=context.context_id,
                            task_id=context.task_id,
                            content=f"Tool call: {call.name}",
                            metadata={"tool_name": call.name, "args": str(call.args)}
                        )
                    
                elif event.content and event.content.parts:
                    # Intermediate response - update status
                    updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(
                            convert_genai_parts_to_a2a(event.content.parts)
                        )
                    )
                    
                    # Track status update - FIX THE NONE TEXT ISSUE
                    content_text = ' '.join([part.text for part in event.content.parts if hasattr(part, 'text') and part.text is not None])
                    tracker.log_event(
                        event_type="status_update",
                        source_agent="research_agent",
                        context_id=context.context_id,
                        task_id=context.task_id,
                        content=f"Status update: {content_text[:100]}..." if content_text else "Status update: (no text content)",
                        metadata={"state": "working"}
                    )
                    
        except Exception as e:
            # FIX THE FAIL METHOD NAME
            updater.fail_task(
                message=updater.new_agent_message([
                    Part(TextPart(text=f"Research failed: {str(e)}"))
                ])
            )
            
            # Track failure
            tracker.log_event(
                event_type="task_failed",
                source_agent="research_agent",
                context_id=context.context_id,
                task_id=context.task_id,
                content=f"Research failed: {str(e)}",
                metadata={"error": str(e)}
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Track cancellation
        tracker.log_event(
            event_type="task_cancelled",
            source_agent="research_agent",
            context_id=context.context_id,
            task_id=context.task_id,
            content="Task was cancelled"
        )
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Get or create a session."""
        return await self.agent.runner.session_service.get_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        ) or await self.agent.runner.session_service.create_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        )