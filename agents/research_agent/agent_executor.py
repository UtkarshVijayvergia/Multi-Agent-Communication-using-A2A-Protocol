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
        
        try:
            # Submit and start work
            if not context.current_task:
                updater.submit()
            updater.start_work()
            
            # Convert A2A message to GenAI format
            new_message = types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts)
            )
            
            # Get or create session
            session = await self._upsert_session(context.context_id)
            
            # Run ADK agent
            async for event in self._run_agent(session.id, new_message, updater):
                logger.debug('Received ADK event: %s', event)
                
                if event.is_final_response():
                    # Convert response and complete task
                    response = convert_genai_parts_to_a2a(event.content.parts)
                    updater.add_artifact(response)
                    updater.complete()
                    break
                    
                elif event.get_function_calls():
                    # Tool is being called - status updates handled in tool
                    pass
                    
                elif event.content and event.content.parts:
                    # Intermediate response - update status
                    updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(
                            convert_genai_parts_to_a2a(event.content.parts)
                        )
                    )
                    
        except Exception as e:
            updater.fail(
                message=updater.new_agent_message([
                    Part(TextPart(text=f"Research failed: {str(e)}"))
                ])
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Get or create a session."""
        return await self.agent.runner.session_service.get_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        ) or await self.agent.runner.session_service.create_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        )