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
        
        async for event in self._run_agent(session_id, new_message, task_updater):
            logger.debug('Received ADK event: %s', event)
            
            if event.is_final_response():
                response = convert_genai_parts_to_a2a(event.content.parts)
                logger.debug('Yielding final response: %s', response)
                task_updater.add_artifact(response)
                task_updater.complete()
                break
                
            if calls := event.get_function_calls():
                for call in calls:
                    if call.name == 'message_research_agent':
                        task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message([
                                Part(root=TextPart(text='🔄 Coordinating with research agent'))
                            ]),
                        )
            elif not event.get_function_calls():
                logger.debug('Yielding update response')
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(event.content.parts)
                    ),
                )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            if not context.current_task:
                updater.submit()
            updater.start_work()
            
            await self._process_request(
                types.UserContent(
                    parts=convert_a2a_parts_to_genai(context.message.parts),
                ),
                context.context_id,
                updater,
            )
                    
        except Exception as e:
            updater.fail(
                message=updater.new_agent_message([
                    Part(TextPart(text=f"Orchestration failed: {str(e)}"))
                ])
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        return await self.agent.runner.session_service.get_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        ) or await self.agent.runner.session_service.create_session(
            app_name=self.agent.runner.app_name, user_id='self', session_id=session_id
        )