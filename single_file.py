import os
import asyncio
import uvicorn
import httpx
import logging
from typing import Any
from uuid import uuid4
from collections.abc import AsyncGenerator, AsyncIterable

from google.adk import Runner
from google.adk.agents import LlmAgent, RunConfig
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from pydantic import ConfigDict

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import get_text_parts
from a2a.types import (
    Artifact,
    AgentSkill,
    SendMessageRequest, 
    Message, 
    MessageSendParams, 
    TextPart, 
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    SendMessageSuccessResponse,
    UnsupportedOperationError
)
from a2a.client import A2AClient
from a2a.utils.errors import ServerError

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure your Google API key for Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Custom RunConfig to pass TaskUpdater through ADK
class A2ARunConfig(RunConfig):
    """Custom override of ADK RunConfig to pass TaskUpdater through the event loop."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    current_task_updater: TaskUpdater
    

# ----- Research Agent (ADK-based) -----
class ResearchAgentExecutor(AgentExecutor):
    """Research Agent using ADK framework."""
    
    def __init__(self):
        # Create ADK LLM Agent with tools
        self._agent = LlmAgent(
            model='gemini-2.0-flash-exp',
            name='research_agent',
            description='An agent that provides research on given topics.',
            instruction="""
You are a research agent that provides comprehensive research on any topic.

When asked about a topic, provide detailed research including:
- Key findings and data points
- Recent trends and developments  
- Expert opinions and insights
- Relevant statistics or studies

Always structure your response clearly and provide actionable insights.
""",
            tools=[self.conduct_research],
        )
        
        # Create ADK Runner
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def conduct_research(self, topic: str, tool_context: ToolContext):
        """Conduct research on the given topic."""
        # Get TaskUpdater from context
        task_updater = self._get_task_updater(tool_context)
        
        # Update status to show we're researching
        task_updater.update_status(
            TaskState.working,
            message=task_updater.new_agent_message([
                Part(TextPart(text=f"🔍 Researching topic: {topic}"))
            ])
        )
        
        # Simulate research process
        await asyncio.sleep(0.5)  # Simulate research time
        
        research_result = (
            f"Research Results for: {topic}\n\n"
            f"📊 Key Findings:\n"
            f"- Found 3 relevant academic papers on {topic}\n"
            f"- Dataset analysis shows 20% increase in related metrics over last year\n"
            f"- Industry experts highlight emerging challenges in {topic} adoption\n\n"
            f"📈 Recent Trends:\n"
            f"- Growing interest in {topic} applications\n"
            f"- New methodologies being developed\n"
            f"- Regulatory frameworks evolving\n\n"
            f"💡 Expert Insights:\n"
            f"- Dr. Smith recommends focusing on scalability aspects\n"
            f"- Recent Stanford study suggests promising future applications\n"
            f"- Industry report indicates 15% market growth expected"
        )
        
        return {"result": research_result}

    def _get_task_updater(self, tool_context: ToolContext) -> TaskUpdater:
        """Extract TaskUpdater from tool context."""
        return tool_context._invocation_context.run_config.current_task_updater

    def _run_agent(self, session_id: str, new_message: types.Content, task_updater: TaskUpdater) -> AsyncGenerator[Event]:
        """Run the ADK agent with custom run config."""
        return self.runner.run_async(
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
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )


# ----- Orchestrator Agent (ADK-based) -----
class OrchestratorAgentExecutor(AgentExecutor):
    # TODO: Fucntion left to implement: _wait_for_dependent_task
         #  dependent functions: _is_task_complete
    """Orchestrator Agent using ADK framework."""
    
    def __init__(self, research_agent_endpoint: str):
        # Create ADK LLM Agent
        self._agent = LlmAgent(
            model='gemini-2.0-flash',
            name='orchestrator_agent',
            description='An agent that coordinates research and provides summaries.',
            instruction="""
    You are an orchestrator agent that coordinates research activities and provides comprehensive summaries.

    Your workflow:
    1. Use the research_agent agent to gather information on the user's topic
    2. Analyze and synthesize the research findings given by research_agent
    4. If the research is incomplete, ask follow-up questions to clarify the user's needs
    5. If the research is sufficient, provide a final summary with actionable insights
    6. If the user asks for more details, provide additional context or data as needed

    Always provide structured, insightful responses that add value beyond the raw research.
    """,
            tools=[self.message_research_agent],
        )
        
        self.research_agent_endpoint = research_agent_endpoint
        
        # Create ADK Runner
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def message_research_agent(self, message: str, tool_context: ToolContext):
        """Call the research agent via A2A."""
        # We take an overly simplistic approach to the A2A state machine:
        # - All requests to the calendar agent use the current session ID as the context ID.
        # - If the last response from the calendar agent (in this session) produced a non-terminal
        #   task state, the request references that task.
        task_updater = self._get_task_updater(tool_context)
        
        # Update status
        task_updater.update_status(
            TaskState.working,
            message=task_updater.new_agent_message([
                Part(TextPart(text="📞 Contacting research agent..."))
            ])
        )
        
        # Prepare A2A request
        request = SendMessageRequest(
            params=MessageSendParams(
                message=Message(
                    contextId=tool_context._invocation_context.session.id,
                    # TODO: Look into TaskID
                    # taskId=tool_context.state.get('task_id'),
                    messageId=str(uuid4()),
                    role=Role.user,
                    parts=[Part(TextPart(text=message))],
                )
            )
        )
        
        response = await self._send_agent_message(request)
        logger.debug('[A2A Client] Received response: %s', response)
        task_id = None
        # Extract research content
        content = []
        if isinstance(response.root, SendMessageSuccessResponse):
            if isinstance(response.root.result, Task):
                task = response.root.result
                if task.artifacts:
                    for artifact in task.artifacts:
                        content.extend(get_text_parts(artifact.parts))
                if not content:
                    content.extend(get_text_parts(task.status.message.parts))
            else:
                content.extend(get_text_parts(response.root.result.parts))
        tool_context.state['task_id'] = task_id
        
        research_text = '\n'.join(content)
        
        # Update status
        task_updater.update_status(
            TaskState.working,
            message=task_updater.new_agent_message([
                Part(TextPart(text="✅ Research data retrieved successfully"))
            ])
        )
        
        return {"research_results": research_text}
        
    async def _send_agent_message(self, request: SendMessageRequest):
        async with httpx.AsyncClient() as client:
            research_agent_client = A2AClient(
                httpx_client=client, url=self.research_agent_endpoint
            )
            return await research_agent_client.send_message(request)

    # Extra function for getting task details from the research agent
    # TODO: task_id left to be implemented in message_research_agent and...
    async def _get_agent_task(self, task_id) -> Task:
        async with httpx.AsyncClient() as client:
            a2a_client = A2AClient(
                httpx_client=client, url=self.research_agent_endpoint
            )
            await a2a_client.get_task({'id': task_id})

    def _get_task_updater(self, tool_context: ToolContext) -> TaskUpdater:
        """Extract TaskUpdater from tool context."""
        return tool_context._invocation_context.run_config.current_task_updater

    def _run_agent(self, session_id: str, new_message: types.Content, task_updater: TaskUpdater) -> AsyncGenerator[Event]:
        """Run the ADK agent with custom run config."""
        return self.runner.run_async(
            session_id=session_id,
            user_id='self',
            new_message=new_message,
            run_config=A2ARunConfig(current_task_updater=task_updater),
        )

    
    async def _process_request(self, new_message: types.Content, session_id: str, task_updater: TaskUpdater) -> AsyncIterable[TaskStatus | Artifact]:
        session = await self._upsert_session(
            session_id,
        )
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
                    # Provide an update on what we're doing.
                    if call.name == 'message_research_agent':
                        task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message(
                                [
                                    Part(
                                        root=TextPart(
                                            text='🔄 Coordinating with research agent'
                                        )
                                    )
                                ]
                            ),
                        )
            elif not event.get_function_calls():
                logger.debug('Yielding update response')
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(event.content.parts)
                    ),
                )
            else:
                logger.debug('Skipping event')


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
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )





def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert a list of A2A Part types into a list of Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type."""
    part = part.root
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.file.uri, mime_type=part.file.mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=part.file.bytes, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    """Convert a list of Google Gen AI Part types into a list of A2A Part types."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type."""
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')


# ----- Application Setup -----

# Research Agent setup
research_skill = AgentSkill(
    id="research",
    name="AI Research",
    description="Provides comprehensive research on any topic using AI.",
    tags=["research", "ai", "analysis"]
)

research_agent_card = {
    "name": "research_agent",
    "description": "AI-powered research agent using ADK framework.",
    "url": "http://127.0.0.1:8001/",
    "version": "2.0",
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "capabilities": {"streaming": True},
    "skills": [research_skill]
}

research_handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore()
)

research_app = A2AStarletteApplication(
    agent_card=research_agent_card,
    http_handler=research_handler
).build()

# Orchestrator Agent setup
orchestrator_handler = DefaultRequestHandler(
    agent_executor=OrchestratorAgentExecutor(research_agent_endpoint='http://127.0.0.1:8001/'),
    task_store=InMemoryTaskStore()
)

orchestrator_app = A2AStarletteApplication(
    agent_card={
        'name': 'orchestrator_agent',
        'description': 'ADK-powered orchestrator that coordinates research and provides summaries.',
        'url': 'http://127.0.0.1:8000/',
        'version': '2.0',
        'defaultInputModes': ['application/json'],
        'defaultOutputModes': ['application/json'],
        'capabilities': {'streaming': True, 'tools': True},
        'skills': [{
            'id': 'coordinate_research', 
            'name': 'Research Coordination', 
            'description': 'Coordinates research activities and provides intelligent summaries.',
            'tags': ['coordination', 'summary', 'adk']
        }]
    },
    http_handler=orchestrator_handler
).build()

# Run servers
if __name__ == '__main__':
    async def run_servers():
        server1 = uvicorn.Server(uvicorn.Config(research_app, port=8001, log_level='info'))
        server2 = uvicorn.Server(uvicorn.Config(orchestrator_app, port=8000, log_level='info'))
        await asyncio.gather(server1.serve(), server2.serve())
    
    asyncio.run(run_servers())