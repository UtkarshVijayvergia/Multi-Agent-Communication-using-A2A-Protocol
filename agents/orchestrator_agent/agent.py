import httpx
from uuid import uuid4
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext

from a2a.client import A2AClient
from a2a.utils import get_text_parts
from a2a.types import (
    SendMessageRequest,
    Message,
    MessageSendParams,
    TextPart,
    Part,
    Role,
    Task,
    TaskState,
    SendMessageSuccessResponse
)

from ..utils.communication_tracker import tracker


class OrchestratorAgent:
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
3. If the research is incomplete, ask follow-up questions to clarify the user's needs
4. If the research is sufficient, provide a final summary with actionable insights
5. If the user asks for more details, provide additional context or data as needed

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
        task_updater = self._get_task_updater(tool_context)
        session_id = tool_context._invocation_context.session.id
        
        # Track outgoing request to research agent
        tracker.log_event(
            event_type="a2a_request",
            source_agent="orchestrator_agent",
            target_agent="research_agent",
            context_id=session_id,
            content=f"Requesting research: {message}",
            metadata={"endpoint": self.research_agent_endpoint, "method": "message/send"}
        )
        
        # Update status
        task_updater.update_status(
            TaskState.working,
            message=task_updater.new_agent_message([
                Part(TextPart(text="📞 Contacting research agent..."))
            ])
        )
        
        # Prepare A2A request
        message_id = str(uuid4())
        request = SendMessageRequest(
            params=MessageSendParams(
                message=Message(
                    contextId=session_id,
                    messageId=message_id,
                    role=Role.user,
                    parts=[Part(TextPart(text=message))],
                )
            )
        )
        
        # Track request details
        tracker.log_event(
            event_type="a2a_request_details",
            source_agent="orchestrator_agent",
            target_agent="research_agent",
            message_id=message_id,
            context_id=session_id,
            content=message,
            metadata={"request_type": "SendMessageRequest"}
        )
        
        response = await self._send_agent_message(request)
        
        # Track response received
        tracker.log_event(
            event_type="a2a_response",
            source_agent="research_agent",
            target_agent="orchestrator_agent",
            message_id=message_id,
            context_id=session_id,
            content="Response received from research agent",
            metadata={"response_type": type(response.root).__name__}
        )
        
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
        
        research_text = '\n'.join(content)
        
        # Track content extraction
        tracker.log_event(
            event_type="content_extraction",
            source_agent="orchestrator_agent",
            context_id=session_id,
            content=f"Extracted {len(content)} text parts, total length: {len(research_text)}",
            metadata={"extracted_parts": len(content), "total_length": len(research_text)}
        )
        
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

    def _get_task_updater(self, tool_context: ToolContext):
        """Extract TaskUpdater from tool context."""
        return tool_context._invocation_context.run_config.current_task_updater