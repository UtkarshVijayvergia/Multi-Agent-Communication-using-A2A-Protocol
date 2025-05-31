import asyncio
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from a2a.types import Part, TextPart, TaskState

from ..utils.communication_tracker import tracker


class ResearchAgent:
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
        
        # Track tool call start
        tracker.log_event(
            event_type="tool_call_start",
            source_agent="research_agent",
            context_id=tool_context._invocation_context.session.id,
            content=f"Starting research on topic: {topic}",
            metadata={"tool": "conduct_research", "topic": topic}
        )
        
        # Update status to show we're researching
        task_updater.update_status(
            TaskState.working,
            message=task_updater.new_agent_message([
                Part(TextPart(text=f"🔍 Researching topic: {topic}"))
            ])
        )
        
        # Track status update
        tracker.log_event(
            event_type="status_update",
            source_agent="research_agent",
            context_id=tool_context._invocation_context.session.id,
            content=f"Status: Researching {topic}",
            metadata={"state": "working"}
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
        
        # Track tool call completion
        tracker.log_event(
            event_type="tool_call_complete",
            source_agent="research_agent",
            context_id=tool_context._invocation_context.session.id,
            content=f"Research completed for: {topic}",
            metadata={"tool": "conduct_research", "result_length": len(research_result)}
        )
        
        return {"result": research_result}

    def _get_task_updater(self, tool_context: ToolContext):
        """Extract TaskUpdater from tool context."""
        return tool_context._invocation_context.run_config.current_task_updater