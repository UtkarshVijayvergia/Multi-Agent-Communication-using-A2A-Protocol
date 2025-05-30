import asyncio
import uvicorn
import logging
from dotenv import load_dotenv

from agents.research_agent.agent_executor import ResearchAgentExecutor
from agents.orchestrator_agent.agent_executor import OrchestratorAgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentSkill

load_dotenv()
logging.basicConfig(level=logging.INFO)


async def run_agents():
    """Run both research and orchestrator agents."""
    
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

    # Run both servers concurrently
    server1 = uvicorn.Server(uvicorn.Config(research_app, port=8001, log_level='info'))
    server2 = uvicorn.Server(uvicorn.Config(orchestrator_app, port=8000, log_level='info'))
    
    await asyncio.gather(server1.serve(), server2.serve())


if __name__ == '__main__':
    asyncio.run(run_agents())