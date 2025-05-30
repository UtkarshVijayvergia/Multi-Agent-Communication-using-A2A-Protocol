import asyncio
import logging
import os

import click
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill

from .agent_executor import ResearchAgentExecutor

load_dotenv()
logging.basicConfig()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8001)
def main(host: str, port: int):
    # Verify an API key is set.
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and '
            'GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )
    
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
        "url": f"http://{host}:{port}/",
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

    uvicorn.run(research_app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()