import asyncio
import functools
import logging
import os

import click
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .agent_executor import OrchestratorAgentExecutor

load_dotenv()
logging.basicConfig()


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8000)
@click.option('--research-agent', 'research_agent', default='http://localhost:8001')
def main(host: str, port: int, research_agent: str):
    # Verify an API key is set
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )

    skill = AgentSkill(
        id='coordinate_research', 
        name='Research Coordination', 
        description='Coordinates research activities and provides intelligent summaries.',
        tags=['coordination', 'summary', 'adk'],
    )

    agent_card = AgentCard(
        name='orchestrator_agent',
        description='ADK-powered orchestrator that coordinates research and provides summaries.',
        url=f'http://{host}:{port}/',
        version='2.0',
        defaultInputModes=['application/json'],
        defaultOutputModes=['application/json'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    
    agent_executor = OrchestratorAgentExecutor(research_agent)
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    
    app = A2AStarletteApplication(agent_card, request_handler)
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == '__main__':
    main()