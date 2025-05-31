import logging
import json
import time
import os
import threading
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class CommunicationEvent:
    timestamp: str
    event_type: str  # 'request', 'response', 'status_update', 'tool_call'
    source_agent: str
    target_agent: str
    message_id: str
    context_id: str
    task_id: str
    content: str
    metadata: Dict[str, Any]

class CommunicationTracker:
    """Tracks all communication between agents for visualization."""
    
    def __init__(self):
        self.events: List[CommunicationEvent] = []
        # Create shared events file
        self.shared_file = Path("communication_events_shared.jsonl")
        self.lock = threading.Lock()
        
        # Enable console logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
    def log_event(self, event_type: str, source_agent: str, target_agent: str = None,
                  message_id: str = None, context_id: str = None, task_id: str = None,
                  content: str = "", metadata: Dict[str, Any] = None):
        """Log a communication event."""
        event = CommunicationEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent or "system",
            message_id=message_id or "",
            context_id=context_id or "",
            task_id=task_id or "",
            content=content[:500] + "..." if len(content) > 500 else content,  # Truncate for readability
            metadata=metadata or {}
        )
        
        # Add to local events
        self.events.append(event)
        
        # Write to shared file for cross-process communication
        self._write_to_shared_file(event)
        
        # Log to console with visual formatting
        self._log_formatted_event(event)
        
    def _write_to_shared_file(self, event: CommunicationEvent):
        """Write event to shared file for cross-process access."""
        try:
            with self.lock:
                with open(self.shared_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(event)) + '\n')
        except Exception as e:
            # Don't let file errors break the main flow
            print(f"Warning: Could not write to shared file: {e}")
    
    def _read_all_events_from_shared_file(self) -> List[CommunicationEvent]:
        """Read all events from shared file across all processes."""
        all_events = []
        
        if not self.shared_file.exists():
            return all_events
            
        try:
            with open(self.shared_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        event_dict = json.loads(line)
                        event = CommunicationEvent(**event_dict)
                        all_events.append(event)
        except Exception as e:
            print(f"Warning: Could not read shared file: {e}")
            
        # Sort by timestamp
        all_events.sort(key=lambda x: x.timestamp)
        return all_events
        
    def _log_formatted_event(self, event: CommunicationEvent):
        """Log event with visual formatting."""
        arrow = "→" if event.target_agent != "system" else "📝"
        
        log_msg = (
            f"\n{'='*80}\n"
            f"🔄 {event.event_type.upper()} | {event.timestamp}\n"
            f"{arrow} {event.source_agent} → {event.target_agent}\n"
            f"📋 Context: {event.context_id} | Task: {event.task_id}\n"
            f"💬 {event.content}\n"
            f"📊 Metadata: {json.dumps(event.metadata, indent=2) if event.metadata else 'None'}\n"
            f"{'='*80}"
        )
        
        self.logger.info(log_msg)
        # Print directly to ensure it shows up
        print(log_msg)
    
    def get_communication_flow(self) -> List[Dict]:
        """Get all events as a list of dictionaries for visualization."""
        # Get events from ALL processes via shared file
        all_events = self._read_all_events_from_shared_file()
        return [asdict(event) for event in all_events]
    
    def export_to_json(self, filename: str = None):
        """Export communication log to JSON file."""
        if not filename:
            filename = f"communication_log_{int(time.time())}.json"
        
        # Get all events from shared file
        all_events_data = self.get_communication_flow()
        
        with open(filename, 'w') as f:
            json.dump(all_events_data, f, indent=2)
        
        print(f"📊 Communication log exported to {filename} ({len(all_events_data)} events)")
        return filename
    
    def export_visualization_data(self):
        """Export data for external visualization tools."""
        flow_data = self.get_communication_flow()
        
        # Create visualization-friendly format
        viz_data = {
            "nodes": [],
            "edges": [],
            "timeline": []
        }
        
        # Extract unique agents as nodes
        agents = set()
        for event in flow_data:
            agents.add(event["source_agent"])
            if event["target_agent"] != "system":
                agents.add(event["target_agent"])
        
        for agent in agents:
            viz_data["nodes"].append({
                "id": agent,
                "label": agent.replace("_", " ").title(),
                "type": "agent"
            })
        
        # Create edges for communications
        for i, event in enumerate(flow_data):
            if event["target_agent"] != "system":
                viz_data["edges"].append({
                    "id": f"edge_{i}",
                    "source": event["source_agent"],
                    "target": event["target_agent"],
                    "label": event["event_type"],
                    "timestamp": event["timestamp"],
                    "content": event["content"][:50] + "..." if len(event["content"]) > 50 else event["content"]
                })
        
        # Timeline data
        viz_data["timeline"] = flow_data
        
        # Save to file
        filename = f"communication_viz_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        print(f"📊 Visualization data exported to {filename} ({len(flow_data)} events, {len(viz_data['nodes'])} agents, {len(viz_data['edges'])} communications)")
        return filename
    
    def clear_shared_file(self):
        """Clear the shared communication file (use at start of new session)."""
        if self.shared_file.exists():
            self.shared_file.unlink()
        print("🧹 Cleared shared communication file")

# Global tracker instance
tracker = CommunicationTracker()