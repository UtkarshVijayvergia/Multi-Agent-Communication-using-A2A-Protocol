import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from agents.utils.communication_tracker import tracker

class CommunicationDashboard:
    """Real-time dashboard for monitoring agent communications."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.last_event_count = 0
        
    async def start_monitoring(self):
        """Start real-time monitoring of communications."""
        print("🔄 Starting Communication Dashboard...")
        print("=" * 80)
        
        while True:
            # Get current events from shared file across all processes
            all_events = tracker.get_communication_flow()
            current_events = len(all_events)
            
            if current_events > self.last_event_count:
                # Display new events
                new_events = all_events[self.last_event_count:]
                self._display_new_events(new_events)
                self.last_event_count = current_events
                
                # Generate flow summary
                self._display_flow_summary(all_events)
            
            await asyncio.sleep(self.update_interval)
    
    def _display_new_events(self, events):
        """Display new communication events."""
        for event in events:
            self._display_event(event)
    
    def _display_event(self, event):
        """Display a single communication event with visual formatting."""
        timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
        
        # Color coding based on event type
        color_map = {
            "a2a_request": "🔄",
            "a2a_response": "✅", 
            "tool_call_start": "🔧",
            "tool_call_complete": "✅",
            "status_update": "📝",
            "content_extraction": "📊"
        }
        
        icon = color_map.get(event["event_type"], "📋")
        
        print(f"\n{icon} [{timestamp}] {event['event_type'].upper()}")
        print(f"   {event['source_agent']} → {event['target_agent']}")
        
        if event["context_id"]:
            print(f"   Context: {event['context_id'][:8]}...")
        
        if event["content"]:
            content_preview = event["content"][:100] + "..." if len(event["content"]) > 100 else event["content"]
            print(f"   💬 {content_preview}")
        
        if event["metadata"]:
            print(f"   📊 {event['metadata']}")
    
    def _display_flow_summary(self, events):
        """Display a summary of the communication flow."""
        if not events:
            return
            
        print(f"\n{'='*60}")
        print(f"📊 COMMUNICATION FLOW SUMMARY")
        print(f"{'='*60}")
        print(f"Total Events: {len(events)}")
        
        # Count by event type
        event_counts = {}
        for event in events:
            event_counts[event["event_type"]] = event_counts.get(event["event_type"], 0) + 1
        
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")
        
        # Show active contexts
        contexts = set(event["context_id"] for event in events if event["context_id"])
        print(f"Active Contexts: {len(contexts)}")
        
        print(f"{'='*60}\n")
    
    def export_visualization_data(self):
        """Export data for external visualization tools."""
        return tracker.export_visualization_data()

# Function to start dashboard in background
async def start_dashboard():
    """Start the communication dashboard."""
    dashboard = CommunicationDashboard()
    await dashboard.start_monitoring()

if __name__ == "__main__":
    asyncio.run(start_dashboard())