import httpx
import json
import asyncio
from uuid import uuid4

async def call_orchestrator_real_time_streaming(question: str):
    """Call orchestrator with real-time streaming that shows words as they appear."""
    
    payload = {
        "jsonrpc": "2.0",
        "id": "req-1", 
        "method": "message/send",
        "params": {
            "message": {
                "contextId": str(uuid4()),
                "messageId": str(uuid4()),
                "role": "user",
                "parts": [
                    {"type": "text", "text": question}
                ]
            },
            "configuration": {
                "blocking": False,  # Don't wait for completion
                "acceptedOutputModes": ["application/json"]
            }
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print("🔄 Sending request...")
            resp = await client.post(
                "http://127.0.0.1:8000/",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            
            response = resp.json()
            result = response.get("result", {})
            
            # Get task ID to poll for updates
            task_id = result.get("id")
            context_id = result.get("contextId")
            
            if not task_id:
                print("❌ No task ID received")
                return
                
            print(f"✅ Task started: {task_id}")
            print(f"👤 User: {question}\n")
            
            # Poll for real-time updates with correct A2A format
            await poll_task_real_time(client, task_id, context_id)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

async def poll_task_real_time(client, task_id, context_id):
    """Poll task for real-time updates using correct A2A API format."""
    
    last_message_count = 0
    seen_streaming_chunks = set()
    current_streaming_text = ""
    is_first_streaming_chunk = True
    
    for poll_count in range(120):  # Poll for up to 2 minutes
        try:
            # Use correct A2A task/get endpoint format
            get_task_payload = {
                "jsonrpc": "2.0",
                "id": f"poll-{poll_count}",
                "method": "tasks/get",  # Fixed: was "task/get", should be "tasks/get"
                "params": {
                    "id": task_id,  # Fixed: was "taskId", should be "id"
                    # Don't include contextId here - it's not needed for task/get
                }
            }
            
            resp = await client.post(
                "http://127.0.0.1:8000/",
                json=get_task_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if resp.status_code == 200:
                task_response = resp.json()
                
                # Check if there's an error in the response
                if "error" in task_response:
                    print(f"❌ Task polling error: {task_response['error']}")
                    break
                    
                task_result = task_response.get("result", {})
                
                # Check task state
                status = task_result.get("status", {})
                state = status.get("state", "unknown")
                
                # Get current history
                history = task_result.get("history", [])
                
                # Show new messages since last poll
                if len(history) > last_message_count:
                    for i in range(last_message_count, len(history)):
                        entry = history[i]
                        role = entry.get("role", "unknown")
                        parts = entry.get("parts", [])
                        
                        for part in parts:
                            text = part.get("text", "")
                            
                            if role == "agent":
                                # Check if this is a streaming chunk (starts with 📝)
                                if text.startswith("📝 "):
                                    chunk_text = text[2:].strip()  # Remove "📝 " prefix
                                    chunk_id = f"{i}_{chunk_text[:20]}"  # Unique ID for this chunk
                                    
                                    if chunk_id not in seen_streaming_chunks:
                                        seen_streaming_chunks.add(chunk_id)
                                        
                                        # Add label for first streaming chunk
                                        if is_first_streaming_chunk:
                                            print("🤖 Agent: Finalized Result: \n\n", end="", flush=True)
                                            is_first_streaming_chunk = False
                                        
                                        # Print word by word preserving natural formatting
                                        await print_word_by_word_with_formatting(chunk_text)
                                        current_streaming_text += chunk_text
                                else:
                                    # Regular status message - ADD LINE BREAK HERE
                                    print(f"🤖 Agent: {text}")
                    
                    last_message_count = len(history)
                
                # Check if task is complete
                if state == "completed":
                    if current_streaming_text:
                        print("\n")  # End the streaming line
                    
                    print(f"\n✅ Task completed!")
                    
                    # Show final artifacts
                    artifacts = task_result.get("artifacts", [])
                    if artifacts:
                        print("\n🎯 FINAL COMPLETE RESULT:")
                        print("=" * 60)
                        for artifact in artifacts:
                            parts = artifact.get("parts", [])
                            for part in parts:
                                final_text = part.get("text", "")
                                print(final_text)
                        print("=" * 60)
                    return
                    
                elif state == "failed":
                    print("❌ Task failed!")
                    return
                    
            else:
                print(f"⚠️ Polling failed (status {resp.status_code})")
                # Try once more with shorter delay
                await asyncio.sleep(1)
                continue
                
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            
        await asyncio.sleep(0.5)  # Poll every 500ms
    
    print("⏰ Polling timeout - task may have completed")

async def print_word_by_word_with_formatting(text: str, delay: float = 0.05):
    """Print text word by word while preserving natural paragraph breaks."""
    
    # Split by double newlines to preserve paragraph breaks
    paragraphs = text.split('\n\n')
    
    for paragraph_idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():  # Skip empty paragraphs
            # Split paragraph into sentences to add natural pauses
            sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
            
            for sentence_idx, sentence in enumerate(sentences):
                words = sentence.split()
                
                for word_idx, word in enumerate(words):
                    print(word, end="", flush=True)
                    
                    # Add period back if it was the end of a sentence
                    if word_idx == len(words) - 1 and sentence_idx < len(sentences) - 1:
                        print(".", end="", flush=True)
                    
                    # Add space after each word except the last in sentence
                    if word_idx < len(words) - 1:
                        print(" ", end="", flush=True)
                    
                    await asyncio.sleep(delay)
                
                # Add period at the end of last sentence if needed
                if sentence_idx == len(sentences) - 1 and not paragraph.endswith('.'):
                    print(".", end="", flush=True)
                
                # Small pause after each sentence
                await asyncio.sleep(delay * 3)
            
            # Add paragraph break (but not after the last paragraph)
            if paragraph_idx < len(paragraphs) - 1:
                print("\n\n", end="", flush=True)
                await asyncio.sleep(delay * 5)  # Longer pause between paragraphs

async def call_orchestrator_streaming(question: str):
    """Call orchestrator with non-blocking mode to get full history (fallback)."""
    
    payload = {
        "jsonrpc": "2.0",
        "id": "req-1", 
        "method": "message/send",
        "params": {
            "message": {
                "contextId": str(uuid4()),
                "messageId": str(uuid4()),
                "role": "user",
                "parts": [
                    {"type": "text", "text": question}
                ]
            },
            "configuration": {
                "blocking": False,
                "acceptedOutputModes": ["application/json"]
            }
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print("🔄 Sending request...")
            resp = await client.post(
                "http://127.0.0.1:8000/",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            
            response = resp.json()
            result = response.get("result", {})
            
            print("✅ Task completed! Here's what happened:\n")
            
            # Display the conversation history
            history = result.get("history", [])
            for i, entry in enumerate(history, 1):
                role = entry.get("role", "unknown")
                parts = entry.get("parts", [])
                
                for part in parts:
                    text = part.get("text", "")
                    if role == "user":
                        print(f"👤 User: {text}")
                    elif role == "agent":
                        print(f"🤖 Agent: {text}")
                print()
            
            # Display final artifacts
            artifacts = result.get("artifacts", [])
            if artifacts:
                print("🎯 FINAL RESULT:")
                print("=" * 50)
                for artifact in artifacts:
                    parts = artifact.get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        print(text)
                print("=" * 50)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    question = "What is your name? What are the adverse effects of eating tomatoes? What is the current time in New York City?"
    
    print("🚀 Testing REAL-TIME streaming (word by word)...")
    try:
        asyncio.run(call_orchestrator_real_time_streaming(question))
    except Exception as e:
        print(f"❌ Real-time streaming failed: {e}")
        print("\n📞 Falling back to batch mode...")
        asyncio.run(call_orchestrator_streaming(question))