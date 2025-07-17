'''
Example of using the Audio Registry feature through the MCP server.

This script demonstrates how to:
1. List all audio entries in the registry
2. Load an audio file and check that it's registered
3. Separate vocals and accompaniment, which creates new registry entries
4. Load an entry from the registry and apply effects
5. Remove entries from the registry

Note: This example only works when connected to a running MCP server.
'''

import sys
import os
import json
import requests

def call_mcp_tool(tool_name, params=None):
    """Call a tool on the MCP server and return the response."""
    url = "http://localhost:8000/invoke"
    data = {
        "tool_name": tool_name,
        "parameters": params or {}
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error calling {tool_name}: {response.text}")
        return None

def pretty_print(obj):
    """Print an object in pretty JSON format."""
    print(json.dumps(obj, indent=2))

def main():
    print("\n=== Audio Registry Example ===\n")
    
    # 1. Clear the registry to start fresh
    print("Clearing the registry...")
    result = call_mcp_tool("clear_audio_registry")
    pretty_print(result)
    
    # 2. Check that the registry is empty
    print("\nListing registry entries (should be empty):")
    result = call_mcp_tool("list_audio_registry")
    pretty_print(result)
    
    # 3. Load an audio file
    input_file = "input.mp3"
    if not os.path.exists(input_file):
        print(f"\nPlease place an audio file named '{input_file}' in the current directory")
        return
    
    print(f"\nLoading audio file: {input_file}")
    result = call_mcp_tool("load_audio_file", {"file_path": input_file})
    pretty_print(result)
    
    # 4. Check that the audio is registered
    print("\nListing registry entries (should have 'current' entry):")
    result = call_mcp_tool("list_audio_registry")
    pretty_print(result)
    
    # 5. Separate vocals and accompaniment
    print("\nSeparating vocals from the audio:")
    result = call_mcp_tool("separate_vocals")
    pretty_print(result)
    
    # 6. Check the registry again - should now have vocals and accompaniment
    print("\nListing registry entries (should have vocals and accompaniment):")
    result = call_mcp_tool("list_audio_registry")
    pretty_print(result)
    
    # 7. Load vocals from registry
    print("\nLoading vocals from registry:")
    result = call_mcp_tool("load_audio_from_registry", {"audio_id": "vocals"})
    pretty_print(result)
    
    # 8. Apply reverb to the vocals
    print("\nApplying reverb to vocals:")
    result = call_mcp_tool("apply_reverb", {"room_size": 0.8, "damping": 0.5})
    pretty_print(result)
    
    # 9. Save the processed vocals
    output_file = "output/vocals_reverb.wav"
    print(f"\nSaving processed vocals to {output_file}:")
    result = call_mcp_tool("save_audio", {"file_path": output_file})
    pretty_print(result)
    
    # 10. Remove vocals from registry
    print("\nRemoving vocals from registry:")
    result = call_mcp_tool("remove_from_audio_registry", {"audio_id": "vocals"})
    pretty_print(result)
    
    # 11. Final registry check
    print("\nFinal registry entries (vocals should be gone):")
    result = call_mcp_tool("list_audio_registry")
    pretty_print(result)
    
    print("\n=== Example Complete ===\n")
    print(f"Check the output directory for processed files.")

if __name__ == "__main__":
    main()
