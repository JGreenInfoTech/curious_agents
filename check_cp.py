import torch

base = r'C:\Users\Johnathan\ClaudeResearch\curious_agents\checkpoints'
cp = torch.load(f'{base}\\checkpoint_ep5000.pt', map_location='cpu', weights_only=False)

print("=== Checkpoint ep5000 ===")
print(f"Episode: {cp['episode']}, Stage: {cp['stage']}, Temp: {cp['temperature']}")
print()

# Environment objects
env = cp['env_state']
print(f"Env objects ({len(env['objects'])}):")
for name, obj in env['objects'].items():
    pos = obj.get('position', '?')
    print(f"  {name}: pos={pos}, keys={list(obj.keys())[:5]}")
print()

# Agents
agents = cp['agents']
print(f"Agents ({len(agents)}):")
for i, a in enumerate(agents):
    print(f"  Agent {i}: type={type(a).__name__}, ", end="")
    if isinstance(a, dict):
        vocab = a.get('vocabulary', {})
        print(f"vocab={len(vocab)}, pos={a.get('position')}")
    elif hasattr(a, 'keys'):
        print(f"keys={list(a.keys())[:5]}")
    else:
        # Might be state_dict or something else
        print(f"repr={str(a)[:100]}")
print()

# Teacher
teacher = cp['teacher']
print(f"Teacher: type={type(teacher).__name__}")
if isinstance(teacher, dict):
    print(f"  Keys: {list(teacher.keys())}")
    wm = teacher.get('word_memories', {})
    print(f"  word_memories ({len(wm)}): {list(wm.keys())[:5]}")
