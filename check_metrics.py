import json

with open('logs/metrics_ep8000.json') as f:
    data = json.load(f)

last = data[-1]
print("=== Episode", last['episode'], "Stage", last['stage'], "Temp", last['temperature'], "===")

print("\n--- Language ---")
for k, v in last['language'].items():
    print(f"  {k}: {v}")

print("\n--- Agents ---")
for aid, ainfo in last['agents'].items():
    print(f"\n  Agent {aid}:")
    if isinstance(ainfo, dict):
        for k, v in ainfo.items():
            if isinstance(v, list) and len(v) > 3:
                print(f"    {k}: list[{len(v)}] last3={v[-3:]}")
            else:
                print(f"    {k}: {v}")

# Also check first few episodes of Phase 2
print("\n\n=== First Phase 2 episode ===")
first_p2 = data[0]
print("Episode", first_p2['episode'], "Stage", first_p2['stage'])
print("Language:", first_p2['language'])
for aid, ainfo in first_p2['agents'].items():
    print(f"Agent {aid}:")
    if isinstance(ainfo, dict):
        for k, v in ainfo.items():
            if isinstance(v, list) and len(v) > 3:
                print(f"  {k}: list[{len(v)}] last3={v[-3:]}")
            else:
                print(f"  {k}: {v}")
