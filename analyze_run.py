"""
Analyze Training Run
====================

Tabular summary of all episodes from the JSON log files.

Usage:
    python analyze_run.py                    # Reads logs/ directory
    python analyze_run.py --log-dir logs     # Explicit log dir
    python analyze_run.py --ep-min 500       # Show from episode 500 onward
    python analyze_run.py --every 100        # Show every Nth episode only
    python analyze_run.py --section lang     # Show only language metrics
    python analyze_run.py --section core     # Show only curiosity/reward metrics
    python analyze_run.py --section comm     # Show only Phase 4 comm/memory metrics

Sections:
    all    Full table (default) — core + language + comm
    core   Prediction error, learning progress, confidence, reward
    lang   Vocab size, naming accuracy, naming loss, discrimination loss
    comm   Utterance rate, referral reward, joint reward, spatial memory
"""

import json
import os
import glob
import argparse
import numpy as np
from typing import List, Dict, Any


def load_logs(log_dir: str, ep_min: int = 0) -> List[Dict[str, Any]]:
    """Load all metric entries from JSON log files, sorted by episode."""
    files = sorted(glob.glob(os.path.join(log_dir, 'metrics_ep*.json')))
    entries = []
    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            for entry in data:
                if entry.get('episode', 0) >= ep_min:
                    entries.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue

    # Deduplicate by episode (keep last seen), then sort
    by_ep = {}
    for e in entries:
        by_ep[e['episode']] = e
    return sorted(by_ep.values(), key=lambda x: x['episode'])


def agent_ids(entries: List[Dict]) -> List[str]:
    """Sorted agent IDs as strings (JSON keys are always strings)."""
    if not entries:
        return []
    sample = entries[0]['agents']
    return sorted(sample.keys(), key=lambda x: int(x))


def print_core(entries: List[Dict], aids: List[str]):
    """Prediction error, learning progress, confidence, total reward."""
    n = len(aids)
    # Header
    agent_hdrs = '  '.join(
        f'{"A"+a+":err":>8} {"prog":>6} {"conf":>5} {"reward":>8}' for a in aids
    )
    print(f'\n{"=== CORE METRICS (curiosity / reward)":<60}')
    print(f'{"EP":>6} {"St":>2} {"Temp":>5}  {agent_hdrs}')
    print('-' * (28 + n * 32))

    for e in entries:
        ep    = e['episode']
        stage = e.get('stage', '?')
        temp  = e.get('temperature', 0.0)
        cols  = []
        for a in aids:
            d = e['agents'].get(a, e['agents'].get(int(a), {}))
            err    = d.get('avg_error', 0.0)
            prog   = d.get('avg_progress', 0.0)
            conf   = d.get('confidence', 0.0)
            reward = d.get('total_reward', 0.0)
            cols.append(f'{err:>8.4f} {prog:>6.4f} {conf:>5.3f} {reward:>8.1f}')
        print(f'{ep:>6} {stage:>2} {temp:>5.3f}  {"  ".join(cols)}')


def print_lang(entries: List[Dict], aids: List[str]):
    """Vocab size, naming accuracy, naming loss, discrimination loss."""
    n = len(aids)
    agent_hdrs = '  '.join(
        f'{"A"+a+":vocab":>7} {"namacc":>6} {"namloss":>7} {"discloss":>8}' for a in aids
    )
    global_hdr = f'{"glb_acc":>7} {"recent":>6} {"teach":>5} {"tests":>5}'
    print(f'\n{"=== LANGUAGE METRICS (grounding / discrimination)":<60}')
    print(f'{"EP":>6} {"St":>2}  {global_hdr}  {agent_hdrs}')
    print('-' * (24 + 30 + n * 34))

    for e in entries:
        ep    = e['episode']
        stage = e.get('stage', '?')
        lang  = e.get('language', {})
        g_acc  = lang.get('naming_accuracy', 0.0)
        g_rec  = lang.get('recent_accuracy', 0.0)
        teach  = lang.get('total_teaching_events', 0)
        tests  = lang.get('total_naming_tests', 0)
        global_col = f'{g_acc:>7.3f} {g_rec:>6.3f} {teach:>5d} {tests:>5d}'

        cols = []
        for a in aids:
            d = e['agents'].get(a, e['agents'].get(int(a), {}))
            vocab  = d.get('vocab_size', 0)
            namacc = d.get('naming_accuracy', 0.0)
            namloss= d.get('avg_naming_loss', 0.0)
            discloss=d.get('avg_discrimination_loss', 0.0)
            cols.append(f'{vocab:>7d} {namacc:>6.3f} {namloss:>7.4f} {discloss:>8.4f}')
        print(f'{ep:>6} {stage:>2}  {global_col}  {"  ".join(cols)}')


def print_comm(entries: List[Dict], aids: List[str]):
    """Utterance rate, property utterance rate, referral reward, joint reward, spatial memory, property vocab."""
    n = len(aids)
    agent_hdrs = '  '.join(
        f'{"A"+a+":utt%":>7} {"putt":>5} {"ref_r":>5} {"prop_r":>6} {"prop_app":>8} {"jnt_r":>5} {"mem":>3} {"pvoc":>4}' for a in aids
    )
    print(f'\n{"=== COMM METRICS (utterances / referral / spatial memory)":<60}')
    print(f'{"EP":>6} {"St":>2}  {agent_hdrs}  {"evt":>3} {"arr":>3}')
    print('-' * (12 + n * 54 + 9))

    for e in entries:
        ep    = e['episode']
        stage = e.get('stage', '?')
        cols  = []
        for a in aids:
            d = e['agents'].get(a, e['agents'].get(int(a), {}))
            utt_rate = d.get('utterance_rate', 0.0)
            putt     = d.get('property_utterance_rate', 0.0)
            ref_r    = d.get('referral_reward', 0.0)
            prop_r   = d.get('property_comm_reward', 0.0)
            prop_app = d.get('property_approach_reward', 0.0)
            jnt_r    = d.get('joint_reward', 0.0)
            mem      = d.get('memory_entries', 0)
            pvoc     = d.get('property_vocab_size', 0)
            cols.append(f'{utt_rate:>7.3f} {putt:>5.3f} {ref_r:>5.2f} {prop_r:>6.2f} {prop_app:>8.2f} {jnt_r:>5.2f} {mem:>3d} {pvoc:>4d}')
        evt = 1 if e.get('event_active', False) else 0
        arr = e.get('event_arrivals', 0)
        print(f'{ep:>6} {stage:>2}  {"  ".join(cols)}  {evt:>3d} {arr:>3d}')



def print_refgame(entries, aids):
    """Reference game outcomes and grammar signals (Stage 4 only)."""
    ref_entries = [e for e in entries if e.get('ref_game', {}).get('active', False)]

    print(f'\n{"=== REFERENCE GAME METRICS (Stage 4)":<60}')
    if not ref_entries:
        print('  No reference game episodes in this window.')
        return

    print(f'{"EP":>6} {"scout":>5} {"run":>3} {"target":>10} {"amb":>3} '
          f'{"outcome":>12} {"prop":>4} {"cp":>3} {"dist":>6}')
    print('-' * 60)

    for e in ref_entries:
        rg = e['ref_game']
        ep      = e['episode']
        scout   = rg.get('scout_id', '?')
        runner  = rg.get('runner_id', '?')
        target  = rg.get('target_key', '?')[:10]
        amb     = 'Y' if rg.get('target_is_ambiguous', False) else 'N'
        outcome = rg.get('outcome', 'timeout')[:12]
        prop    = 'Y' if rg.get('scout_used_property', False) else 'N'
        cprop   = 'Y' if rg.get('scout_used_correct_property', False) else 'N'
        dist    = rg.get('runner_min_distance') or 999.9
        print(f'{ep:>6} {str(scout):>5} {str(runner):>3} {target:>10} {amb:>3} '
              f'{outcome:>12} {prop:>4} {cprop:>3} {dist:>6.1f}')

    # Summary statistics
    n = len(ref_entries)
    n_amb     = sum(1 for e in ref_entries if e['ref_game'].get('target_is_ambiguous'))
    n_correct = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'correct')
    n_wrong   = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'wrong_variant')
    n_timeout = sum(1 for e in ref_entries if e['ref_game'].get('outcome') == 'timeout')
    n_prop    = sum(1 for e in ref_entries if e['ref_game'].get('scout_used_property'))
    n_cprop   = sum(1 for e in ref_entries if e['ref_game'].get('scout_used_correct_property'))
    dists     = [e['ref_game'].get('runner_min_distance') or 999.9 for e in ref_entries]
    avg_dist  = float(np.mean(dists))

    print(f'\n  Summary ({n} ref game episodes in window):')
    print(f'    Ambiguous targets: {n_amb:>3}/{n} ({100*n_amb/n:>4.0f}%)')
    print(f'    Correct:           {n_correct:>3}/{n} ({100*n_correct/n:>4.0f}%)')
    print(f'    Wrong variant:     {n_wrong:>3}/{n} ({100*n_wrong/n:>4.0f}%)')
    print(f'    Timeout:           {n_timeout:>3}/{n} ({100*n_timeout/n:>4.0f}%)')
    print(f'    Scout used prop:   {n_prop:>3}/{n} ({100*n_prop/n:>4.0f}%)')
    print(f'    Correct prop word: {n_cprop:>3}/{n} ({100*n_cprop/n:>4.0f}%)')
    print(f'    Avg min distance:  {avg_dist:>6.1f}')


def print_summary(entries: List[Dict], aids: List[str]):
    """High-level summary statistics across the full run."""
    if not entries:
        return
    first, last = entries[0], entries[-1]
    print(f'\n{"=== RUN SUMMARY":<60}')
    print(f'  Episodes:  {first["episode"]} -> {last["episode"]}  ({len(entries)} snapshots)')
    print(f'  Stages:    {first.get("stage","?")} -> {last.get("stage","?")}')
    print(f'  Temp:      {first.get("temperature",0):.3f} -> {last.get("temperature",0):.3f}')

    for a in aids:
        d0 = first['agents'].get(a, first['agents'].get(int(a), {}))
        d1 = last['agents'].get(a, last['agents'].get(int(a), {}))
        v0, v1 = d0.get('vocab_size', 0), d1.get('vocab_size', 0)
        e0, e1 = d0.get('avg_error', 0), d1.get('avg_error', 0)
        c0, c1 = d0.get('confidence', 0), d1.get('confidence', 0)
        na1    = d1.get('naming_accuracy', 0.0)
        nl1    = d1.get('avg_naming_loss', 0.0)
        dl1    = d1.get('avg_discrimination_loss', 0.0)
        words  = d1.get('words_known', [])
        pv1    = d1.get('property_vocab_size', 0)
        print(f'\n  Agent {a}:')
        direction = "down" if e1 < e0 else "UP"
        print(f'    Pred error:   {e0:.4f} -> {e1:.4f}  ({direction})')
        print(f'    Confidence:   {c0:.3f} -> {c1:.3f}')
        print(f'    Vocabulary:   {v0} -> {v1} words  {words}')
        print(f'    Naming acc:   {na1:.3f}   naming_loss={nl1:.4f}   disc_loss={dl1:.4f}')
        print(f'    Prop vocab:   {pv1} words')
        app_r = d1.get('property_approach_reward', 0.0)
        print(f'    Prop approach: {app_r:.2f}')

    # Event summary
    n_events = sum(1 for e in entries if e.get('event_active', False))
    total_arrivals = sum(e.get('event_arrivals', 0) for e in entries)
    avg_arrivals = total_arrivals / n_events if n_events > 0 else 0.0
    print(f'\n  Food Events:')
    print(f'    Episodes with event: {n_events} / {len(entries)}')
    print(f'    Total arrivals:      {total_arrivals}')
    print(f'    Avg arrivals per event: {avg_arrivals:.2f}')

    # Reference game summary (Stage 4)
    rg_entries = [e for e in entries if e.get('ref_game', {}).get('active', False)]
    if rg_entries:
        n_rg = len(rg_entries)
        n_correct = sum(1 for e in rg_entries if e['ref_game'].get('outcome') == 'correct')
        n_cprop   = sum(1 for e in rg_entries if e['ref_game'].get('scout_used_correct_property'))
        print(f'\n  Reference Game (Stage 4):')
        print(f'    Games played:      {n_rg} / {len(entries)}')
        print(f'    Success rate:      {100*n_correct/n_rg:.0f}%')
        print(f'    Correct prop use:  {100*n_cprop/n_rg:.0f}%')


def main():
    parser = argparse.ArgumentParser(description='Analyze curious_agents training run')
    parser.add_argument('--log-dir', default='logs', help='Log directory (default: logs)')
    parser.add_argument('--ep-min', type=int, default=0, help='Skip episodes below this')
    parser.add_argument('--every', type=int, default=1,
                        help='Show every Nth episode snapshot (default: 1 = all)')
    parser.add_argument('--section', choices=['all', 'core', 'lang', 'comm', 'refgame'],
                        default='all',
                        help='Which metrics to display (default: all)')
    args = parser.parse_args()

    entries = load_logs(args.log_dir, ep_min=args.ep_min)
    if not entries:
        print(f'No log entries found in {args.log_dir!r} (ep >= {args.ep_min})')
        return

    # Thin by episode stride
    if args.every > 1:
        entries = [e for i, e in enumerate(entries) if i % args.every == 0
                   or i == len(entries) - 1]

    aids = agent_ids(entries)

    print_summary(entries, aids)

    if args.section in ('all', 'core'):
        print_core(entries, aids)

    if args.section in ('all', 'lang'):
        print_lang(entries, aids)

    if args.section in ('all', 'comm'):
        print_comm(entries, aids)

    if args.section in ('all', 'refgame'):
        print_refgame(entries, aids)


if __name__ == '__main__':
    main()
