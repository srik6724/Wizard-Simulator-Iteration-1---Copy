#!/usr/bin/env python3
"""
Training Monitor - Track AI learning progress
Run this in a separate terminal while training
"""

import time
import os
import sys
from collections import defaultdict
import json

def parse_match_log(filepath):
    """Parse match log and extract statistics"""
    if not os.path.exists(filepath):
        return None
    
    stats = {
        'total_matches': 0,
        'team1_wins': 0,
        'team2_wins': 0,
        'avg_rounds': [],
        'recent_winners': []  # Last 20 matches
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Split by match
        matches = content.split('MATCH #')
        stats['total_matches'] = len(matches) - 1  # First split is header
        
        for match in matches[1:]:  # Skip first (before any match)
            lines = match.split('\n')
            
            # Find winner
            for line in lines:
                if 'WINNER:' in line:
                    if 'Team 1' in line:
                        stats['team1_wins'] += 1
                        stats['recent_winners'].append('T1')
                    elif 'Team 2' in line:
                        stats['team2_wins'] += 1
                        stats['recent_winners'].append('T2')
                    
                    # Extract round count
                    if 'Final Round:' in line:
                        try:
                            round_num = int(line.split('Final Round:')[1].strip())
                            stats['avg_rounds'].append(round_num)
                        except:
                            pass
        
        # Keep only last 20 winners
        stats['recent_winners'] = stats['recent_winners'][-20:]
        
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None
    
    return stats

def check_server_health():
    """Check if PPO server is responsive"""
    try:
        import requests
        response = requests.get('http://localhost:5000/health', timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def print_header():
    """Print monitoring header"""
    print("=" * 80)
    print(" " * 20 + "WIZARD101 AI TRAINING MONITOR")
    print("=" * 80)
    print()

def print_stats(stats, health):
    """Print formatted statistics"""
    os.system('clear' if os.name != 'nt' else 'cls')
    print_header()
    
    # Match Statistics
    if stats:
        total = stats['total_matches']
        t1_wins = stats['team1_wins']
        t2_wins = stats['team2_wins']
        
        print(f"📊 MATCH STATISTICS")
        print(f"{'─' * 80}")
        print(f"Total Matches:      {total}")
        print(f"Team 1 Wins:        {t1_wins} ({t1_wins/max(1,total)*100:.1f}%)")
        print(f"Team 2 Wins:        {t2_wins} ({t2_wins/max(1,total)*100:.1f}%)")
        
        if stats['avg_rounds']:
            avg_rounds = sum(stats['avg_rounds']) / len(stats['avg_rounds'])
            print(f"Average Rounds:     {avg_rounds:.1f}")
        
        # Recent trend
        if stats['recent_winners']:
            recent = stats['recent_winners']
            t1_recent = recent.count('T1')
            t2_recent = recent.count('T2')
            print(f"\nRecent Form (last {len(recent)}):  ", end="")
            for winner in recent[-10:]:  # Show last 10
                print(f"{'🔵' if winner == 'T1' else '🔴'}", end="")
            print(f"  T1:{t1_recent} T2:{t2_recent}")
        
        print()
    
    # Server Health
    if health:
        print(f"🖥️  SERVER STATUS")
        print(f"{'─' * 80}")
        print(f"Status:             {health.get('status', 'unknown').upper()}")
        print(f"Device:             {health.get('device', 'unknown')}")
        print(f"State Dimension:    {health.get('state_dim', 'N/A')}")
        print()
        
        print(f"Team 1 Agent:")
        print(f"  Experiences:      {health.get('team1_experiences', 0)}")
        print(f"  Total Updates:    {health.get('team1_updates', 0)}")
        print(f"  W/L Record:       {health.get('team1_wl', 'N/A')}")
        print()
        
        print(f"Team 2 Agent:")
        print(f"  Experiences:      {health.get('team2_experiences', 0)}")
        print(f"  Total Updates:    {health.get('team2_updates', 0)}")
        print(f"  W/L Record:       {health.get('team2_wl', 'N/A')}")
        print()
    else:
        print(f"⚠️  SERVER STATUS: OFFLINE or UNREACHABLE")
        print(f"{'─' * 80}")
        print(f"Cannot connect to http://localhost:5000")
        print(f"Make sure PPO server is running:")
        print(f"  python ppo_agent.py")
        print()
    
    # Learning Assessment
    if stats and stats['total_matches'] > 0:
        print(f"🎓 LEARNING ASSESSMENT")
        print(f"{'─' * 80}")
        
        matches = stats['total_matches']
        t2_winrate = stats['team2_wins'] / max(1, matches) * 100
        
        if matches < 50:
            phase = "Early Exploration"
            assessment = "🔍 AI is exploring randomly - too early to judge"
        elif matches < 150:
            phase = "Basic Learning"
            if t2_winrate > 55:
                assessment = "✅ AI is learning! Shows improvement"
            else:
                assessment = "⏳ Still learning basics - give it time"
        elif matches < 300:
            phase = "Tactical Development"
            if t2_winrate > 60:
                assessment = "🎯 Good progress! Developing strategy"
            else:
                assessment = "⚠️  Slower than expected - check rewards"
        else:
            phase = "Advanced Training"
            if t2_winrate > 70:
                assessment = "🏆 Excellent! Near-optimal play"
            elif t2_winrate > 60:
                assessment = "✓ Solid performance - continuing to improve"
            else:
                assessment = "⚠️  May need tuning - check reward signals"
        
        print(f"Training Phase:     {phase} ({matches} matches)")
        print(f"Assessment:         {assessment}")
        
        # Recommendations
        if matches < 100:
            print(f"\n💡 Recommendation: Keep training - need at least 100 matches")
        elif matches >= 100 and t2_winrate < 52:
            print(f"\n⚠️  Warning: No clear learning after {matches} matches")
            print(f"   Consider: Check reward function, reduce learning rate")
        
        print()
    
    print(f"{'─' * 80}")
    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Press Ctrl+C to exit")
    print(f"{'─' * 80}")

def main():
    """Main monitoring loop"""
    print_header()
    print("Starting monitor... Will update every 5 seconds")
    print()
    time.sleep(2)
    
    try:
        while True:
            stats = parse_match_log('match_log.txt')
            health = check_server_health()
            print_stats(stats, health)
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Training summary saved to training_summary.txt")
        
        # Save final summary
        stats = parse_match_log('match_log.txt')
        if stats:
            with open('training_summary.txt', 'w') as f:
                f.write(f"Training Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(f"Total Matches: {stats['total_matches']}\n")
                f.write(f"Team 1 Wins: {stats['team1_wins']} ({stats['team1_wins']/max(1,stats['total_matches'])*100:.1f}%)\n")
                f.write(f"Team 2 Wins: {stats['team2_wins']} ({stats['team2_wins']/max(1,stats['total_matches'])*100:.1f}%)\n")
                
                if stats['avg_rounds']:
                    avg = sum(stats['avg_rounds']) / len(stats['avg_rounds'])
                    f.write(f"Average Round Length: {avg:.1f}\n")
        
        sys.exit(0)

if __name__ == '__main__':
    # Check if requests is available
    try:
        import requests
    except ImportError:
        print("Installing required package: requests")
        os.system('pip install requests')
        import requests
    
    main()