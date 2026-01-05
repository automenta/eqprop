#!/usr/bin/env python3
"""
Comprehensive smoke test for all new tracks (34-40).
"""

from validation.tracks import new_tracks
from validation.core import Verifier

print('='*60)
print('COMPREHENSIVE SMOKE TEST: New Tracks 34-40')
print('='*60)
print()

v = Verifier(quick_mode=True, seed=42)

results = {}
for track_id in [34, 35, 36, 37, 38, 40]:
    track_fn = new_tracks.NEW_TRACKS[track_id]
    print(f'[Track {track_id}] Testing...')
    try:
        result = track_fn(v)
        status_icon = '✅' if result.status == 'pass' else '⚠️' if result.status == 'partial' else '❌'
        print(f'{status_icon} Track {track_id}: {result.name}')
        print(f'   Status: {result.status.upper()} | Score: {result.score}/100')
        print(f'   Time: {result.time_seconds:.1f}s')
        results[track_id] = result
    except Exception as e:
        print(f'❌ Track {track_id} FAILED: {e}')
        import traceback
        traceback.print_exc()
        results[track_id] = None

print('\n' + '='*60)
print('SMOKE TEST SUMMARY')
print('='*60)

passed = sum(1 for r in results.values() if r and r.status == 'pass')
partial = sum(1 for r in results.values() if r and r.status == 'partial')
failed = sum(1 for r in results.values() if not r or r.status == 'fail')

print(f'✅ Passed:  {passed}/6')
print(f'⚠️  Partial: {partial}/6')
print(f'❌ Failed:  {failed}/6')

if passed + partial == 6:
    print('\n🎉 ALL TRACKS EXECUTED SUCCESSFULLY!')
else:
    print(f'\n⚠️ {failed} track(s) failed')

print()
