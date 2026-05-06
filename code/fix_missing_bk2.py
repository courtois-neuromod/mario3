import sys, glob
import pandas as pd
from pandas.errors import EmptyDataError
import retro
from retro.enums import State
import numpy as np

GAME_KEYS = 'udlrabxy'
GAME_KEYS_AND_TTL=GAME_KEYS

MARIO3_KEY_SET = "ya__udlrb"
MARIO3_FRAMERATE = 60.099826520671044
KEY_READOUT_DELAY = 0.000

def keypresses_to_replay(bk2_path, duration, key_presses, bk2_output_path, validate=False):
    movie = retro.Movie(bk2_path)
#    movie.step()
    emulator = None
    try:
        game = movie.get_game()
        emulator = retro.make(game, state=State.NONE, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=False)
        framerate = emulator.em.get_screen_rate()
        emulator.initial_state = movie.get_state()
        emulator.reset()
        emulator.record_movie(bk2_output_path)
        #buttons = [(b[:1].lower() if b else '_') for b in emulator.buttons]
        buttons = MARIO3_KEY_SET
        frame_idx = 0
        
        timestamp = 0
        _done = False
        while timestamp < duration+100/framerate: #allow for small jitter
            if _done:
                print(f"DONE BEFORE THE END: {frame_idx}")
            timestamp = frame_idx / framerate
            keys = [
                any([ (kp[0]==k and kp[2]<=frame_idx and kp[4]>=frame_idx) for kp in key_presses])
                for k in buttons
            ]
            if validate:
                movie.step()
                movie_keys = [movie.get_key(k, 0) for k in range(len(buttons))]
                for k in range(len(buttons)):
                    assert movie_keys[k] == keys[k]
            frame, rew, _done, truncate, info = emulator.step(keys)
            frame_idx += 1
        if not _done:
            print("DONE CONDITION not met", frame_idx)
        if abs(timestamp-duration) > 5/framerate:
            print("DURATION DO NOT MATCH", duration, timestamp)
    finally:
        emulator.stop_record()
        if emulator is not None:
            emulator.close()
        movie.close()



for evt_path in sorted(glob.glob('/unf/eyetracker/neuromod/mario3/sourcedata/sub-03/ses-*/*task-mario*_events.tsv')):
    evt = pd.read_csv(evt_path, delimiter='\t')
    game_events = evt[evt.trial_type=='gym-retro_game']
    repeated_evts = game_events.stim_file.duplicated(keep=False)
    if not repeated_evts.any():
        continue
    log_path = evt_path.split('_task-')[0] + '.log'
    try:
        log = pd.read_csv(log_path, delimiter='\t', names=['time','evt_type','evt_str'])
    except EmptyDataError:
        print("cannot read {log_path}: empty")
        continue
    if not len(log):
        print("empty log {log_path}")
        continue
    
    for bk2_idx, bk2_path in enumerate(game_events.stim_file[repeated_evts]):
        bk2_path_rel = bk2_path.split('sourcedata/')[1]
        evt_log_start = (log.evt_str == f"VideoGame: recording movie in /scratch/neuromod/data/mario3/sourcedata/{bk2_path_rel}").idxmax()

        #run_start = (log.evt_str[evt_log_start:] == "fMRI TTL 0").idxmax()
        
        #level_start = (log.evt_str[evt_log_start:] == "level step: 0").idxmax()
        level_start = np.where(log.evt_str[evt_log_start:] == "level step: 0")[0][bk2_idx] + evt_log_start
        
        t_start = log.loc[level_start].time
        level_stop = (log.evt_str[level_start:].str.contains('stopped at')).idxmax()
        t_stop = log.loc[level_stop].time
        duration = t_stop-t_start
        print(bk2_path, t_start, level_stop, duration)

        key_presses = []
        key_pressed = {k:None for k in GAME_KEYS_AND_TTL}
        steps = [(0, 0.)]


        ttl_diffs = []
        for idx, line in log.loc[level_start:level_stop].iterrows():
            key = line.evt_str.split(': ')[-1]
            if key == '5':
                if 'Keypress' in line.evt_str:
                    ttl_start = line.time
                elif 'Keyrelease' in line.evt_str:
                    ttl_diffs.append(ttl_start - line.time)
        ttl_diff = min(ttl_diffs)
        ttl_diff = ttl_diffs[0]
        if abs(ttl_diff) < 1:
            ttl_diff = 0
        
        for idx, line in log.loc[level_start:level_stop].iterrows():
            key = line.evt_str.split(': ')[-1]
            if 'Keypress' in line.evt_str and key in GAME_KEYS_AND_TTL:
                press_time = float(line.time - t_start)
                step = steps[-1][0] + int( np.ceil((press_time - steps[-1][1] - KEY_READOUT_DELAY +.008) * MARIO3_FRAMERATE))
                key_pressed[key] = (press_time, step)
            elif 'Keyrelease' in line.evt_str and key in GAME_KEYS_AND_TTL:
                if key_pressed[key]:
                    release_time = float(line.time + ttl_diff - t_start)
                    release_step = steps[-1][0] + int( np.ceil((release_time - steps[-1][1] - KEY_READOUT_DELAY) * MARIO3_FRAMERATE))
                    key_presses.append((key, *key_pressed[key], release_time, release_step))
                    key_pressed[key] = None
                else:
                    print(f'error: no press found for release for {line.time}', bk2_path, log_path)
            elif 'level step' in line.evt_str:
                steps.append((int(line.evt_str.split(': ')[1]), float(line.time-t_start)))
        
        bk2_output_path = bk2_path_rel.replace('.bk2', f"_recon{bk2_idx:02d}.bk2")
        keypresses_to_replay('/unf/eyetracker/neuromod/mario3/sourcedata/'+bk2_path_rel, duration, key_presses, bk2_output_path, validate=bk2_idx==100)
