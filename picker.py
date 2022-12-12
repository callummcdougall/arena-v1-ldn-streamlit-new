#!/usr/bin/env python

from pick import pick
import subprocess
import os

folders = [f for f in os.listdir('.') if os.path.isdir(f)]
max_chapter = max([int(f[2:]) for f in folders if f.startswith('ch')])

title = 'Choose chapter (press SPACE to mark, ENTER to continue):'
options = [f'CH{n}'for n in range(max_chapter + 1)]

try:
    selected = pick(options, title, multiselect=True, min_selection_count=1, indicator='→', default_index=max_chapter)
except KeyboardInterrupt:
    exit()

commands = [f"streamlit run ch{n}/Home.py" for (name, n) in selected]
print(f'Opening chapters {", ".join([name for (name, n) in selected])}...')

processes = [subprocess.Popen(command.split(), stdout=subprocess.PIPE) for command in commands]

# catch interrupt and kill all processes
try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    for p in processes:
        p.kill()
    print("\n\nBye! 👋")