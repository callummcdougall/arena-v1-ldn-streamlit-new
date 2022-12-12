from pick import pick
import subprocess

title = 'Choose chapter (press SPACE to mark, ENTER to continue):'
options = [f'CH{n}'for n in range(7)]
selected = pick(options, title, multiselect=True, min_selection_count=1)

# selected = [('CH0', "0"), ('CH1', 1), ('CH2', 2), ('CH3', 3), ('CH4', 4), ('CH5', 5), ('CH6', 6)]
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