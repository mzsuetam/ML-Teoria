import os
import datetime
import time

def autosave():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.system('git add .')
    os.system(f'git commit -m "{timestamp}"')
    print(f'Autosaved at {timestamp}')

if __name__ == '__main__':
    try:
        while True:
            autosave()
            time.sleep(60)
    except KeyboardInterrupt:
        quit()

