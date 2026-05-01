import sys, traceback
try:
    exec(open("main.py").read(), {"__name__": "__main__"})
except SystemExit as e:
    print(f"SystemExit: {e.code}")
except:
    traceback.print_exc()
