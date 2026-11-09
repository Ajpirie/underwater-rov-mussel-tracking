"""Optional demonstration of independent tracking observations; no ROV needed."""
import math
import threading

from GUI.control_panel import ControlPanel
from Integration.live_input import publish_observation


def demo_observations(stop_event):
    """Generate two asynchronous, swaying mussel tracks."""
    i = 0
    while not stop_event.is_set():
        t = 0.5 * i
        publish_observation(5, -0.10 + 0.02 * math.sin(t), 0.02, 0.40)
        publish_observation(6, -0.05 + 0.02 * math.sin(t + 0.4), 0.04, 0.42)
        i += 1
        stop_event.wait(0.5)


if __name__ == '__main__':
    app = ControlPanel()
    app.vars['mode'].set('Run')
    app._set_mode()
    app._start()
    event = threading.Event()
    worker = threading.Thread(target=demo_observations, args=(event,), daemon=True)
    worker.start()
    try:
        app.mainloop()
    finally:
        event.set()
