import threading, time

from common.model_load import load_once


def test_concurrent_loads_build_one_instance():
    built = []

    def factory():
        time.sleep(.05)  # a slow load: the second thread must wait, not build its own copy
        built.append(object())
        return built[-1]

    got = []
    threads = [threading.Thread(target=lambda: got.append(load_once(('test', 'slow-model'), factory))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1 and all(g is built[0] for g in got)
