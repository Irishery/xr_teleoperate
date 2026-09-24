"""Observe G1/Dex3 DDS state without creating command publishers."""

import argparse
import math
import threading
import time


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--seconds", type=positive_seconds, default=10.0)
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    topics = (
        ("rt/lowstate", LowState_),
        ("rt/dex3/left/state", HandState_),
        ("rt/dex3/right/state", HandState_),
    )
    stats = {name: {"count": 0, "q": None} for name, _ in topics}
    lock = threading.Lock()
    subscribers = []
    collecting = False

    def make_handler(name):
        def on_state(message):
            with lock:
                if not collecting:
                    return
                stats[name]["count"] += 1
                if name != "rt/lowstate":
                    stats[name]["q"] = [float(m.q) for m in message.motor_state]
        return on_state

    try:
        for name, message_type in topics:
            subscriber = ChannelSubscriber(name, message_type)
            subscribers.append(subscriber)
            subscriber.Init(make_handler(name))

        print(
            f"Observing domain 0 on {args.network_interface} for {args.seconds:g}s; "
            "no command publishers.", flush=True,
        )
        with lock:
            collecting = True
        time.sleep(args.seconds)
        with lock:
            collecting = False
        for name, _ in topics:
            state = stats[name]
            print(f"{name}: messages={state['count']}")
            if state["q"] is not None:
                print("  last q (rad): " + ", ".join(f"{q:.4f}" for q in state["q"]))

        missing = [name for name, state in stats.items() if state["count"] == 0]
        if missing:
            print("No samples received during this window: " + ", ".join(missing))
            print("This does not distinguish absent publishers from DDS delivery/type issues.")
            return 1
        print("Received state on all three topics (zero joint angles count as valid state).")
        return 0
    finally:
        for subscriber in subscribers:
            subscriber.Close()


if __name__ == "__main__":
    raise SystemExit(main())
