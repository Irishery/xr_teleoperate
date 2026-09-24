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
    parser.add_argument(
        "--include-commands", action="store_true",
        help="also observe Dex3 command topics; does not publish commands",
    )
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowState_

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    topics = [
        ("rt/lowstate", LowState_, None),
        ("rt/dex3/left/state", HandState_, "motor_state"),
        ("rt/dex3/right/state", HandState_, "motor_state"),
    ]
    if args.include_commands:
        topics.extend([
            ("rt/dex3/left/cmd", HandCmd_, "motor_cmd"),
            ("rt/dex3/right/cmd", HandCmd_, "motor_cmd"),
        ])
    stats = {name: {"count": 0, "q": None, "received_at": None} for name, _, _ in topics}
    lock = threading.Lock()
    subscribers = []
    collecting = False

    def make_handler(name, motor_field):
        def on_state(message):
            with lock:
                if not collecting:
                    return
                stats[name]["count"] += 1
                stats[name]["received_at"] = time.monotonic()
                if motor_field is not None:
                    stats[name]["q"] = [float(m.q) for m in getattr(message, motor_field)]
        return on_state

    try:
        for name, message_type, motor_field in topics:
            subscriber = ChannelSubscriber(name, message_type)
            subscribers.append(subscriber)
            subscriber.Init(make_handler(name, motor_field))

        print(
            f"Observing domain 0 on {args.network_interface} for {args.seconds:g}s; "
            "no command publishers.", flush=True,
        )
        with lock:
            collecting = True
        time.sleep(args.seconds)
        with lock:
            collecting = False
            finished_at = time.monotonic()
        for name, _, _ in topics:
            state = stats[name]
            print(f"{name}: messages={state['count']}")
            if state["q"] is not None:
                print("  last q (rad): " + ", ".join(f"{q:.4f}" for q in state["q"]))
                print(f"  last sample age (s): {finished_at - state['received_at']:.3f}")

        if args.include_commands:
            print("Thumb comparison (rad), latest samples, not time-synchronized:")
            for side in ("left", "right"):
                command = stats[f"rt/dex3/{side}/cmd"]["q"]
                actual = stats[f"rt/dex3/{side}/state"]["q"]
                if command is None or actual is None:
                    print(f"  {side}: command or state unavailable")
                    continue
                for index in range(min(3, len(command), len(actual))):
                    print(
                        f"  {side} thumb{index}: cmd={command[index]:+.4f} "
                        f"state={actual[index]:+.4f} "
                        f"state-cmd={actual[index] - command[index]:+.4f}"
                    )

        missing = [name for name, state in stats.items() if state["count"] == 0]
        if missing:
            print("No samples received during this window: " + ", ".join(missing))
            print("This does not distinguish absent publishers from DDS delivery/type issues.")
            return 1
        print("Received messages on all requested topics (zero joint angles count as valid samples).")
        return 0
    finally:
        for subscriber in subscribers:
            subscriber.Close()


if __name__ == "__main__":
    raise SystemExit(main())
