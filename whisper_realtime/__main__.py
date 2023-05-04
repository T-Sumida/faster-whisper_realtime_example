import argparse
from typing import Optional, Union

from whisper_realtime.app import App
from whisper_realtime.audio import get_input_devices


def get_args() -> argparse.Namespace:
    """引数取得

    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true", help="デバイスリストを表示して終了する."
    )
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        input_dev_list, _ = get_input_devices()
        for dev_name, dev_id in input_dev_list.items():
            print(f"{dev_name} : {dev_id}")
        parser.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser],
    )
    parser.add_argument("--mic", type=Optional[Union[str, int]], default=None, help="マイクデバイス名 or マイクID")
    parser.add_argument(
        "--model", type=str, default="large-v2", help="モデルサイズ"
    )
    parser.add_argument("--cuda", help="GPUを利用する", action="store_true")
    parser.add_argument("--compute_type", type=str, default="float16", help="実行形式を指定, [float16(default), int8, int8_float16]")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    args = parser.parse_args(remaining)
    return args


def main():
    args = get_args()
    app = App(
        args.model,
        args.cuda,
        args.compute_type,
        args.mic,
    )
    app.run()


if __name__ == "__main__":
    main()
