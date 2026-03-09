from backend.model_utils import DEVICE, export_model_snapshot, get_model


def main() -> None:
    print(f'Using device: {DEVICE}')
    get_model()
    snapshot_path = export_model_snapshot()
    print(f'Exported neural snapshot to {snapshot_path}')


if __name__ == '__main__':
    main()
