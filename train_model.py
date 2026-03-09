from backend.model_utils import DEVICE, MODEL_PATH, export_model_snapshot, train_model


def main() -> None:
    print(f'Using device: {DEVICE}')
    model = train_model()
    snapshot_path = export_model_snapshot()
    print(model)
    print(f'Model stored at {MODEL_PATH}')
    print(f'Model snapshot stored at {snapshot_path}')


if __name__ == '__main__':
    main()