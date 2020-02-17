import os
import json
from transcribe import transcribe_video


if __name__ == "__main__":
    data_dir = 'data'
    out_dir = 'transcriptions'
    f_names = os.listdir(data_dir)
    for f_name in f_names:
        path = os.path.join(data_dir, f_name)

        result = transcribe_video(path)

        base_name, _ = os.path.splitext(f_name)
        out_name = base_name + '.json'
        out_path = os.path.join(out_dir, out_name)

        with open(out_path, 'w') as f:
            f.write(json.dumps(result))

